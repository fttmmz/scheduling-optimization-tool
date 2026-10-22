"""ML-guided greedy: every allowed (room, timeslot) scored, the best one taken.

A one-pass construction heuristic. Sections are placed in GRASP's difficulty
order (fewest options first). For each one, EVERY viable room x valid
timeslot that passes the hard rules against the occupancy so far is scored,
and the highest score wins. There is no sampling and no random choice among
the top few; randomness only breaks exact ties in the score.

The score is the one GRASP V3 uses (grasp_ml_experiments/v3_scoring_trace.py):

    score = ROOM_SCORE_SCALE * room model score      (0..1 per section)
          + 50 if the hour fits the course-type guideline
          - 50 * time_soft_penalty (e.g. an undergraduate class in the evening)

The room model score is read from an offline table that an XGBoost ranker
wrote (ml/export_room_scores.py); no xgboost is needed here. A section the
table does not cover, or a missing table (Render, synthetic tests), falls back
to GRASP's formula room score, so the algorithm always runs.

SCORE_SOURCES chooses the table (env ML_GREEDY_SCORES, or score_source=):
    hybrid          the model trained on hybrid's schedules (default; V3's)
    hybrid_xfit     the same model cross-fitted: every course scored by a model
                    that never saw it in training
    manual          the model trained on the registrar's manual schedule
    manual+hybrid   manual + 0.01 x hybrid, as GRASP V2 combines them
    formula         no model at all: GRASP's formula room score

Measured (real snapshot, 2026-10-05, seeds 1-30): see q_handover.md section 5.
Why it beats GRASP with the same score: GRASP scores only a random 15 rooms x
15 timeslots per section and then picks at random among its top 3.
"""
import csv
import os
import random
import statistics
from functools import lru_cache
from pathlib import Path

from backend.models.models import ScheduleItem
from backend.Optimization.constraints import (
    OverlapSet,
    finalize_schedule,
    instructor_occupancy_ids,
    room_soft_penalty,
    seats_needed,
    time_soft_penalty,
)
from backend.Optimization.Algorithims import grasp
from backend.Optimization.benchmark_seeds import seed_all, seed_for_run
from backend.Optimization.evaluation import (
    build_timeslot_guideline_cache,
    calculate_fitness,
)

SOFT_SCORE_SCALE = grasp.SOFT_SCORE_SCALE              # 50 points per penalty unit
TIMESLOT_GUIDELINE_PENALTY = grasp.TIMESLOT_GUIDELINE_PENALTY
# Points between a section's worst and best room by the model: one penalty
# unit, as in V3 (50 and 150 measured the same there; 400 let rooms override
# hours).
ROOM_SCORE_SCALE = 50.0

SCORE_SOURCES = {
    "hybrid": ("hybrid/room_scores.csv",),
    "hybrid_xfit": ("hybrid/room_scores_xfit.csv",),
    "manual": ("manual/room_scores.csv",),
    "manual+hybrid": ("manual/room_scores.csv", "hybrid/room_scores.csv"),
    "formula": (),
}
DEFAULT_SCORE_SOURCE = os.environ.get("ML_GREEDY_SCORES", "hybrid")
# manual+hybrid: the hybrid model only reorders rooms the manual model scores
# within 0.01 of each other (GRASP V2's TRACE_SHARE).
SECONDARY_SHARE = 0.01


# ============================================================
# Room model scores (offline table)
# ============================================================
def _find_data_dir():
    """$ML_DATA_DIR if set, else ml/data found by walking up from this file."""
    if os.environ.get("ML_DATA_DIR"):
        return Path(os.environ["ML_DATA_DIR"])
    relative = Path("ml") / "data"
    for parent in Path(__file__).resolve().parents:
        if (parent / relative).exists():
            return parent / relative
    return Path(__file__).resolve().parents[4] / relative


def _read_table(path):
    """{section_key: {room_id: score}}, rescaled to 0..1 within each section --
    a ranker's score only compares rooms of the same section."""
    raw = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw.setdefault(row["m_section_id"], {})[str(row["m_room_id"])] = float(row["score"])
    table = {}
    for key, by_room in raw.items():
        lo, hi = min(by_room.values()), max(by_room.values())
        span = hi - lo
        table[key] = {room: (s - lo) / span if span > 0 else 0.0 for room, s in by_room.items()}
    return table


@lru_cache(maxsize=None)
def load_room_scores(source):
    """{section_key: {room_id: 0..1}} for a SCORE_SOURCES name; {} when the
    table is missing (the formula then scores every section)."""
    if source not in SCORE_SOURCES:
        raise ValueError(f"Unknown score source {source!r}; choose from {list(SCORE_SOURCES)}")
    paths = [_find_data_dir() / name for name in SCORE_SOURCES[source]]
    if not paths:
        return {}
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"Warning: room score table not found ({missing[0]}); ml_greedy uses the formula")
        return {}
    primary = _read_table(paths[0])
    if len(paths) == 1:
        return primary
    secondary = _read_table(paths[1])
    return {
        key: {room: s + SECONDARY_SHARE * secondary.get(key, {}).get(room, 0.0)
              for room, s in by_room.items()}
        for key, by_room in primary.items()
    }


def section_key(section):
    """The score tables' m_section_id: "<course id>|<section no>|<pattern index>"."""
    return f"{section.course.id}|{section.no}|{section.pattern_index}"


# ============================================================
# Scoring: a room part plus a time part, each computed once per section
# ============================================================
def formula_room_score(section, room):
    """GRASP's room score: the soft penalty, plus a snug-fit bonus."""
    score = -room_soft_penalty(section, room) * SOFT_SCORE_SCALE
    remaining_capacity = room.capacity - seats_needed(section)
    if remaining_capacity >= 0:
        score += grasp.CAPACITY_FIT_BONUS - min(remaining_capacity, grasp.CAPACITY_FIT_BONUS)
    return score


def room_score(section, room, room_scores):
    by_room = room_scores.get(section_key(section))
    if by_room is None:
        return formula_room_score(section, room)
    return ROOM_SCORE_SCALE * by_room.get(str(room.id), 0.0)


def time_score(section, timeslot, guideline_ids):
    score = SOFT_SCORE_SCALE * TIMESLOT_GUIDELINE_PENALTY if timeslot.id in guideline_ids else 0.0
    return score - time_soft_penalty(section, timeslot) * SOFT_SCORE_SCALE


# ============================================================
# Construction
# ============================================================
def _best_pair(section, viable_rooms, valid_ts, room_scores, guideline_ids,
               occupied_rooms, occupied_instructors, rng):
    """The highest-scoring free (room, timeslot), or (None, None). Exact ties
    are broken uniformly at random (reservoir sampling), so only ties draw
    random numbers."""
    room_parts = [room_score(section, room, room_scores) for room in viable_rooms]
    instructors = instructor_occupancy_ids(section)
    best, best_score, ties = (None, None), None, 0

    for timeslot in valid_ts:
        # Same test as passes_hard_constraints with occupancy maps: every
        # instructor free and the room free. A busy teacher rules out the
        # whole hour, so it is checked once rather than once per room.
        if any((i, timeslot.id) in occupied_instructors for i in instructors):
            continue
        t_part = time_score(section, timeslot, guideline_ids)
        for room, r_part in zip(viable_rooms, room_parts):
            if (room.id, timeslot.id) in occupied_rooms:
                continue
            score = r_part + t_part
            if best_score is None or score > best_score:
                best, best_score, ties = (room, timeslot), score, 1
            elif score == best_score:
                ties += 1
                if rng.random() < 1.0 / ties:
                    best = (room, timeslot)
    return best


def construct_ml_greedy(section_candidates, room_scores, valid_timeslot_cache, rng):
    schedule = []
    occupied_rooms = OverlapSet()
    occupied_instructors = OverlapSet()

    for section, needs, viable_rooms, valid_ts in section_candidates:
        if not needs.room and not needs.time:
            room = timeslot = None
        elif not needs.room:
            # Time only, as GRASP: the first slot that leaves EVERY instructor
            # free; unscheduled rather than a forced clash.
            room = None
            section_instructors = instructor_occupancy_ids(section)
            timeslot = next(
                (ts for ts in valid_ts
                 if all((i, ts.id) not in occupied_instructors for i in section_instructors)),
                None,
            )
            if timeslot is None and valid_ts and not section_instructors:
                timeslot = valid_ts[0]
        elif not needs.time:
            # Room only: the best room by the same room score. max() keeps the
            # first on a tie, i.e. get_viable_rooms' own order.
            room = (max(viable_rooms, key=lambda r: room_score(section, r, room_scores))
                    if viable_rooms else None)
            timeslot = None
        else:
            guideline_ids = valid_timeslot_cache.get(
                (getattr(section.course, "id", None), str(getattr(section, "no", None))), set()
            )
            room, timeslot = _best_pair(
                section, viable_rooms, valid_ts, room_scores, guideline_ids,
                occupied_rooms, occupied_instructors, rng,
            )

        schedule.append(ScheduleItem.from_section(
            section,
            room_id=room.id if room else None,
            timeslot_id=timeslot.id if timeslot else None,
        ))
        # A roomless class still holds its teachers' hour; supervision holds
        # nothing (instructor_occupancy_ids returns ()).
        if timeslot is not None:
            if room is not None:
                occupied_rooms.add((room.id, timeslot.id))
            for instructor_id in instructor_occupancy_ids(section):
                occupied_instructors.add((instructor_id, timeslot.id))

    return schedule


def ml_greedy_schedule(sections, timeslots, rooms, seed=None, score_source=None,
                       valid_timeslot_cache=None, section_candidates=None):
    """One finalized schedule."""
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    if section_candidates is None:
        section_candidates = grasp.order_sections_by_difficulty(
            grasp.build_section_candidates(sections, rooms, timeslots)
        )
    room_scores = load_room_scores(score_source or DEFAULT_SCORE_SOURCE)
    schedule = construct_ml_greedy(section_candidates, room_scores, valid_timeslot_cache,
                                   random.Random(seed))
    return finalize_schedule(schedule, sections, timeslots, valid_timeslot_cache, rooms=rooms)


def ml_greedy_runs(sections, timeslots, rooms, num_runs=1, seeds=None, score_source=None):
    """Best of num_runs schedules. Runs differ only in how exact score ties are
    broken, so a handful of runs is plenty."""
    source = score_source or DEFAULT_SCORE_SOURCE
    room_scores = load_room_scores(source)
    covered = sum(1 for s in sections if section_key(s) in room_scores)
    print(f"\n=== ML greedy: {num_runs} runs | room scores: {source} "
          f"({covered}/{len(sections)} sections covered) ===")

    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    section_candidates = grasp.order_sections_by_difficulty(
        grasp.build_section_candidates(sections, rooms, timeslots)
    )
    best_schedule, best_fitness, scores = None, -1.0, []
    for run in range(num_runs):
        run_seed = seed_all(seed_for_run(run, seeds))
        schedule = ml_greedy_schedule(
            sections, timeslots, rooms, seed=run_seed, score_source=source,
            valid_timeslot_cache=valid_timeslot_cache, section_candidates=section_candidates,
        )
        fitness = calculate_fitness(schedule, rooms, sections=sections, timeslots=timeslots,
                                    valid_timeslot_cache=valid_timeslot_cache)
        scores.append(fitness)
        print(f"Run {run + 1:2d} (seed {run_seed}): Fitness = {fitness:.4f}")
        if fitness > best_fitness:
            best_schedule, best_fitness = schedule, fitness

    if len(scores) > 1:
        print(f"Best {max(scores):.4f} | mean {statistics.mean(scores):.4f} | "
              f"std {statistics.stdev(scores):.4f}")
    return best_schedule
