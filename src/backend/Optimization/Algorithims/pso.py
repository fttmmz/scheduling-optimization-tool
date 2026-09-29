

import random
import copy
import statistics

import numpy as np

from backend.models.models import ScheduleItem
from backend.Optimization.constraints import (
    get_valid_timeslots,
    get_viable_rooms,
    instructor_occupancy_ids,
    finalize_schedule,
    section_needs,
)


def build_section_candidates(sections, rooms, timeslots):
    """(section, needs, viable_rooms, valid_timeslots) per section.

    Candidates are only generated along the axes a section actually uses, so
    thesis/supervision sections no longer consume rooms and roomless studio and
    graduate-lab sections stop competing for them.
    """
    candidates = []
    for section in sections:
        needs = section_needs(section)
        candidates.append((
            section,
            needs,
            get_viable_rooms(section, rooms) if needs.room else [],
            get_valid_timeslots(section, timeslots) if needs.time else [],
        ))
    return candidates
from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
)
from backend.Optimization.Algorithims.grasp import scan_candidates
from backend.Optimization.benchmark_seeds import seed_all, seed_for_run

#parameters
N_PARTICLES = 10
ITERATIONS = 14
NO_IMPROVE_LIMIT = 5


CONSTRUCTION_ROOM_SAMPLE = 8
CONSTRUCTION_TIMESLOT_SAMPLE = 8


RESCUE_ROOM_SAMPLE = 10
RESCUE_TIMESLOT_SAMPLE = 10

C1 = 2.05
C2 = 2.05
_PHI = C1 + C2
assert _PHI > 4, "c1 + c2 must exceed 4 for the constriction factor to be defined"
CHI = 2.0 / abs(2 - _PHI - (_PHI * _PHI - 4 * _PHI) ** 0.5)  # ~0.7298 (Clerc & Kennedy 2002)
V_CLAMP = 4.0


POSITION_RANGE = 9.0
VELOCITY_INIT_RANGE = 0.5



# Decode: continuous priority vector -> discrete timetable

def _bounded_candidates(
    section, rooms, timeslots,
    occupied_instructors, occupied_rooms,
    room_sample_size, timeslot_sample_size,
    valid_timeslot_cache=None,
):
    """
    Scan a bounded random sample first, falling back to the full
    room x timeslot space only if the sample turns up nothing feasible
    -- the same bounded-then-fallback pattern grasp.py uses, so a section
    is never left unscheduled just because sampling missed its only
    valid slot.
    """
    room_sample = (
        rooms if len(rooms) <= room_sample_size
        else random.sample(rooms, room_sample_size)
    )
    timeslot_sample = (
        timeslots if len(timeslots) <= timeslot_sample_size
        else random.sample(timeslots, timeslot_sample_size)
    )

    candidates = scan_candidates(
        section, room_sample, timeslot_sample,
        occupied_instructors, occupied_rooms,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    if not candidates:
        candidates = scan_candidates(
            section, rooms, timeslots,
            occupied_instructors, occupied_rooms,
            valid_timeslot_cache=valid_timeslot_cache,
        )
    return candidates


def choose_best_assignment(
    section, rooms, timeslots,
    occupied_instructors, occupied_rooms,
    valid_timeslot_cache=None,
):
    
    candidates = _bounded_candidates(
        section, rooms, timeslots,
        occupied_instructors, occupied_rooms,
        CONSTRUCTION_ROOM_SAMPLE, CONSTRUCTION_TIMESLOT_SAMPLE,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    if not candidates:
        return None, None

    best = max(candidates, key=lambda c: c[0])
    return best[1], best[2]


def decode(position, section_candidates, valid_timeslot_cache):
    
    order = np.argsort(-position)
    n = len(section_candidates)
    schedule = [None] * n
    occupied_rooms = set()
    occupied_instructors = set()

    for idx in order:
        section, needs, viable_rooms, valid_ts = section_candidates[idx]

        if not needs.room and not needs.time:
            room = timeslot = None
        elif not needs.room:
            # Time only: first slot that leaves the instructor free.
            room = None
            section_instructors = instructor_occupancy_ids(section)
            timeslot = next(
                (
                    ts for ts in valid_ts
                    if all((instructor_id, ts.id) not in occupied_instructors
                           for instructor_id in section_instructors)
                ),
                None,
            )
            # No free slot: leave it unscheduled rather than force a clash. An
            # unscheduled section costs 2.0, a hard conflict 3.0, so falling
            # back to valid_ts[0] bought a placement at more than it was worth.
            if timeslot is None and valid_ts and not section_instructors:
                timeslot = valid_ts[0]
        elif not needs.time:
            room = viable_rooms[0] if viable_rooms else None
            timeslot = None
        else:
            room, timeslot = choose_best_assignment(
                section, viable_rooms, valid_ts,
                occupied_instructors, occupied_rooms,
                valid_timeslot_cache=valid_timeslot_cache,
            )

        item = ScheduleItem(
            course_id=section.course.id,
            course_name=section.course.name,
            course_type=section.course.type,
            course_dept=section.course.dept,
            capacity=section.capacity,
            instructor_id=section.instructor_id,
            room_id=room.id if room else None,
            timeslot_id=timeslot.id if timeslot else None,
            section=str(section.no),
            instructor_ids=section.instructor_ids,
            pattern_index=section.pattern_index,
            # Classification is keyed on level -- without it a detached item
            # looks like an ordinary room+time lecture.
            level=getattr(section.course, "level", None),
            course_class=getattr(section.course, "course_class", None),
        )
        schedule[idx] = item

        # A roomless section still occupies its instructor's time.
        if timeslot is not None:
            if room is not None:
                occupied_rooms.add((room.id, timeslot.id))
            for instructor_id in instructor_occupancy_ids(section):
                occupied_instructors.add((instructor_id, timeslot.id))

    return schedule


# ============================================================
# Ejection-chain repair for sections decode() left unscheduled
# ============================================================
def _build_occupancy_maps(schedule):
    room_owner = {}
    instructor_owner = {}
    occupied_rooms = set()
    occupied_instructors = set()

    for idx, item in enumerate(schedule):
        if item.room_id is not None and item.timeslot_id is not None:
            key = (item.room_id, item.timeslot_id)
            room_owner[key] = idx
            occupied_rooms.add(key)
            for instructor_id in instructor_occupancy_ids(item):
                ikey = (instructor_id, item.timeslot_id)
                instructor_owner[ikey] = idx
                occupied_instructors.add(ikey)

    return room_owner, instructor_owner, occupied_rooms, occupied_instructors


def _place(item, idx, room, timeslot, room_owner, instructor_owner, occupied_rooms, occupied_instructors):
    item.room_id = room.id
    item.timeslot_id = timeslot.id
    key = (room.id, timeslot.id)
    room_owner[key] = idx
    occupied_rooms.add(key)
    for instructor_id in instructor_occupancy_ids(item):
        ikey = (instructor_id, timeslot.id)
        instructor_owner[ikey] = idx
        occupied_instructors.add(ikey)


def _evict(item, room_owner, instructor_owner, occupied_rooms, occupied_instructors):
    key = (item.room_id, item.timeslot_id)
    room_owner.pop(key, None)
    occupied_rooms.discard(key)
    for instructor_id in instructor_occupancy_ids(item):
        ikey = (instructor_id, item.timeslot_id)
        instructor_owner.pop(ikey, None)
        occupied_instructors.discard(ikey)
    item.room_id = None
    item.timeslot_id = None


def rescue_unscheduled(schedule, section_candidates, valid_timeslot_cache):

    room_owner, instructor_owner, occupied_rooms, occupied_instructors = (
        _build_occupancy_maps(schedule)
    )

   
    unscheduled = []
    for idx, item in enumerate(schedule):
        needs = section_candidates[idx][1]
        if needs.room and item.room_id is None:
            unscheduled.append(idx)
        elif needs.time and item.timeslot_id is None:
            unscheduled.append(idx)

    for idx in unscheduled:
        item = schedule[idx]
        section, needs, viable_rooms, valid_ts = section_candidates[idx]
        # None for supervision and for items with no instructor, matching what
        # _build_occupancy_maps / _place / _evict record.
        item_instructors = instructor_occupancy_ids(item)

        # The ejection chain trades (room, timeslot) pairs, so it only applies
        # to sections that need both.
        if not (needs.room and needs.time):
            continue


        candidates = _bounded_candidates(
            section, viable_rooms, valid_ts, set(), set(),
            RESCUE_ROOM_SAMPLE, RESCUE_TIMESLOT_SAMPLE,
            valid_timeslot_cache=valid_timeslot_cache,
        )
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0], reverse=True)

        for _, room, timeslot in candidates:
            room_blocker = room_owner.get((room.id, timeslot.id))
            instr_blocker = next(
                (owner for owner in (
                    instructor_owner.get((instructor_id, timeslot.id))
                    for instructor_id in item_instructors
                ) if owner is not None),
                None,
            )

            if room_blocker is None and instr_blocker is None:
                # Genuinely free right now -- take it.
                _place(item, idx, room, timeslot, room_owner, instructor_owner,
                       occupied_rooms, occupied_instructors)
                break

            if room_blocker is not None and instr_blocker is not None and room_blocker != instr_blocker:
                continue  # two different blockers -- outside depth-1 scope

            blocker_idx = room_blocker if room_blocker is not None else instr_blocker
            if blocker_idx == idx:
                continue

            blocker = schedule[blocker_idx]
            b_section, b_needs, b_rooms, b_ts = section_candidates[blocker_idx]
            if not (b_needs.room and b_needs.time):
                continue  # can't relocate a blocker that doesn't use both axes

            trial_rooms = occupied_rooms - {(blocker.room_id, blocker.timeslot_id)}
            trial_instructors = occupied_instructors
            blocker_instructors = instructor_occupancy_ids(blocker)
            if blocker_instructors:
                trial_instructors = occupied_instructors - {
                    (instructor_id, blocker.timeslot_id)
                    for instructor_id in blocker_instructors
                }

            if room_blocker is not None:
                # Room-block: the blocker sits at exactly (room, timeslot),
                # which is also what we just freed above -- re-mark it
                # occupied for the blocker's OWN search, or a tie would let
                # choose_best_assignment hand it right back (a no-op "move"
                # that silently defeats the rescue).
                b_ts_search = b_ts
                trial_rooms = trial_rooms | {(room.id, timeslot.id)}
            else:
                # Instructor-block: the conflict is with `timeslot` itself,
                # in any room -- excluding just (room, timeslot) wouldn't
                # stop choose_best_assignment from placing the blocker in a
                # DIFFERENT room at the same clashing timeslot.
                b_ts_search = [t for t in b_ts if t.id != timeslot.id]

            new_room, new_timeslot = choose_best_assignment(
                b_section, b_rooms, b_ts_search, trial_instructors, trial_rooms,
                valid_timeslot_cache=valid_timeslot_cache,
            )
            if new_room is None:
                continue  # blocker has nowhere else to go -- leave it alone

            _evict(blocker, room_owner, instructor_owner, occupied_rooms, occupied_instructors)
            _place(blocker, blocker_idx, new_room, new_timeslot, room_owner, instructor_owner,
                   occupied_rooms, occupied_instructors)
            _place(item, idx, room, timeslot, room_owner, instructor_owner,
                   occupied_rooms, occupied_instructors)
            break

    return schedule



def _refine(schedule, section_candidates, rooms, timeslots, sections, valid_timeslot_cache):
    rescue_unscheduled(schedule, section_candidates, valid_timeslot_cache)
    fitness = calculate_fitness(
        schedule, rooms, sections=sections, timeslots=timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    return schedule, fitness


# ============================================================
# PSO Algorithm
# ============================================================
def pso_schedule(sections, timeslots, rooms, valid_timeslot_cache=None, section_candidates=None, seed=None):
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    if section_candidates is None:
        section_candidates = build_section_candidates(sections, rooms, timeslots)

    n = len(sections)
    rng = np.random.default_rng(seed)

    positions = rng.uniform(0.0, POSITION_RANGE, size=(N_PARTICLES, n))
    velocities = rng.uniform(-VELOCITY_INIT_RANGE, VELOCITY_INIT_RANGE, size=(N_PARTICLES, n))
    pbest_pos = positions.copy()
    pbest_score = np.full(N_PARTICLES, -1.0)

    gbest_pos = positions[0].copy()
    gbest_sched = None
    gbest_score = -1.0
    no_improve_count = 0

    for _ in range(ITERATIONS):
        improved_this_iter = False

        for i in range(N_PARTICLES):
            sched = decode(positions[i], section_candidates, valid_timeslot_cache)
            score = calculate_fitness(
                sched, rooms, sections=sections, timeslots=timeslots,
                valid_timeslot_cache=valid_timeslot_cache,
            )

            if score > pbest_score[i]:
                pbest_score[i] = score
                pbest_pos[i] = positions[i].copy()

            if score > gbest_score:
               
                refined_sched, refined_score = _refine(
                    sched, section_candidates, rooms, timeslots, sections, valid_timeslot_cache
                )
                gbest_score = refined_score
                gbest_pos = positions[i].copy()
                gbest_sched = copy.deepcopy(refined_sched)
                improved_this_iter = True

        no_improve_count = 0 if improved_this_iter else no_improve_count + 1
        if no_improve_count >= NO_IMPROVE_LIMIT:
            break

        # Constriction-factor velocity + position update (Clerc & Kennedy 2002)
        r1 = rng.random((N_PARTICLES, n))
        r2 = rng.random((N_PARTICLES, n))
        cognitive = C1 * r1 * (pbest_pos - positions)
        social = C2 * r2 * (gbest_pos - positions)
        velocities = CHI * (velocities + cognitive + social)
        np.clip(velocities, -V_CLAMP, V_CLAMP, out=velocities)
        positions = positions + velocities

    # Siblings (meeting blocks of one section) belong in one room. The
    # placement loops have no cross-block state, so unify afterwards; the
    # pass only ever moves a block into a room that is free at its hour.
    return finalize_schedule(
        gbest_sched, sections, timeslots, valid_timeslot_cache, rooms=rooms
    ), gbest_score


def pso_runs(sections, timeslots, rooms, num_runs=30, seeds=None):
    fitness_scores = []
    best_overall_schedule = None
    best_overall_fitness = -1

    print(f"\n=== PSO Algorithm: {num_runs} runs ===")
    print(f"Total sections: {len(sections)}, rooms: {len(rooms)}, timeslots: {len(timeslots)}")

    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    section_candidates = build_section_candidates(sections, rooms, timeslots)

    for run in range(num_runs):
        # This module draws from TWO generators: the global `random` (candidate sampling in
        # scan_candidates) and its own np.random.default_rng (the particles). A local
        # generator ignores the global seed, so the seed has to be threaded in as well --
        # seeding only one of the two leaves half the run unreproducible.
        run_seed = seed_all(seed_for_run(run, seeds))

        best_schedule, score = pso_schedule(
            sections, timeslots, rooms,
            valid_timeslot_cache=valid_timeslot_cache,
            section_candidates=section_candidates,
            seed=run_seed,
        )

        fitness_scores.append(score)

        if score > best_overall_fitness:
            best_overall_fitness = score
            best_overall_schedule = copy.deepcopy(best_schedule)

        scheduled = sum(
            1 for item in best_schedule
            if item.room_id is not None and item.timeslot_id is not None
        )

        print(f"Run {run + 1:2d} (seed {run_seed}): Fitness = {score:.4f} | Scheduled = {scheduled}")

    best_score = max(fitness_scores)
    worst_score = min(fitness_scores)
    avg_score = sum(fitness_scores) / len(fitness_scores)
    std_dev = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0

    print("\n=== Results ===")
    print(f"Best fitness:       {best_score:.4f}")
    print(f"Worst fitness:      {worst_score:.4f}")
    print(f"Average fitness:    {avg_score:.4f}")
    print(f"Standard deviation: {std_dev:.4f}")

    return best_overall_schedule
