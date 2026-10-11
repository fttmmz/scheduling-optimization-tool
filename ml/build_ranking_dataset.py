"""Build learning-to-rank datasets for the candidate scorer.

Two files, one per decision, because a room and an hour are different choices
with different features:

    ml/data/room_candidates.csv       one group per section that needs a ROOM
    ml/data/timeslot_candidates.csv   one group per section that needs a TIME

A group is the candidate list the scheduler actually hands GRASP (and PSO,
which reuses GRASP's scorer) for that section -- get_viable_rooms() and
get_valid_timeslots(), exactly as build_section_candidates() calls them. Every
candidate is scored in isolation: room_soft_penalty() and time_soft_penalty()
depend only on the (section, candidate) pair, never on the rest of the
schedule, so no run needs instrumenting.

Reads the local OFFLINE snapshot in tests/fixtures/ (gitignored -- create it
with tests/snapshot_refresh.py), the same data loader.load_section_details()
pulls from Supabase (schedule_id=1). Nothing here
imports backend.database.db. See ml/DATA_DICTIONARY.md for every column.

Run from the repo root:

    .venv311/Scripts/python.exe ml/build_ranking_dataset.py
    .venv311/Scripts/python.exe ml/build_ranking_dataset.py --room-limit all
"""
import argparse
import csv
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend.models.models import (  # noqa: E402
    Room, Section, Timeslot, build_sections, group_rows_by_pattern, instructors_in,
    section_info_index,
)
from backend.Optimization import constraints as C  # noqa: E402
from backend.Optimization.constraints import rooms as room_rules  # noqa: E402
from backend.Optimization.Algorithims import grasp  # noqa: E402
from backend.Optimization.evaluation import build_timeslot_guideline_cache  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
OUT_DIR = ROOT / "ml" / "data"

DAY_CODES = ("U", "M", "T", "W", "R", "F")


# ── Labels ───────────────────────────────────────────────────────────────────
# XGBoost's ranking objectives take INTEGER graded relevance, higher = better
# (rank:ndcg with the default ndcg_exp_gain=true caps it at 31; rank:map wants
# 0/1). The 0-4 scale is the convention of the public LTR benchmarks (MSLR-WEB,
# Yahoo LTR). The raw penalty is kept alongside in m_soft_penalty so a different
# binning is a one-line change rather than a rebuild.
#
# Room bands follow the weight structure in constraints/rooms.py (room_type
# 4.0, campus 3.0, capacity 2.0 + 4.0 x overflow, department 1.0):
#
#   4  penalty == 0        perfect fit
#   3  (0, 1]              only the department courtesy is bent
#   2  (1, 3]              one mid rule: campus alone, or a small (<=25%) overflow
#   1  (3, 6]              room type wrong, or two mid rules together
#   0  > 6                 several serious rules broken at once
ROOM_LABEL_BANDS = ((0.0, 4), (1.0, 3), (3.0, 2), (6.0, 1))


def room_label(penalty):
    for upper, grade in ROOM_LABEL_BANDS:
        if penalty <= upper:
            return grade
    return 0


def timeslot_label(penalty):
    """Binary. time_soft_penalty() only ever returns 0.0 or one constant per
    section (3.0 supervision-in-daytime, 2.0 teaching-in-evening), so there are
    no finer grades to express -- inventing them would be noise."""
    return 1 if penalty == 0.0 else 0


# ── Loading ──────────────────────────────────────────────────────────────────

def _read(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_data():
    """Sections, rooms, real timeslots, and each section's MANUAL placement.

    Sections are built exactly as models.build_sections() does, but walked by
    hand so the source room/timeslot of each meeting pattern -- which
    build_sections deliberately discards -- can be kept as metadata. The ids
    are asserted equal to build_sections() so the two cannot drift.
    """
    rows = _read("schedule_1.json")
    rooms = [Room(r) for r in _read("rooms.json")]
    # Registrar facts (campus code, enrolment) -- optional, as in the loader.
    info = (section_info_index(_read("section_info.json"))
            if (FIXTURES / "section_info.json").exists() else {})
    # Same filter as loader.load_timeslots(): drop the TBA bucket and the
    # zero-duration placeholders.
    timeslots = [
        ts for ts in (Timeslot(t) for t in _read("timeslots.json"))
        if not C.is_placeholder_timeslot(ts)
    ]

    sections, manual = [], []
    for section_key, patterns in group_rows_by_pattern(rows).items():
        for pattern_index, (key, pattern_rows) in enumerate(patterns.items()):
            sections.append(Section(
                pattern_rows[0],
                instructor_ids=instructors_in(pattern_rows),
                pattern_index=pattern_index,
                pattern_count=len(patterns),
                info=info.get(section_key),
            ))
            manual.append((key[0], key[1]))  # pattern_key = (room, timeslot, cap)

    assert [s.id for s in sections] == [s.id for s in build_sections(rows, info)]
    return sections, rooms, timeslots, manual


# ── Features ─────────────────────────────────────────────────────────────────

def _suffix_kind(section):
    suffix = C._section_suffix(section.no)
    return suffix if suffix in {"L", "X", "Y", "T"} else ("none" if not suffix else "other")


def _time_preference(section):
    if C.prefers_evening(section):
        return "evening"
    if C.prefers_daytime(section):
        return "daytime"
    return "neutral"


def section_features(section, needs, group_size):
    """Shared by both files: everything known about the section before any
    candidate is looked at."""
    return {
        "f_course_type": section.course.type,
        "f_level": section.course.level or "",
        "f_course_class": section.course.course_class or "",
        "f_needs": C.needs_to_label(needs),
        "f_enrollment": C.seats_needed(section),
        "f_campus_code": section.campus or "",
        "f_course_dept": str(section.course.dept),
        "f_is_supervision": int(C.is_supervision(section)),
        "f_time_preference": _time_preference(section),
        "f_n_instructors": len(section.instructor_ids),
        "f_n_occupying_instructors": len(C.instructor_occupancy_ids(section)),
        "f_pattern_count": section.pattern_count,
        "f_section_suffix": _suffix_kind(section),
        "f_section_campus": C.section_campus(section),
        "f_required_room_type": C.get_required_room_type(section.course.type) or "none",
        "f_is_combined_course": int(C.combined_group_key(section.course.id) is not None),
        "f_group_size": group_size,
    }


def section_meta(section, needs):
    return {
        "m_section_id": f"{section.course.id}|{section.no}|{section.pattern_index}",
        "m_course_id": section.course.id,
        "m_section_no": section.no,
        "m_pattern_index": section.pattern_index,
        "m_sibling_key": f"{section.course.id}|{section.no}",
        "m_course_dept": section.course.dept,
    }


def _lab_access(section, room):
    """How the room relates to the section's department, as the lab rules see it."""
    if not room.dept_id:
        return "open"
    if room.dept_id == section.course.dept:
        return "own"
    if room_rules._may_use_lab(section.course.dept, room):
        return "shared"
    return "other"


def room_features(section, room):
    enrollment, capacity = C.seats_needed(section), room.capacity
    known = capacity is not None and capacity > 0
    overflow = (
        min(1.0, (enrollment - capacity) / capacity)
        if known and enrollment is not None and enrollment > capacity else 0.0
    )
    room_campus = C.get_building_campus(room.building)
    return {
        "f_room_type": room.type,
        "f_room_capacity": capacity,
        "f_capacity_unknown": int(not known),
        "f_capacity_slack": (capacity - enrollment) if known else "",
        "f_capacity_utilisation": round(enrollment / capacity, 4) if known else "",
        "f_capacity_overflow_ratio": round(overflow, 4),
        "f_room_type_match": int(C.is_room_type_match(section, room)),
        "f_room_campus": room_campus,
        "f_campus_match": int(C.is_campus_match(section, room)),
        "f_room_has_dept": int(bool(room.dept_id)),
        "f_dept_match": int(C.is_department_match(section, room)),
        # Identity, not rules: which building and whose room. The registrar's
        # habits (a department's usual building) live here; the soft rules
        # cannot express them.
        "f_room_building": room.building,
        "f_room_dept": str(room.dept_id or ""),
        "f_room_access": _lab_access(section, room),
    }


def timeslot_features(ts):
    start = C._parse_time_to_minutes(ts.start)
    days = ts.day or ""
    feats = {
        "f_day_pattern": days,
        "f_n_days": len(days),
        "f_start_minute": start,
        "f_end_minute": C._parse_time_to_minutes(ts.end),
        "f_duration_minutes": C._timeslot_duration(ts),
        "f_is_evening": int(C.is_evening_timeslot(ts)),
    }
    for code in DAY_CODES:
        feats[f"f_day_{code}"] = int(code in days)
    return feats


# ── The current scorer, as a baseline ────────────────────────────────────────
# grasp.score_candidate() scores a (room, timeslot) PAIR, and its terms are
# additive per axis, so each file carries its own axis's share. Summing the two
# parts reproduces score_candidate() exactly -- checked in _check_baseline().

def grasp_room_score(section, room):
    score = -C.room_soft_penalty(section, room) * grasp.SOFT_SCORE_SCALE
    remaining = room.capacity - C.seats_needed(section)
    if remaining >= 0:
        score += grasp.CAPACITY_FIT_BONUS - min(remaining, grasp.CAPACITY_FIT_BONUS)
    return score


def grasp_time_score(section, ts, guideline_cache):
    score = 0.0
    if ts.id in guideline_cache.get((section.course.id, str(section.no)), set()):
        score += grasp.SOFT_SCORE_SCALE * grasp.TIMESLOT_GUIDELINE_PENALTY
    return score - C.time_soft_penalty(section, ts) * grasp.SOFT_SCORE_SCALE


def _check_baseline(pairs, guideline_cache):
    for section, room, ts in pairs:
        whole = grasp.score_candidate(section, room, ts, valid_timeslot_cache=guideline_cache)
        parts = grasp_room_score(section, room) + grasp_time_score(section, ts, guideline_cache)
        assert abs(whole - parts) < 1e-9, (section.id, room.id, ts.id, whole, parts)


# ── Build ────────────────────────────────────────────────────────────────────

def candidate_lists(sections, rooms, timeslots, room_limit):
    """(needs, ranked rooms, ranked timeslots) per section -- the lists the
    scheduler hands its placement loop."""
    out = []
    for section in sections:
        needs = C.section_needs(section)
        out.append((
            needs,
            C.get_viable_rooms(section, rooms, limit=room_limit) if needs.room else [],
            C.get_valid_timeslots(section, timeslots) if needs.time else [],
        ))
    return out


def demand_counts(candidates):
    """How many sections compete for each room and each timeslot.

    Static contention, computable from the candidate lists alone -- no run
    needed. The soft penalty ignores contention entirely, so formula labels
    never reflect it; run-trace labels do (a room every section wants is often
    taken), and without these the model would have no feature to learn that
    from. 'Perfect' counts only the sections for which it is a zero-penalty
    fit, i.e. the ones that will actually fight over it first.
    """
    room_any, room_perfect, slot_any, slot_perfect = Counter(), Counter(), Counter(), Counter()
    for section, (_needs, ranked_rooms, ranked_slots) in candidates:
        for room in ranked_rooms:
            room_any[room.id] += 1
            if C.room_soft_penalty(section, room) == 0.0:
                room_perfect[room.id] += 1
        for ts in ranked_slots:
            slot_any[ts.id] += 1
            if C.time_soft_penalty(section, ts) == 0.0:
                slot_perfect[ts.id] += 1
    return room_any, room_perfect, slot_any, slot_perfect


def build(room_limit):
    sections, rooms, timeslots, manual = load_data()
    guideline_cache = build_timeslot_guideline_cache(sections, timeslots)
    candidates = candidate_lists(sections, rooms, timeslots, room_limit)
    room_any, room_perfect, slot_any, slot_perfect = demand_counts(zip(sections, candidates))

    room_rows, time_rows = [], []
    stats = Counter()
    by_needs = Counter()
    baseline_pairs = []

    for index, (section, (manual_room, manual_ts), (needs, ranked_rooms, ranked_slots)) in \
            enumerate(zip(sections, manual, candidates)):
        by_needs[C.needs_to_label(needs)] += 1
        meta = section_meta(section, needs)

        if needs.room:
            ranked = ranked_rooms
            position = {room.id: i for i, room in enumerate(ranked)}
            shared = section_features(section, needs, len(ranked))
            # Row order inside a group is room_id, NOT the ranked order:
            # the ranked order IS the formula, and leaving it in place would leak
            # the label into any evaluation that breaks score ties by position.
            for room in sorted(ranked, key=lambda r: r.id):
                parts = C.room_soft_penalty_parts(section, room)
                penalty = sum(parts.values())
                room_rows.append({
                    "qid": 2 * index,
                    "label": room_label(penalty),
                    **shared,
                    **room_features(section, room),
                    "f_room_demand": room_any[room.id],
                    "f_room_perfect_demand": room_perfect[room.id],
                    **meta,
                    "m_decision": "room",
                    "m_room_id": room.id,
                    "m_building": room.building,
                    "m_candidate_position": position[room.id],
                    "m_penalty_room_type": parts["room_type"],
                    "m_penalty_campus": parts["campus"],
                    "m_penalty_department": parts["department"],
                    "m_penalty_capacity": round(parts["capacity"], 6),
                    "m_soft_penalty": round(penalty, 6),
                    "m_grasp_room_score": round(grasp_room_score(section, room), 6),
                    "m_manual_choice": int(room.id == manual_room),
                })
            stats["room_groups"] += 1
            stats["room_manual_has_room"] += int(manual_room is not None)
            stats["room_manual_in_candidates"] += int(manual_room in position)

        if needs.time:
            ranked = ranked_slots
            position = {ts.id: i for i, ts in enumerate(ranked)}
            shared = section_features(section, needs, len(ranked))
            for ts in sorted(ranked, key=lambda t: t.id):
                parts = C.time_soft_penalty_parts(section, ts)
                penalty = sum(parts.values())
                time_rows.append({
                    "qid": 2 * index + 1,
                    "label": timeslot_label(penalty),
                    **shared,
                    **timeslot_features(ts),
                    "f_slot_demand": slot_any[ts.id],
                    "f_slot_perfect_demand": slot_perfect[ts.id],
                    **meta,
                    "m_decision": "timeslot",
                    "m_timeslot_id": ts.id,
                    "m_candidate_position": position[ts.id],
                    "m_penalty_supervision_daytime": parts["supervision_daytime"],
                    "m_penalty_teaching_evening": parts["teaching_evening"],
                    "m_time_penalty": penalty,
                    "m_grasp_time_score": grasp_time_score(section, ts, guideline_cache),
                    "m_manual_choice": int(ts.id == manual_ts),
                })
            stats["time_groups"] += 1
            stats["time_manual_has_slot"] += int(manual_ts in {t.id for t in timeslots})
            stats["time_manual_in_candidates"] += int(manual_ts in position)

        if needs.room and needs.time and len(baseline_pairs) < 500:
            baseline_pairs.append((section, rooms[index % len(rooms)],
                                   timeslots[index % len(timeslots)]))

    _check_baseline(baseline_pairs, guideline_cache)
    return sections, room_rows, time_rows, stats, by_needs


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["qid"], []).append(row["label"])
    sizes = sorted(len(labels) for labels in groups.values())
    degenerate = sum(1 for labels in groups.values() if len(set(labels)) == 1)
    return {
        "groups": len(groups),
        "rows": len(rows),
        "avg_group_size": round(len(rows) / len(groups), 2),
        "min_group_size": sizes[0],
        "median_group_size": sizes[len(sizes) // 2],
        "max_group_size": sizes[-1],
        "label_distribution": dict(sorted(Counter(r["label"] for r in rows).items())),
        # A group whose candidates all share one label yields no pairs, so it
        # contributes NOTHING to a pairwise/LambdaMART gradient.
        "single_label_groups": degenerate,
        "informative_groups": len(groups) - degenerate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--room-limit", default=str(C.DEFAULT_ROOM_CANDIDATES),
                        help="get_viable_rooms limit, or 'all' (default: the "
                             "scheduler's own, %(default)s)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    room_limit = None if args.room_limit == "all" else int(args.room_limit)
    out_dir = pathlib.Path(args.out_dir)

    sections, room_rows, time_rows, stats, by_needs = build(room_limit)

    # Sorted by qid, non-decreasing -- XGBoost requires it. Already true by
    # construction (qid grows with the section index); asserted, not assumed.
    for rows in (room_rows, time_rows):
        assert all(a["qid"] <= b["qid"] for a, b in zip(rows, rows[1:]))

    write_csv(out_dir / "room_candidates.csv", room_rows)
    write_csv(out_dir / "timeslot_candidates.csv", time_rows)

    summary = {
        "source": "tests/fixtures (schedule_id=1 snapshot)",
        "room_limit": args.room_limit,
        "sections": len(sections),
        "sections_by_needs": dict(by_needs),
        "room": group_summary(room_rows),
        "timeslot": group_summary(time_rows),
        "manual_coverage": {
            "room_groups_whose_manual_placement_has_a_room": stats["room_manual_has_room"],
            "room_groups_whose_manual_room_is_a_candidate": stats["room_manual_in_candidates"],
            "time_groups_whose_manual_placement_is_a_real_slot": stats["time_manual_has_slot"],
            "time_groups_whose_manual_slot_is_a_candidate": stats["time_manual_in_candidates"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
