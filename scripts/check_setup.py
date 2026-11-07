"""Check this device gets the team's results. Reads the database, writes nothing.

    python scripts/check_setup.py

Compares the private files, the data in its exact order, and one greedy run with the expected values
below, and exits with an error if any differ. See REPRODUCING.md. If the data changes on purpose, every
device's numbers change with it: rerun the benchmarks and update these values together.
"""
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXPECTED_SECTIONS = 3270
EXPECTED_ROSTER_COURSES = 4
EXPECTED_DATA = "7ed48dd4f0efcec7"
EXPECTED_ROOM_SCORES = "72cb020e43ac46b5"   # ml/data_registrar/hybrid/room_scores.csv
EXPECTED_RL_MODEL = "768d423abd8a0f7b"      # ml/models_registrar/rl/fqiet_xgb_et280.json
EXPECTED_GREEDY = {"unscheduled": 7, "hard": 1, "fitness": 0.9482}

failures = []


def report(label, actual, expected):
    matches = actual == expected
    print(f"[{'OK' if matches else 'DIFFERENT'}] {label}: {actual}" + ("" if matches else f" (expected {expected})"))
    if not matches:
        failures.append(label)


def data_fingerprint(sections, rooms, timeslots):
    """A hash of the ids in their exact order: same hash, same input to the algorithms."""
    content = [
        [[s.id, s.pattern_index, list(s.instructor_ids or ())] for s in sections],
        [r.id for r in rooms],
        [t.id for t in timeslots],
    ]
    return hashlib.sha256(json.dumps(content, default=str).encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path):
    if not path.exists():
        return f"missing: {path}"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    from backend.Optimization.Algorithims.greedy import greedy_schedule
    from backend.Optimization.Algorithims.ml_greedy import _find_data_dir
    from backend.Optimization.Algorithims.rl_scheduler import DEFAULT_MODEL, _model_dir
    from backend.Optimization.constraints import (
        active_registrar_practice, tag_intro_it_pairs, tag_section_links,
    )
    from backend.Optimization.constraints.classification import TEAM_ROSTER_COURSE_IDS
    from backend.Optimization.evaluation import (
        build_timeslot_guideline_cache, calculate_fitness, count_hard_conflicts, count_scheduled_sections,
    )
    from backend.database.loader import get_scheduling_data

    data = get_scheduling_data()
    sections, rooms, timeslots = data["sections"], data["rooms"], data["timeslots"]
    tag_section_links(sections)
    tag_intro_it_pairs(sections)

    report("registrar practice loaded", active_registrar_practice() is not None, True)
    report("team-roster courses (institution_data.py)", len(TEAM_ROSTER_COURSE_IDS), EXPECTED_ROSTER_COURSES)
    report("sections", len(sections), EXPECTED_SECTIONS)
    report("data and its order", data_fingerprint(sections, rooms, timeslots), EXPECTED_DATA)
    report("ml_greedy score table", file_fingerprint(_find_data_dir() / "hybrid" / "room_scores.csv"), EXPECTED_ROOM_SCORES)
    report("RL model", file_fingerprint(_model_dir() / f"{DEFAULT_MODEL}.json"), EXPECTED_RL_MODEL)

    print("Running greedy (about 15 s)...")
    schedule = greedy_schedule(sections, timeslots, rooms)
    cache = build_timeslot_guideline_cache(sections, timeslots)
    report("greedy result", {
        "unscheduled": len(sections) - count_scheduled_sections(schedule, sections),
        "hard": count_hard_conflicts(schedule, rooms),
        "fitness": round(calculate_fitness(schedule, rooms, sections=sections, timeslots=timeslots,
                                           valid_timeslot_cache=cache), 4),
    }, EXPECTED_GREEDY)

    if failures:
        print(f"\nNOT MATCHING: {', '.join(failures)}. See REPRODUCING.md.")
        return 1
    print("\nAll checks match: this device should reproduce the team's results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
