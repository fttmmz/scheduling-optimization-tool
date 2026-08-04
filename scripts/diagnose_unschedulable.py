"""
Diagnose why real sections end up unschedulable under the current hard
constraints, independent of which algorithm runs. Loads the real dataset
the same way the API does (get_scheduling_data + tag_section_links +
tag_intro_it_pairs) and, for every section that has zero viable rooms
under the STATIC hard constraints (ignoring room/instructor occupancy --
i.e. even before any algorithm starts placing other sections), reports
which single constraint is responsible.

This isolates "structurally infeasible no matter what" from "infeasible
only because of scheduling order/contention" -- the former needs a
constraint change (relax something from hard to soft), the latter needs
a better algorithm/more capacity, not a constraint change.

Usage (from the repo root, with real Supabase credentials available --
either a populated src/backend/.env, or SUPABASE env vars already set):

    python scripts/diagnose_unschedulable.py

Read-only: this only calls get_scheduling_data() (SELECT queries), it
never writes to the database.
"""

import os
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(_REPO_ROOT, "src", "backend", ".env"))

from backend.database.loader import get_scheduling_data  # noqa: E402
from backend.Optimization.constraints import (  # noqa: E402
    classify_section,
    tag_section_links,
    tag_intro_it_pairs,
    get_viable_rooms,
    get_valid_timeslots_for_section,
    is_room_type_match,
    is_department_match,
    is_capacity_ok,
    is_campus_match,
    get_required_room_type,
    get_section_campus,
    get_building_campus,
)

CONSTRAINT_CHECKS = [
    ("room_type", is_room_type_match),
    ("department", is_department_match),
    ("capacity", is_capacity_ok),
    ("campus", is_campus_match),
]


def analyze_section(section, rooms):
    """
    For a section with zero viable rooms, find out which single-constraint
    relaxation would unlock the most rooms, and how many rooms are blocked
    by more than one constraint at once (relaxing just one wouldn't help
    those).
    """
    single_block_counts = Counter()
    multi_block_count = 0

    for room in rooms:
        failing = [name for name, check in CONSTRAINT_CHECKS if not check(section, room)]
        if len(failing) == 1:
            single_block_counts[failing[0]] += 1
        elif len(failing) > 1:
            multi_block_count += 1
        # len(failing) == 0 shouldn't happen here since get_viable_rooms
        # was already confirmed empty for this section.

    return single_block_counts, multi_block_count


def main():
    print("Loading real scheduling data...")
    data = get_scheduling_data()
    if data is None:
        print("ERROR: get_scheduling_data() returned None -- check Supabase credentials.")
        sys.exit(1)

    sections, rooms, timeslots = data["sections"], data["rooms"], data["timeslots"]
    tag_section_links(sections)
    tag_intro_it_pairs(sections)

    print(f"Loaded {len(sections)} sections, {len(rooms)} rooms, {len(timeslots)} timeslots.\n")

    needs_nothing = 0
    fully_ok = 0
    zero_viable_rooms = []
    zero_valid_timeslots = []

    # If relaxing constraint X alone would unlock >=1 room for a section,
    # that section is counted here under X -- a section can be counted
    # under multiple constraints if more than one relaxation would help it.
    would_unlock = {name: 0 for name, _ in CONSTRAINT_CHECKS}
    needs_multi_relaxation = 0  # no single relaxation suffices

    for section in sections:
        requirement = classify_section(section)
        if requirement == "NEEDS_NOTHING":
            needs_nothing += 1
            continue

        viable_rooms = get_viable_rooms(section, rooms)
        needs_timeslot = requirement == "NEEDS_ROOM_AND_TIME"
        valid_ts = get_valid_timeslots_for_section(section, timeslots) if needs_timeslot else None

        if needs_timeslot and not valid_ts:
            zero_valid_timeslots.append(section)

        if viable_rooms:
            fully_ok += 1
            continue

        zero_viable_rooms.append(section)
        single_block_counts, multi_block_count = analyze_section(section, rooms)

        if single_block_counts:
            for name in single_block_counts:
                would_unlock[name] += 1
        elif multi_block_count > 0:
            needs_multi_relaxation += 1

    total_needing_placement = len(sections) - needs_nothing

    print("=== Summary ===")
    print(f"Sections needing no room/timeslot at all: {needs_nothing}")
    print(f"Sections needing a room (and possibly a timeslot): {total_needing_placement}")
    print(f"  -> have at least one statically viable room:     {fully_ok}")
    print(f"  -> have ZERO viable rooms (structurally stuck):  {len(zero_viable_rooms)}")
    if zero_valid_timeslots:
        print(f"Sections with ZERO valid timeslots (rare -- get_valid_timeslots_for_section "
              f"has fallbacks, so this usually means 0): {len(zero_valid_timeslots)}")
    print()

    print("=== If ZERO viable rooms: which single constraint relaxation would unlock a room? ===")
    print("(a section can appear under more than one row if multiple relaxations would each help it alone)\n")
    for name, count in sorted(would_unlock.items(), key=lambda kv: -kv[1]):
        pct = (count / len(zero_viable_rooms) * 100) if zero_viable_rooms else 0
        print(f"  {name:12s}: {count:5d} sections ({pct:.1f}% of the stuck ones) would gain a room if this were softened")
    print(f"  {'(needs >=2)':12s}: {needs_multi_relaxation:5d} sections need MORE THAN ONE constraint relaxed simultaneously")
    print()

    print("=== Coarse capacity check: sections needing a room vs. rooms available, by (room_type, campus) ===")
    demand = Counter()
    for section in sections:
        requirement = classify_section(section)
        if requirement == "NEEDS_NOTHING":
            continue
        room_type = get_required_room_type(section.course.type) or "(any)"
        campus = get_section_campus(section.no)
        demand[(room_type, campus)] += 1

    supply = Counter()
    for room in rooms:
        campus = get_building_campus(room.building)
        supply[(room.type, campus)] += 1
        supply[("(any)", campus)] += 1  # rooms with no specific type requirement can use any room

    print(f"{'room_type':12s} {'campus':8s} {'sections':>10s} {'rooms':>8s}")
    for key in sorted(set(demand) | set(supply)):
        d = demand.get(key, 0)
        s = supply.get(key, 0)
        flag = "  <-- more sections than rooms" if d > s else ""
        print(f"{key[0]:12s} {key[1]:8s} {d:10d} {s:8d}{flag}")

    print(
        "\nNote: rooms are reusable across timeslots, so 'sections > rooms' here does NOT\n"
        "automatically mean infeasible -- it's only a red flag when it's a large multiple\n"
        "of the number of usable timeslots per week for that section type."
    )

    # Dump a small sample of stuck sections for manual inspection.
    if zero_viable_rooms:
        print(f"\n=== Sample of stuck sections (first 15 of {len(zero_viable_rooms)}) ===")
        for section in zero_viable_rooms[:15]:
            single_block_counts, multi_block_count = analyze_section(section, rooms)
            print(
                f"  course={section.course.id} section={section.no} "
                f"type={section.course.type!r} dept={section.course.dept!r} "
                f"capacity={section.capacity} instructor={section.instructor_id} "
                f"| single-constraint blocks: {dict(single_block_counts)} "
                f"| multi-constraint-blocked rooms: {multi_block_count}"
            )


if __name__ == "__main__":
    main()
