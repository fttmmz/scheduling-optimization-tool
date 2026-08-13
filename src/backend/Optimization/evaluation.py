# CONFLICT COUNTING  (for schedule scoring / comparison)
from backend.Optimization.constraints import (
    classify_section,
    get_required_room_type,
    get_section_campus,
    get_building_campus,
    get_valid_timeslots_for_section,
    room_soft_penalty,
    room_soft_penalty_parts,
)

# ── Scoring tiers ────────────────────────────────────────────────────────────
# Mirrors the hard/soft split in constraints.py. The old scorer summed all six
# constraint types into one flat conflict count, so a campus mismatch cost
# exactly as much as putting two classes in one room -- which both hid real
# double-bookings and made the manual schedule look far worse than it is.
HARD_CONFLICT_WEIGHT = 3.0    # room/instructor double-booking: never acceptable
UNSCHEDULED_WEIGHT = 2.0      # a missing class is worse than a compromised one
SOFT_PENALTY_WEIGHT = 0.5     # bent soft rules: real, but recoverable

# Divisor that maps the average per-section soft penalty into roughly [0, 1].
# ~10 penalty units is a section that breaks essentially every soft rule.
SOFT_PENALTY_REFERENCE = 10.0

def count_instructor_conflicts(schedule: list) -> int:
    seen, conflicts = set(), 0
    for item in schedule:
        if item.instructor_id is None or item.timeslot_id is None:
            continue
        key = (item.instructor_id, item.timeslot_id)
        if key in seen:
            conflicts += 1
        else:
            seen.add(key)
    return conflicts


def count_room_conflicts(schedule: list) -> int:
    seen, conflicts = set(), 0
    for item in schedule:
        if item.room_id is None or item.timeslot_id is None:
            continue
        key = (item.room_id, item.timeslot_id)
        if key in seen:
            conflicts += 1
        else:
            seen.add(key)
    return conflicts


def count_campus_conflicts(schedule: list, rooms: list) -> int:
    room_map = {room.id: room for room in rooms}
    conflicts = 0
    for item in schedule:
        room = room_map.get(item.room_id)
        if not room:
            continue
        if get_section_campus(item.section) != get_building_campus(room.building):
            conflicts += 1
    return conflicts


def count_room_type_conflicts(schedule: list, rooms: list) -> int:
    room_map = {room.id: room for room in rooms}
    conflicts = 0
    for item in schedule:
        if item.room_id is None:
            continue
        room = room_map.get(item.room_id)
        if not room:
            continue
        required_room_type = get_required_room_type(item.course_type)
        if required_room_type and room.type != required_room_type:
            conflicts += 1
    return conflicts

def count_department_conflicts(schedule: list, rooms: list) -> int:
    room_map = {room.id: room for room in rooms}
    conflicts = 0
    for item in schedule:
        room = room_map.get(item.room_id)
        if not room:
            continue
        if room.dept_id and item.course_dept != room.dept_id:
            conflicts += 1
    return conflicts

def count_capacity_conflicts(schedule: list, rooms: list) -> int:
    room_map = {room.id: room for room in rooms}
    conflicts = 0
    for item in schedule:
        room = room_map.get(item.room_id)
        if not room:
            continue
        if item.capacity > room.capacity:
            conflicts += 1
    return conflicts


def count_hard_conflicts(schedule: list) -> int:
    """HARD tier: physically impossible placements only. A usable schedule has 0.

    This is the number that decides whether a schedule can actually be run, and
    the one the manual schedule scores ~0 on -- which is the whole point of
    separating it from the soft counts below.
    """
    return count_instructor_conflicts(schedule) + count_room_conflicts(schedule)


def soft_violation_counts(schedule: list, rooms: list) -> dict:
    """SOFT tier, per rule: how many placements bend each soft rule.

    Counts, not costs -- for reporting "this schedule breaks campus 41 times",
    which is directly comparable against the manual schedule's own rates.
    """
    room_map = {room.id: room for room in rooms}
    counts = {"room_type": 0, "department": 0, "campus": 0, "capacity": 0}

    for item in schedule:
        room = room_map.get(item.room_id)
        if not room:
            continue
        for rule, cost in room_soft_penalty_parts(item, room).items():
            if cost > 0:
                counts[rule] += 1

    return counts


def total_soft_penalty(schedule: list, rooms: list) -> float:
    """SOFT tier, weighted: total soft cost of every room assignment."""
    room_map = {room.id: room for room in rooms}
    return sum(
        room_soft_penalty(item, room_map[item.room_id])
        for item in schedule
        if item.room_id in room_map
    )


def build_timeslot_guideline_cache(sections: list, timeslots: list) -> dict:
    """Build a cache of valid timeslot IDs for each section."""
    cache = {}
    for sec in sections:
        key = (sec.course.id, str(sec.no))
        valid = get_valid_timeslots_for_section(sec, timeslots)
        cache[key] = {ts.id for ts in valid}
    return cache


def count_timeslot_guideline_conflicts(
    schedule: list,
    sections: list,
    timeslots: list,
    valid_timeslot_cache: dict = None,
) -> int:
    """
    Count sections assigned to timeslots that violate course-type guidelines.
    
    Each course type has specific timeslot requirements (e.g. lectures on MW/TR,
    labs on single days, etc.). This function checks if each scheduled section's
    assigned timeslot matches the guidelines for its course type.
    
    Returns the count of sections violating timeslot guidelines.
    """
    # Build lookup maps for fast access
    section_map = {(sec.course.id, str(sec.no)): sec for sec in sections}
    timeslot_map = {ts.id: ts for ts in timeslots}
    
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    
    conflicts = 0
    for item in schedule:
        # Skip if section has no timeslot assigned
        if item.timeslot_id is None:
            continue
        
        # Get the section object and assigned timeslot
        section = section_map.get((item.course_id, item.section))
        timeslot = timeslot_map.get(item.timeslot_id)
        
        if not section or not timeslot:
            continue
        
        valid_ids = valid_timeslot_cache.get((item.course_id, item.section), set())
        if item.timeslot_id not in valid_ids:
            conflicts += 1
    
    return conflicts


def count_conflicts(
    schedule: list,
    rooms: list,
    sections: list = None,
    timeslots: list = None,
    valid_timeslot_cache: dict = None,
) -> int:
    """
    Flat count of ALL rule violations, hard and soft alike, one point each.

    NOTE: this deliberately mixes the tiers, so it is a "total blemishes"
    display number only -- do NOT read it as a feasibility measure. A schedule
    with 40 campus mismatches and no double-booking is perfectly runnable and
    scores 40 here, while one with 3 double-bookings is unrunnable and scores 3.
    Use count_hard_conflicts() to decide whether a schedule is usable, and
    soft_violation_counts() / total_soft_penalty() for quality. Kept because
    main.py reports it and grasp.py scores against it.

    If sections and timeslots are provided, also includes timeslot guideline
    violations.
    """
    total = (
        count_instructor_conflicts(schedule)
        + count_room_conflicts(schedule)
        + count_campus_conflicts(schedule, rooms)
        + count_room_type_conflicts(schedule, rooms)
        + count_department_conflicts(schedule, rooms)
        + count_capacity_conflicts(schedule, rooms)
    )
    
    # Add timeslot guideline conflicts if data provided
    if sections is not None and timeslots is not None:
        total += count_timeslot_guideline_conflicts(
            schedule,
            sections,
            timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )
    
    return total


def count_scheduled_sections(schedule: list, sections: list) -> int:
    section_lookup = {(section.course.id, section.no): section for section in sections}
    scheduled = 0
    for item in schedule:
        if item.room_id is not None or item.timeslot_id is not None:
            scheduled += 1
            continue

        section = section_lookup.get((item.course_id, item.section))
        if section is not None and classify_section(section) == "NEEDS_NOTHING":
            scheduled += 1

    return scheduled


def calculate_fitness(
    schedule: list,
    rooms: list,
    sections: list = None,
    total_sections: int = None,
    timeslots: list = None,
    valid_timeslot_cache: dict = None,
) -> float:
    """
    Tiered, normalized weighted-penalty fitness.

    Three tiers, priced so they can never trade against each other wrongly:

        HARD        double-booked room/instructor      weight 3.0
        UNSCHEDULED section left with no placement     weight 2.0
        SOFT        campus/dept/room_type/capacity     weight 0.5

    The previous version summed all six constraint types into one flat count at
    weight 1.0, so three campus mismatches outscored a double-booking and the
    search had no reason to prefer a runnable schedule. Splitting the tiers is
    what lets "we beat the manual schedule" mean something: the manual schedule
    scores ~0 hard and a measurable soft total, so an algorithm beats it by
    matching 0 hard while carrying less soft penalty -- not by quietly breaking
    rules the old scorer priced identically.

    Based on:
    - Constraint hierarchy: Ceschia et al. (2023), Müller et al. (2025)
    - Weighted penalty normalization: Burke et al. (1994)
    - Hard/soft separation: Sylejmani et al. (2023)
    """

    # --- Setup ---
    if sections is not None:
        scheduled_count = count_scheduled_sections(schedule, sections)
        total_sections  = len(sections)
    else:
        scheduled_count = sum(
            1 for item in schedule
            if item.room_id is not None and item.timeslot_id is not None
        )
        if total_sections is None:
            total_sections = len(schedule)

    if total_sections == 0:
        return 0.0

    hard_conflicts = count_hard_conflicts(schedule)
    unscheduled    = max(0, total_sections - scheduled_count)
    soft_penalty   = total_soft_penalty(schedule, rooms)

    # Timeslot-guideline (duration/day) violations are a soft rule on the time
    # axis, so they join the soft tier rather than counting as hard conflicts.
    if sections is not None and timeslots is not None:
        soft_penalty += count_timeslot_guideline_conflicts(
            schedule,
            sections,
            timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )

    # --- Normalize each tier by total sections ---
    hard_rate        = hard_conflicts / total_sections
    unscheduled_rate = unscheduled / total_sections
    soft_rate        = soft_penalty / (total_sections * SOFT_PENALTY_REFERENCE)

    total_penalty = (
        hard_rate * HARD_CONFLICT_WEIGHT
        + unscheduled_rate * UNSCHEDULED_WEIGHT
        + soft_rate * SOFT_PENALTY_WEIGHT
    )

    # --- Convert to fitness in (0, 1] ---
    return round(1.0 / (1.0 + total_penalty), 4)

def debug_conflicts_ui(schedule, rooms, sections=None, timeslots=None, valid_timeslot_cache=None):
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader("Conflict Breakdown")

    instructor  = count_instructor_conflicts(schedule)
    room        = count_room_conflicts(schedule)
    campus      = count_campus_conflicts(schedule, rooms)
    room_type   = count_room_type_conflicts(schedule, rooms)
    department  = count_department_conflicts(schedule, rooms)
    capacity    = count_capacity_conflicts(schedule, rooms)
    timeslot_violations = 0
    
    # Include timeslot guideline conflicts if data available
    if sections is not None and timeslots is not None:
        timeslot_violations = count_timeslot_guideline_conflicts(
            schedule,
            sections,
            timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )
    
    total = instructor + room + campus + room_type + department + capacity + timeslot_violations

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Instructor Conflicts", instructor)
        st.metric("Room Conflicts",       room)
        st.metric("Campus Conflicts",     campus)

    with col2:
        st.metric("Room Type Conflicts",  room_type)
        st.metric("Department Conflicts", department)
        st.metric("Capacity Conflicts",   capacity)

    # Show timeslot conflicts if applicable
    if timeslot_violations > 0 or (sections is not None and timeslots is not None):
        col3, _ = st.columns(2)
        with col3:
            st.metric("Timeslot Guideline Violations", timeslot_violations)

    st.divider()

    color = "red" if total > 0 else "green"
    st.markdown(f"### Total Conflicts: :{color}[{total}]")

    if total == 0:
        st.success("No conflicts found. Schedule looks clean!")
    else:
        # show which constraint is the worst offender
        breakdown = {
            "Instructor": instructor,
            "Room":       room,
            "Campus":     campus,
            "Room Type":  room_type,
            "Department": department,
            "Capacity":   capacity,
        }
        if timeslot_violations > 0:
            breakdown["Timeslot Guidelines"] = timeslot_violations
        worst = max(breakdown, key=breakdown.get)
        st.warning(f"Biggest issue: **{worst}** conflicts ({breakdown[worst]})")