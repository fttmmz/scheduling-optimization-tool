# CONFLICT COUNTING  (for schedule scoring / comparison)
from backend.Optimization.constraints import (
    classify_section,
    get_required_room_type,
    get_section_campus,
    get_building_campus,
    get_valid_timeslots_for_section,
)

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
    Count all conflicts in a schedule.
    
    If sections and timeslots are provided, also includes timeslot guideline violations.
    Otherwise, counts only hard constraint violations.
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
    Normalized weighted penalty fitness function.
    
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

    conflicts   = count_conflicts(
        schedule,
        rooms,
        sections,
        timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    unscheduled = max(0, total_sections - scheduled_count)

    # --- Normalize both by total sections ---
    unscheduled_rate = unscheduled / total_sections   # 0.0 = perfect
    conflict_rate    = conflicts   / total_sections   # 0.0 = perfect

    # --- Separate penalty terms ---
    # Unscheduled weighted higher — a missing class is worse than a conflict
    # because at least a conflicted class exists in the schedule
    unscheduled_penalty = unscheduled_rate * 2.0      # range [0, 2]
    conflict_penalty    = conflict_rate    * 1.0      # range [0, ~0.7 for your data]

    total_penalty = unscheduled_penalty + conflict_penalty

    # --- Convert to fitness in (0, 1] ---
    fitness = 1.0 / (1.0 + total_penalty)

    return round(fitness, 4)

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