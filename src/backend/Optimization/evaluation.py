# CONFLICT COUNTING  (for schedule scoring / comparison)
import streamlit as st
from Optimization.constraints import get_section_campus, get_building_campus

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
        room = room_map.get(item.room_id)
        if not room:
            continue
        if item.course_type != room.type:
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

def count_conflicts(schedule: list, rooms: list) -> int:
    return (
        count_instructor_conflicts(schedule)
        + count_room_conflicts(schedule)
        + count_campus_conflicts(schedule, rooms)
        + count_room_type_conflicts(schedule, rooms)
        + count_department_conflicts(schedule, rooms)
        + count_capacity_conflicts(schedule, rooms)
    )


def calculate_fitness(schedule: list, rooms: list, total_sections: int = None) -> float:
    scheduled_count = sum(
        1 for item in schedule 
        if item.room_id is not None and item.timeslot_id is not None
    )
    
    # if total_sections not provided, use the schedule length
    if total_sections is None:
        total_sections = len(schedule)

    conflict_penalty = count_conflicts(schedule, rooms) * 10
    unscheduled_penalty = max(0, total_sections - scheduled_count) * 20

    fitness = 1000.0 - conflict_penalty - unscheduled_penalty
    return max(0.0, fitness)


def debug_conflicts_ui(schedule, rooms):
    st.subheader("Conflict Breakdown")

    instructor  = count_instructor_conflicts(schedule)
    room        = count_room_conflicts(schedule)
    campus      = count_campus_conflicts(schedule, rooms)
    room_type   = count_room_type_conflicts(schedule, rooms)
    department  = count_department_conflicts(schedule, rooms)
    capacity    = count_capacity_conflicts(schedule, rooms)
    total       = instructor + room + campus + room_type + department + capacity

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Instructor Conflicts", instructor)
        st.metric("Room Conflicts",       room)
        st.metric("Campus Conflicts",     campus)

    with col2:
        st.metric("Room Type Conflicts",  room_type)
        st.metric("Department Conflicts", department)
        st.metric("Capacity Conflicts",   capacity)

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
        worst = max(breakdown, key=breakdown.get)
        st.warning(f"Biggest issue: **{worst}** conflicts ({breakdown[worst]})")