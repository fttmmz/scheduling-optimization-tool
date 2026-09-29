
import networkx as nx
from collections import defaultdict

from backend.models.models import ScheduleItem
from backend.Optimization.constraints import (
    classify_section,
    get_required_room_type,
    get_valid_timeslots,
    get_section_campus,
    instructor_occupancy_ids,
    finalize_schedule,
    DEFAULT_ROOM_CANDIDATES,
    SOFT_WEIGHT_CAMPUS,
    SOFT_WEIGHT_DEPARTMENT,
    SOFT_WEIGHT_ROOM_TYPE,
    get_building_campus,
)


# Room lookup

def _build_room_lookup(rooms):
    """
    Returns (dept_typed, open_typed, all_rooms).

    dept_typed: (room_type, dept_id, campus) → rooms
    open_typed: (room_type, campus)          → rooms
    all_rooms:  the full list, used for the soft fallback tail in _viable_rooms.

    Type, department, and campus are static properties, so bucketing them once
    here means the inner scheduling loop only checks availability (O(1)).
    """
    dept_typed = defaultdict(list)
    open_typed = defaultdict(list)

    for room in rooms:
        campus = get_building_campus(room.building)
        if room.dept_id:
            dept_typed[(room.type, room.dept_id, campus)].append(room)
        else:
            open_typed[(room.type, campus)].append(room)

    return dept_typed, open_typed, list(rooms)


def _greedy_room_penalty(room, room_type, course_dept, section_campus) -> float:
    """Soft penalty for greedy's lookup path, which knows only the
    (type, dept, campus) triple rather than the Section object. Capacity is
    scored separately by the caller, which knows the section's enrolment."""
    penalty = 0.0
    if room_type is not None and room.type != room_type:
        penalty += SOFT_WEIGHT_ROOM_TYPE
    if room.dept_id and room.dept_id != course_dept:
        penalty += SOFT_WEIGHT_DEPARTMENT
    if get_building_campus(room.building) != section_campus:
        penalty += SOFT_WEIGHT_CAMPUS
    return penalty


def _viable_rooms(
    dept_typed,
    open_typed,
    room_type,
    course_dept,
    section_campus,
    all_rooms=None,
    limit=DEFAULT_ROOM_CANDIDATES,
):
    """Yield rooms best-fit first for a given (type, dept, campus) triple.

    Tiers 1 and 2 are the exact matches this function used to return, in the
    same order. Tier 3 is new: the least-bad remaining rooms, ranked. Without
    it a section whose triple had no exact match simply went unscheduled, which
    is the greedy-side half of the unscheduled ceiling.
    """
    exact_dept = dept_typed.get((room_type, course_dept, section_campus), [])
    exact_open = open_typed.get((room_type, section_campus), [])

    # 1. dept-assigned rooms matching this course's department
    yield from exact_dept
    # 2. open rooms (no dept restriction)
    yield from exact_open

    if not all_rooms:
        return

    # 3. soft fallback — everything else, cheapest first, so an imperfect room
    #    beats no room at all.
    already = {id(room) for room in exact_dept}
    already.update(id(room) for room in exact_open)

    rest = [room for room in all_rooms if id(room) not in already]
    rest.sort(
        key=lambda room: _greedy_room_penalty(
            room, room_type, course_dept, section_campus
        )
    )
    yield from rest[:limit] if limit is not None else rest


# Conflict graph

def build_conflict_graph(sections):
    G = nx.Graph()

    for idx, section in enumerate(sections):
        node_id = f"{section.course.id}-{section.no or 0}-{idx}"
        G.add_node(
            node_id,
            section_obj=section,
            course_id=section.course.id,
            course_name=section.course.name,
            course_type=section.course.type,
            course_dept=section.course.dept,
            instructor_id=section.instructor_id,
            instructor_ids=section.instructor_ids,
            section_no=section.no,
            capacity=section.capacity,
            classification=classify_section(section),
        )

    by_instructor = defaultdict(list)
    for node_id in G.nodes:
        for instr in G.nodes[node_id]["instructor_ids"]:
            by_instructor[instr].append(node_id)

    for node_ids in by_instructor.values():
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                G.add_edge(node_ids[i], node_ids[j])

    return G



# Greedy scheduler

def greedy_schedule(sections, timeslots, rooms):
    
    G = build_conflict_graph(sections)
    dept_typed, open_typed, all_rooms = _build_room_lookup(rooms)

    
    occupied_rooms = set() 
    occupied_instructors = set() 
    assigned_timeslots = {} 

    room_list_cache: dict = {}
    def cached_viable_rooms(room_type, course_dept, section_campus):
            key = (room_type, course_dept, section_campus)
            if key not in room_list_cache:
                room_list_cache[key] = list(
                    _viable_rooms(
                        dept_typed,
                        open_typed,
                        room_type,
                        course_dept,
                        section_campus,
                        all_rooms=all_rooms,
                    )
                )
            return room_list_cache[key]


    def order_by_capacity(viable, capacity_needed):
        """Rooms that seat the section first (in their existing best-fit order),
        then the ones that overflow, least-overflow first.

        Capacity used to be a hard skip, so a section larger than every room in
        its department/campus/type bucket went unscheduled. Overbooking is
        normal in the real schedule (11% of manual placements), so it is now
        only a preference -- an overfull room beats no room.
        """
        fits, overflows = [], []
        for room in viable:
            (fits if room.capacity >= capacity_needed else overflows).append(room)
        overflows.sort(key=lambda room: -room.capacity)
        return fits + overflows

    schedule = []
    nodes_by_priority = sorted(G.nodes, key=lambda n: G.degree[n], reverse=True)

    for node_id in nodes_by_priority:
        node = G.nodes[node_id]
        section = node["section_obj"]
        classification = node["classification"]
        instructor_id = node["instructor_id"]
        course_type = node["course_type"]
        course_dept = node["course_dept"]
        capacity_needed = node["capacity"]
        section_no = node["section_no"]
        section_campus = get_section_campus(section_no)

        # The instructor identity that occupies a timeslot: None for supervision
        # and for sections with no instructor. `instructor_id` above stays the
        # real recorded value, because the item still has to REPORT it.
        occupying_instructors = instructor_occupancy_ids(section)

        def make_item(room_id, timeslot_id):
            return ScheduleItem(
                course_id=node["course_id"],
                course_name=node["course_name"],
                course_type=course_type,
                course_dept=course_dept,
                capacity=capacity_needed,
                instructor_id=instructor_id,
                room_id=room_id,
                timeslot_id=timeslot_id,
                section=section_no,
                instructor_ids=section.instructor_ids,
                pattern_index=section.pattern_index,
                # Classification is keyed on level -- without it a detached item
                # re-classifies as an ordinary room+time lecture and the
                # level-aware fix silently undoes itself. greedy was the one
                # algorithm still missing this.
                level=getattr(section.course, "level", None),
                course_class=getattr(section.course, "course_class", None),
            )

        # NEEDS_NOTHING
        if classification == "NEEDS_NOTHING":
            schedule.append(make_item(None, None))
            continue

        # NEEDS_TIME_ONLY — needs a timeslot but no room (studios, graduate
        # labs, office hours, training, ...). Pick the first valid slot that
        # doesn't double-book the instructor (or a same-instructor neighbour).
        if classification == "NEEDS_TIME_ONLY":
            blocked_by_neighbors = {
                assigned_timeslots[nei]
                for nei in G.neighbors(node_id)
                if nei in assigned_timeslots
            }
            chosen_ts = None
            for timeslot in get_valid_timeslots(section, timeslots):
                ts_id = timeslot.id
                if any((instructor_id, ts_id) in occupied_instructors
                       for instructor_id in occupying_instructors):
                    continue
                if ts_id in blocked_by_neighbors:
                    continue
                chosen_ts = ts_id
                for instructor_id in occupying_instructors:
                    occupied_instructors.add((instructor_id, ts_id))
                assigned_timeslots[node_id] = ts_id
                break
            schedule.append(make_item(None, chosen_ts))
            continue

        room_type = get_required_room_type(course_type) or "classroom"
        viable = order_by_capacity(
            cached_viable_rooms(room_type, course_dept, section_campus),
            capacity_needed,
        )

        # NEEDS_ROOM_ONLY
        if classification == "NEEDS_ROOM_ONLY":
            assigned_room = viable[0] if viable else None
            schedule.append(
                make_item(assigned_room.id if assigned_room else None, None)
            )
            continue

        # NEEDS_ROOM_AND_TIME
        candidate_slots = get_valid_timeslots(section, timeslots)

        # Compute neighbour-blocked timeslot ids once per node 
        blocked_by_neighbors = {
            assigned_timeslots[nei]
            for nei in G.neighbors(node_id)
            if nei in assigned_timeslots
        }

        found = False
        for timeslot in candidate_slots:
            ts_id = timeslot.id

            # check instructor double-booked
            if any((instructor_id, ts_id) in occupied_instructors
                   for instructor_id in occupying_instructors):
                continue

            # check neighbour (same instructor) already at this slot
            if ts_id in blocked_by_neighbors:
                continue

            for room in viable:
                # check room double-booked -- the only hard reason to skip a
                # room. Capacity/type/dept/campus are soft and already reflected
                # in the order of `viable`, so the first free room here is the
                # best-fit one still available.
                if (room.id, ts_id) in occupied_rooms:
                    continue

                # assignment
                occupied_rooms.add((room.id, ts_id))
                for instructor_id in occupying_instructors:
                    occupied_instructors.add((instructor_id, ts_id))
                assigned_timeslots[node_id] = ts_id
                schedule.append(make_item(room.id, ts_id))
                found = True
                break

            if found:
                break

        if not found:
            schedule.append(make_item(None, None))

    # Meeting blocks of one section need their own hour, and share one room.
    # The placement loops have no cross-block state, so repair here.
    return finalize_schedule(schedule, sections, timeslots, rooms=rooms)
