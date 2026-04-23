
import networkx as nx
from collections import defaultdict

from backend.models.models import ScheduleItem
from backend.Optimization.constraints import (
    classify_section,
    get_required_room_type,
    get_valid_timeslots,
    get_section_campus,
    _build_room_lookup,
    _viable_rooms
)


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
            section_no=section.no,
            capacity=section.capacity,
            classification=classify_section(section),
        )

    by_instructor = defaultdict(list)
    for node_id in G.nodes:
        instr = G.nodes[node_id]["instructor_id"]
        if instr:
            by_instructor[instr].append(node_id)

    for node_ids in by_instructor.values():
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                G.add_edge(node_ids[i], node_ids[j])

    return G



# Greedy scheduler

def greedy_schedule(sections, timeslots, rooms):
    
    G = build_conflict_graph(sections)
    dept_typed, open_typed = _build_room_lookup(rooms)

    
    occupied_rooms = set() 
    occupied_instructors = set() 
    assigned_timeslots = {} 

    room_list_cache: dict = {}
    def cached_viable_rooms(room_type, course_dept, section_campus):
            key = (room_type, course_dept, section_campus)
            if key not in room_list_cache:
                room_list_cache[key] = list(
                    _viable_rooms(
                        dept_typed, open_typed, room_type, course_dept, section_campus
                    )
                )
            return room_list_cache[key]


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
            )

        # NEEDS_NOTHING
        if classification == "NEEDS_NOTHING":
            schedule.append(make_item(None, None))
            continue

        room_type = get_required_room_type(course_type) or "classroom"
        viable = cached_viable_rooms(room_type, course_dept, section_campus)

        # NEEDS_ROOM_ONLY 
        if classification == "NEEDS_ROOM_ONLY":
            assigned_room = next(
                (r for r in viable if r.capacity >= capacity_needed), None
            )
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
            if instructor_id and (instructor_id, ts_id) in occupied_instructors:
                continue

            # check neighbour (same instructor) already at this slot
            if ts_id in blocked_by_neighbors:
                continue

            for room in viable:
                # check room double-booked
                if (room.id, ts_id) in occupied_rooms:
                    continue

                # capacity  per-section check
                if room.capacity < capacity_needed:
                    continue

                # assignment 
                occupied_rooms.add((room.id, ts_id))
                if instructor_id:
                    occupied_instructors.add((instructor_id, ts_id))
                assigned_timeslots[node_id] = ts_id
                schedule.append(make_item(room.id, ts_id))
                found = True
                break

            if found:
                break

        if not found:
            schedule.append(make_item(None, None))

    return schedule
