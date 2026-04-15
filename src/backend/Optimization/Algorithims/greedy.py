from backend.Optimization.constraints import (
    get_valid_timeslots,
    passes_hard_constraints,
    is_room_type_match,
    is_department_match,
    is_capacity_ok,
    is_campus_match,
)
from backend.models.models import ScheduleItem


def get_viable_rooms(section, rooms):
    """Return rooms that satisfy the section's static constraints."""
    return [
        room
        for room in rooms
        if is_room_type_match(section, room)
        and is_department_match(section, room)
        and is_capacity_ok(section, room)
        and is_campus_match(section, room)
    ]


def greedy2_schedule(sections, timeslots, rooms):
    """Greedy schedule builder using Section model fields."""
    schedule = []
    occupied_instructors = set()
    occupied_rooms = set()

    for section in sections:
        course_id = section.course.id
        course_name = section.course.name
        course_type = section.course.type
        course_dept = section.course.dept
        capacity = section.capacity
        instructor_id = section.instructor_id
        section_label = str(section.no)

        viable_rooms = get_viable_rooms(section, rooms)
        candidate_timeslots = get_valid_timeslots(section, timeslots)

        assigned = False
        for timeslot in candidate_timeslots:
            if instructor_id is not None and (instructor_id, timeslot.id) in occupied_instructors:
                continue

            for room in viable_rooms:
                if (room.id, timeslot.id) in occupied_rooms:
                    continue

                if passes_hard_constraints(
                    section,
                    room,
                    timeslot,
                    occupied_instructors=occupied_instructors,
                    occupied_rooms=occupied_rooms,
                ):
                    schedule.append(
                        ScheduleItem(
                            course_id=course_id,
                            course_name=course_name,
                            course_type=course_type,
                            course_dept=course_dept,
                            capacity=capacity,
                            instructor_id=instructor_id,
                            room_id=room.id,
                            timeslot_id=timeslot.id,
                            section=section_label,
                        )
                    )
                    occupied_rooms.add((room.id, timeslot.id))
                    if instructor_id is not None:
                        occupied_instructors.add((instructor_id, timeslot.id))
                    assigned = True
                    break

            if assigned:
                break

        if not assigned:
            schedule.append(
                ScheduleItem(
                    course_id=course_id,
                    course_name=course_name,
                    course_type=course_type,
                    course_dept=course_dept,
                    capacity=capacity,
                    instructor_id=instructor_id,
                    room_id=None,
                    timeslot_id=None,
                    section=section_label,
                )
            )

    return schedule
