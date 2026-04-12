from models import ScheduleItem


def greedy_schedule(courses, rooms, timeslots, instructors, mapping, constraints):
    """
    Greedy scheduling algorithm.
    Assigns each course the first available room + timeslot
    that satisfies all constraints (instructor, room, campus).

    Parameters:
        courses    : list of Course objects
        rooms      : list of Room objects
        timeslots  : list of TimeSlot objects
        instructors: list of Instructor objects
        mapping    : dict mapping course_id -> instructor_id or dict with instructor_id + section
        constraints: list of constraint functions

    Returns:
        schedule   : list of ScheduleItem objects
    """
    schedule = []
    data = {"rooms": rooms}

    for course in courses:
        # Get instructor_id and section from mapping
        mapped = mapping.get(course.course_id)
        if mapped is None:
            print(f"No instructor mapped for course {course.course_id}, skipping.")
            continue

        # mapping value can be a dict {instructor_id, section} or just instructor_id
        if isinstance(mapped, dict):
            instructor_id = mapped["instructor_id"]
            section = str(mapped["section"])
        else:
            instructor_id = mapped
            section = "1"  # default section if not provided

        assigned = False

        for timeslot in timeslots:
            for room in rooms:
                # Build a candidate ScheduleItem (matches models.py exactly)
                candidate = ScheduleItem(
                    course_id=course.course_id,
                    instructor_id=instructor_id,
                    room_id=room.room_id,
                    timeslot_id=timeslot.timeslot_id,
                    section=section
                )

                # Temporarily add candidate to schedule
                schedule.append(candidate)

                # Run all constraint checks (instructor, room, campus)
                valid = all(constraint(schedule, data) for constraint in constraints)

                if valid:
                    assigned = True
                    break  # valid combo found, move to next course
                else:
                    schedule.pop()  # remove and try next combo

            if assigned:
                break  # stop trying rooms/timeslots for this course

        if not assigned:
            print(f"Could not assign course {course.course_id} — no valid slot found.")

    return schedule
