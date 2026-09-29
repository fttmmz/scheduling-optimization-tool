"""The HARD tier: a room or an instructor in two places at once. Nothing else."""

from .classification import instructor_occupancy_ids


# ── HARD CONSTRAINT VALIDATORS  (item-level — used inside algorithm loops) ────
# Each function returns True / False (violation).


def is_instructor_free(schedule: list, instructor_id, timeslot_id) -> bool:

    """No instructor may teach two sections at the same timeslot."""

    if instructor_id is None or timeslot_id is None:
        return True
    for item in schedule:
        if item.timeslot_id != timeslot_id:
            continue
        # Membership, not equality: a team-taught item commits all its
        # instructors, and equality only ever saw the first one.
        if instructor_id in instructor_occupancy_ids(item):
            return False
    return True


def is_room_free(schedule: list, room_id, timeslot_id) -> bool:

    """No room may host two sections at the same timeslot."""

    if room_id is None or timeslot_id is None:
        return True
    for item in schedule:
        if item.room_id == room_id and item.timeslot_id == timeslot_id:
            return False
    return True


def is_instructor_free_map(occupied_instructors: set, instructor_id, timeslot_id) -> bool:
    """instructor availability check using a prebuilt set."""
    if instructor_id is None or timeslot_id is None:
        return True
    return (instructor_id, timeslot_id) not in occupied_instructors


def is_room_free_map(occupied_rooms: set, room_id, timeslot_id) -> bool:
    """room availability check using a prebuilt set."""
    if room_id is None or timeslot_id is None:
        return True
    return (room_id, timeslot_id) not in occupied_rooms


def passes_hard_constraints(
    section,
    room,
    timeslot,
    schedule: list = None,
    occupied_instructors: set = None,
    occupied_rooms: set = None,
) -> bool:
    """
    Run the HARD constraints for a candidate (section, room, timeslot):
    no double-booked room, no double-booked instructor. That is all — see the
    tier note at the top of rooms.py.

    This used to also require room_type/department/capacity/campus to match,
    which made a merely-imperfect room indistinguishable from a physically
    impossible one and left ~30% of the real schedule's placements unreachable.
    Those four are now priced by room_soft_penalty() instead, so a section takes
    an imperfect room rather than going unscheduled.

    If occupancy maps are provided, use those for instructor/room checks
    instead of scanning the partial schedule.
    """
    # Supervision does not consume its instructors' teaching time, so it neither
    # creates occupancy nor is blocked by it -- instructor_occupancy_ids returns
    # () and the check short-circuits to free. See classification.is_supervision().
    # EVERY instructor must be free, not just the first: a team-taught class
    # cannot run if any one of its teachers is already committed.
    timeslot_id = timeslot.id if timeslot else None
    instructor_ids = instructor_occupancy_ids(section)

    if occupied_instructors is not None:
        instructor_free = all(
            is_instructor_free_map(occupied_instructors, instructor_id, timeslot_id)
            for instructor_id in instructor_ids
        )
    else:
        instructor_free = all(
            is_instructor_free(schedule, instructor_id, timeslot_id)
            for instructor_id in instructor_ids
        )

    if occupied_rooms is not None:
        room_free = is_room_free_map(
            occupied_rooms,
            room.id if room else None,
            timeslot.id if timeslot else None,
        )
    else:
        room_free = is_room_free(
            schedule, room.id if room else None, timeslot.id if timeslot else None
        )

    return instructor_free and room_free
