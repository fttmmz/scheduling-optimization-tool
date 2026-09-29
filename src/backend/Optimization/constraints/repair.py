"""Post-passes every algorithm runs on its result (via finalize_schedule).

They fix sibling meeting blocks, which no algorithm couples while placing."""

from collections import Counter

from .classification import group_siblings, instructor_occupancy_ids, sibling_key
from .rooms import get_viable_rooms_for_schedule_item
from .timeslots import get_valid_timeslots_for_section


def _occupancy_of(schedule):
    rooms = set()
    instructors = set()
    for item in schedule:
        if item.timeslot_id is None:
            continue
        if item.room_id is not None:
            rooms.add((item.room_id, item.timeslot_id))
        for instructor_id in instructor_occupancy_ids(item):
            instructors.add((instructor_id, item.timeslot_id))
    return rooms, instructors


def resolve_sibling_time_conflicts(schedule, sections, timeslots,
                                   valid_timeslot_cache=None):
    """Give each meeting block of one section its own hour.

    The five algorithms place blocks independently and have no cross-block
    state, so a section that meets twice happily gets both blocks at the same
    time -- which is impossible for the students, and which nothing else in the
    model catches: the blocks may carry no room and no instructor.

    Threading a coupling constraint through five different placement loops for
    the eleven sections this affects would be disproportionate, so it is
    repaired here instead. A block only moves to a slot that is valid for its
    course type and free for its room and every one of its instructors, so this
    can never introduce a conflict; blocks with nowhere to go are left alone and
    still register in count_sibling_conflicts().
    """
    groups = group_siblings(schedule)
    if not groups:
        return schedule

    if valid_timeslot_cache is None:
        # Only the handful of sections that actually have siblings, rather than
        # rebuilding the whole cache.
        valid_timeslot_cache = {
            sibling_key(section): {
                ts.id for ts in get_valid_timeslots_for_section(section, timeslots)
            }
            for section in sections
            if sibling_key(section) in groups
        }

    occupied_rooms, occupied_instructors = _occupancy_of(schedule)

    for key, group in groups.items():
        candidates = sorted(valid_timeslot_cache.get(key) or ())
        taken = set()

        for item in group:
            current = item.timeslot_id
            if current is None:
                continue
            if current not in taken:
                taken.add(current)
                continue

            instructors = instructor_occupancy_ids(item)
            replacement = next(
                (
                    slot for slot in candidates
                    if slot not in taken
                    and (item.room_id is None
                         or (item.room_id, slot) not in occupied_rooms)
                    and all((instructor_id, slot) not in occupied_instructors
                            for instructor_id in instructors)
                ),
                None,
            )
            if replacement is None:
                continue

            if item.room_id is not None:
                occupied_rooms.discard((item.room_id, current))
                occupied_rooms.add((item.room_id, replacement))
            for instructor_id in instructors:
                occupied_instructors.discard((instructor_id, current))
                occupied_instructors.add((instructor_id, replacement))

            item.timeslot_id = replacement
            taken.add(replacement)

    return schedule


def finalize_schedule(schedule, sections, timeslots, valid_timeslot_cache=None,
                      rooms=None):
    """Repairs every algorithm applies to the schedule it returns.

    Order matters: separate the sibling blocks in TIME first, because two blocks
    sharing an hour can never also share a room -- unifying rooms before the
    times are fixed would silently do nothing.
    """
    schedule = resolve_sibling_time_conflicts(
        schedule, sections, timeslots, valid_timeslot_cache
    )
    return unify_sibling_rooms(schedule, rooms)


def unify_sibling_rooms(schedule, rooms=None):
    """Move every meeting block of one section into a single shared room.

    A section that meets twice a week meets in the SAME place both times; the
    source data does exactly that. The algorithms place each block independently
    and have no cross-block state, so rather than thread a coupling constraint
    through five different placement loops, this repairs it afterwards -- the
    affected population is six sections, so a repair pass is the proportionate
    tool.

    Pass *rooms* to search the section's whole ranked room list, not only the
    rooms its blocks already happen to sit in. Without it the repair gives up
    whenever those two or three rooms are busy at one of the hours, even though
    some other perfectly good room is free at all of them -- which left one
    section split in roughly 3% of runs. With it, a section stays split only when
    NO room it could use is free across all of its meeting hours.

    Rooms already in use by the group are tried FIRST, so a schedule that is
    already unified is never churned, and the wider list is ranked by soft
    penalty so the fallback prefers rooms that suit the section anyway.

    Never creates a conflict: a block only moves if the target room is actually
    free at that block's timeslot. Blocks that cannot move are left where they
    are, so this can improve the schedule but never break it. Mutates in place
    and returns the schedule.
    """
    occupied = {
        (item.room_id, item.timeslot_id)
        for item in schedule
        if item.room_id is not None and item.timeslot_id is not None
    }

    for group in group_siblings(schedule).values():
        roomed = [item for item in group
                  if item.room_id is not None and item.timeslot_id is not None]
        if len(roomed) < 2:
            continue
        if len({item.room_id for item in roomed}) == 1:
            continue  # already unified

        # Free the group's own bookings first, so a room is not judged busy on
        # account of the very blocks being moved.
        for item in roomed:
            occupied.discard((item.room_id, item.timeslot_id))

        # Try the rooms these blocks already sit in -- all are viable for this
        # section by construction. Most-used first, ties on lowest id so repeated
        # runs converge rather than oscillate. Trying every candidate rather than
        # only the most common one matters: the popular room is often busy at one
        # sibling's hour while a less popular one is free at all of them.
        counts = Counter(item.room_id for item in roomed)
        candidates = sorted(counts, key=lambda room_id: (-counts[room_id], room_id))

        # Then every other room this section could use, best-fit first. Ranked
        # over the FULL room list rather than the default top-25, because this
        # is a last resort: a slightly worse room shared by both blocks beats a
        # good room used by only one.
        if rooms:
            seen = set(candidates)
            for room in get_viable_rooms_for_schedule_item(
                roomed[0], rooms, limit=len(rooms)
            ):
                if room.id not in seen:
                    seen.add(room.id)
                    candidates.append(room.id)

        target = next(
            (
                room_id for room_id in candidates
                if all((room_id, item.timeslot_id) not in occupied
                       for item in roomed)
            ),
            None,
        )

        if target is not None:
            for item in roomed:
                item.room_id = target

        for item in roomed:
            occupied.add((item.room_id, item.timeslot_id))

    return schedule
