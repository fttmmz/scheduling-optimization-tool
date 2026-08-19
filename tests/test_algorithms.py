"""Smoke tests over a small synthetic dataset.

Not a quality check -- these assert that an algorithm's OUTPUT is well-formed,
which is where the recurring bugs have been. Chiefly: every ScheduleItem must
carry `level`, because classification is level-aware and an item that loses it
silently re-classifies as an ordinary room+time lecture. That regression has
now been found three times (hybrid's clone_item, then genetic/GRASP/PSO, then
greedy on 2026-08-18) -- hence a test rather than another fix.
"""
import pytest

from backend.models.models import Room, Section, Timeslot


def make_sections():
    """A spread across the classification table, including the awkward cases.

    Deliberately includes:
      * team-taught TIME-ONLY sections sharing instructors (courses 5 and 9).
        Two separate bugs lived exactly here -- a placement path that checked
        only the FIRST instructor, and an occupancy map seeded only from ROOMED
        items, so time-only teachers were invisible and got double-booked.
      * a two-block section (course 10) to exercise the sibling rules.
    """
    specs = [
        # course_id, course_type, level, section, instructor_ids
        (1, "Lecture Undergraduate", "Undergraduate", "61", ["1551"]),
        (2, "Lecture Undergraduate", "Undergraduate", "62", ["1552"]),
        (3, "Laboratory", "Undergraduate", "61L", ["1553"]),
        (4, "Laboratory", "Master", "61", ["1554"]),            # grad lab: no room
        (5, "Clinical Practice", "Master", "71", ["1560", "1561"]),   # time, no room
        (6, "Thesis / Dissertation Master", "Master", "61", ["1556"]),  # nothing
        (7, "Senior Project Supervision", "Undergraduate", "61", ["1557"]),
        (8, "Lecture Graduate", "Master", "61", ["1560"]),      # shares 1560 with 5
        # 1561 is this one's SECOND instructor and course 5's second too. A
        # placement path that checks only instructor_ids[0] sees 1564 free and
        # books them both onto the same hour -- which is the bug.
        (9, "Clinical Practice", "Master", "72", ["1564", "1561"]),
        (10, "Studio Undergraduate", "Undergraduate", "61", ["1563"]),  # two blocks
        # Two blocks and NO instructor -- so nothing except the sibling rule
        # keeps them apart. Most real multi-block sections are exactly this
        # shape (clinical, roomless, instructor column empty), and without the
        # repair every algorithm stacks both blocks on the same hour.
        (11, "Clinical Practice", "Master", "71", []),
    ]
    sections = []
    for course_id, course_type, level, section_no, instructor_ids in specs:
        row = {
            "courses": {
                "course_id": course_id, "name": f"Course {course_id}",
                "course_type": course_type, "dept_id": 1,
                "level": level, "course_class": None,
            },
            "instructor_id": instructor_ids[0] if instructor_ids else None,
            "section": section_no,
            "sec_capacity": 25,
        }
        if course_id in (10, 11):
            for pattern_index in (0, 1):
                sections.append(Section(row, instructor_ids=instructor_ids,
                                        pattern_index=pattern_index, pattern_count=2))
        else:
            sections.append(Section(row, instructor_ids=instructor_ids))
    return sections


def make_rooms():
    return [
        Room({"room_id": i, "capacity": 40,
              "room_type": "lab" if i % 3 == 0 else "classroom",
              "building": "M8", "dept_id": 1, "room_num": str(i)})
        for i in range(1, 9)
    ]


def make_timeslots():
    slots, tid = [], 1
    for day in ("MW", "TR", "M", "T", "W", "R"):
        for start, end in (("08:00:00", "09:15:00"), ("10:00:00", "12:50:00"),
                           ("17:00:00", "18:15:00")):
            slots.append(Timeslot({"timeslot_id": tid, "day": day,
                                   "start_time": start, "end_time": end}))
            tid += 1
    return slots


@pytest.fixture(scope="module")
def dataset():
    return make_sections(), make_timeslots(), make_rooms()


def _run(name, sections, timeslots, rooms):
    """Call one algorithm and normalise its return to just the schedule."""
    if name == "greedy":
        from backend.Optimization.Algorithims.greedy import greedy_schedule
        return greedy_schedule(sections, timeslots, rooms)
    if name == "hybrid":
        from backend.Optimization.Algorithims.hybrid import genetic_schedule
        return genetic_schedule(sections, timeslots, rooms)
    if name == "genetic":
        from backend.Optimization.Algorithims.genetic import genetic_schedule
        return genetic_schedule(sections, timeslots, rooms)
    if name == "grasp":
        from backend.Optimization.Algorithims.grasp import grasp_schedule
        return grasp_schedule(sections, timeslots, rooms)[0]
    if name == "pso":
        from backend.Optimization.Algorithims.pso import pso_schedule
        return pso_schedule(sections, timeslots, rooms, seed=1)[0]
    raise AssertionError(name)


ALGORITHMS = ["greedy", "hybrid", "genetic", "grasp", "pso"]


@pytest.fixture(scope="module", params=ALGORITHMS)
def result(request, dataset):
    sections, timeslots, rooms = dataset
    return request.param, _run(request.param, sections, timeslots, rooms)


def test_no_hard_conflicts(result, dataset):
    """No algorithm may emit a physically impossible schedule.

    Covers all three hard rules at once -- double-booked room, double-booked
    instructor (counting EVERY instructor, not just the first), and a section
    holding two of its own meeting blocks at one hour.
    """
    from backend.Optimization.evaluation import (
        count_hard_conflicts, count_instructor_conflicts,
        count_room_conflicts, count_sibling_conflicts,
    )
    name, schedule = result
    assert count_instructor_conflicts(schedule) == 0, name
    assert count_room_conflicts(schedule) == 0, name
    assert count_sibling_conflicts(schedule) == 0, name
    assert count_hard_conflicts(schedule) == 0, name


def test_every_instructor_is_carried_through(result, dataset):
    """Dropping the list back to a scalar would silently lose team teachers."""
    sections, _, _ = dataset
    name, schedule = result
    expected = {s.id: set(s.instructor_ids) for s in sections}

    for item in schedule:
        key = (item.course_id, item.section, getattr(item, "pattern_index", 0))
        if key in expected:
            assert set(item.instructor_ids) == expected[key], f"{name} {key}"


def test_sibling_blocks_share_a_room(result):
    """A section that meets twice meets in the same place both times."""
    from backend.Optimization.constraints import count_sibling_room_splits
    name, schedule = result
    assert count_sibling_room_splits(schedule) == 0, name


def test_all_levels_survive(result, dataset):
    """`level` must reach every item, in every algorithm -- not just greedy."""
    sections, _, _ = dataset
    name, schedule = result
    expected = {s.course.id: s.course.level for s in sections}
    for item in schedule:
        assert item.level == expected[item.course_id], f"{name} {item.course_id}"


def test_grasp_local_search_sees_roomless_instructors():
    """GRASP's occupancy map must count instructors of ROOMLESS classes too.

    Seeding it only from roomed items made the teachers of time-only classes
    (clinical, studios, office hours) invisible, so local search would happily
    move a roomed class onto an hour its instructor was already teaching.

    The scenario below is built so the ONLY soft improvement available is
    exactly that illegal move: the lecture sits in a wrong-type room at ts=1,
    the correctly-typed room is free at ts=2, and instructor 'X' is already
    teaching a roomless clinical class at ts=2.
    """
    from backend.Optimization.Algorithims.grasp import local_search
    from backend.Optimization.evaluation import (
        build_timeslot_guideline_cache, count_instructor_conflicts,
    )
    from backend.models.models import Room, ScheduleItem, Section, Timeslot

    timeslots = [
        Timeslot({"timeslot_id": 1, "day": "M",
                  "start_time": "08:00:00", "end_time": "09:15:00"}),
        Timeslot({"timeslot_id": 2, "day": "W",
                  "start_time": "08:00:00", "end_time": "09:15:00"}),
    ]
    rooms = [
        # Wrong type for a lecture -> carries a soft penalty.
        Room({"room_id": 1, "capacity": 50, "room_type": "lab",
              "building": "M8", "dept_id": 1, "room_num": "1"}),
        # Correct type -> the move local search wants to make.
        Room({"room_id": 2, "capacity": 50, "room_type": "classroom",
              "building": "M8", "dept_id": 1, "room_num": "2"}),
    ]

    def section(course_id, course_type, section_no, instructor_ids):
        return Section({
            "courses": {"course_id": course_id, "name": f"C{course_id}",
                        "course_type": course_type, "dept_id": 1,
                        "level": "Master", "course_class": None},
            "instructor_id": instructor_ids[0] if instructor_ids else None,
            "section": section_no, "sec_capacity": 30,
        }, instructor_ids=instructor_ids)

    sections = [
        section(1, "Lecture Graduate", "61", ["X"]),
        section(2, "Clinical Practice", "71", ["X"]),   # time-only, no room
    ]

    def item(course_id, course_type, section_no, room_id, timeslot_id):
        return ScheduleItem(
            course_id=course_id, course_name=f"C{course_id}",
            course_type=course_type, course_dept=1, capacity=30,
            instructor_id="X", room_id=room_id, timeslot_id=timeslot_id,
            section=section_no, level="Master", instructor_ids=["X"],
        )

    schedule = [
        item(1, "Lecture Graduate", "61", 1, 1),      # roomed, badly typed room
        item(2, "Clinical Practice", "71", None, 2),  # roomless, same instructor
    ]
    assert count_instructor_conflicts(schedule) == 0, "scenario must start clean"

    cache = build_timeslot_guideline_cache(sections, timeslots)
    improved, _ = local_search(schedule, rooms, timeslots, sections, cache)

    assert count_instructor_conflicts(improved) == 0, (
        "local search moved a class onto an hour its instructor was already "
        "teaching a roomless class"
    )


def test_greedy_carries_level_into_every_item(dataset):
    """greedy was the last algorithm still dropping `level`. Fixed 2026-08-18.

    Without it, every greedy item classifies as NEEDS_ROOM_AND_TIME downstream
    -- so a Master's Laboratory or a thesis would be handed a room it does not
    need, and the level-aware classification fix quietly does nothing.
    """
    from backend.Optimization.Algorithims.greedy import greedy_schedule

    sections, timeslots, rooms = dataset
    schedule = greedy_schedule(sections, timeslots, rooms)

    assert schedule
    levels = {item.course_id: item.level for item in schedule}
    expected = {section.course.id: section.course.level for section in sections}
    assert levels == expected
    assert all(item.level is not None for item in schedule)


def test_greedy_respects_classification(dataset):
    """Sections that need no room must not be given one."""
    from backend.Optimization.Algorithims.greedy import greedy_schedule
    from backend.Optimization.constraints import needs_for

    sections, timeslots, rooms = dataset
    schedule = greedy_schedule(sections, timeslots, rooms)

    for item in schedule:
        needs = needs_for(item.course_type, item.level)
        if not needs.room:
            assert item.room_id is None, (
                f"course {item.course_id} ({item.course_type}, {item.level}) "
                f"needs no room but got {item.room_id}"
            )


def test_greedy_produces_no_hard_conflicts(dataset):
    from backend.Optimization.Algorithims.greedy import greedy_schedule
    from backend.Optimization.evaluation import count_hard_conflicts

    sections, timeslots, rooms = dataset
    assert count_hard_conflicts(greedy_schedule(sections, timeslots, rooms)) == 0


def test_supervision_does_not_consume_a_room_it_does_not_need(dataset):
    """The thesis section needs neither room nor time."""
    from backend.Optimization.Algorithims.greedy import greedy_schedule

    sections, timeslots, rooms = dataset
    schedule = greedy_schedule(sections, timeslots, rooms)

    thesis = next(i for i in schedule if i.course_id == 6)
    assert thesis.room_id is None
    assert thesis.timeslot_id is None
