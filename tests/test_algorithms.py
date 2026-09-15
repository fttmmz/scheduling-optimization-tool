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
        from backend.Optimization.Algorithims.hybrid import hybrid_schedule
        return hybrid_schedule(sections, timeslots, rooms)
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


# ── The evening preference (HANDOVER.md section 10) ──────────────────────────

def _pref_section(course_id, course_type, section_no, level="Undergraduate"):
    return Section({
        "courses": {"course_id": course_id, "name": f"C{course_id}",
                    "course_type": course_type, "dept_id": 1,
                    "level": level, "course_class": None},
        # A distinct instructor each, so an instructor clash can never be the
        # reason a section ends up on the wrong half of the day.
        "instructor_id": f"I{course_id}",
        "section": section_no, "sec_capacity": 30,
    }, instructor_ids=[f"I{course_id}"])


def _preference_dataset():
    """Built so ONLY the time preference can decide the placement.

    Three properties matter, and each was added after watching a mutation
    survive without it:

      * MORE THAN 15 valid slots per section. GRASP and PSO scan a
        random.sample() of the candidate slots when there are more than
        CONSTRUCTION_TIMESLOT_SAMPLE of them (15 and 8). Below that threshold no
        sampling happens, the ranked order of get_valid_timeslots() survives
        into the scan, and `max(candidates)` returns the first best slot -- which
        is a preferred one whether or not the scorer knows about time. A small
        dataset therefore passes even with the fix reverted. 24 slots per
        population forces the sample and destroys the ordering, which is exactly
        what happens on the real 466-slot dataset.

      * BOTH requirement shapes. Lectures and Senior Project Supervision need a
        room and a time; Thesis and Clinical Practice need a time and no room.
        Hybrid scores those two through different branches, and reverting the
        time term in one of them is invisible to a dataset that only exercises
        the other.

      * NO OTHER AXIS IN PLAY. Identical rooms, valid days and durations for
        every course type, capacity far exceeding demand, and equal numbers of
        daytime and evening slots. An algorithm that cannot see the preference
        is choosing by coin toss across 24 sections.
    """
    sections = []
    for i in range(10):     # daytime-preferring, needs room + time
        sections.append(_pref_section(100 + i, "Lecture Undergraduate", f"6{i}"))
    for i in range(6):      # evening-preferring, needs room + time
        sections.append(_pref_section(200 + i, "Senior Project Supervision", f"7{i}"))
    for i in range(4):      # evening-preferring, TIME ONLY
        sections.append(_pref_section(300 + i, "Thesis", f"8{i}", level="Master"))
    for i in range(4):      # daytime-preferring, TIME ONLY
        sections.append(
            _pref_section(400 + i, "Clinical Practice", f"9{i}", level="Master")
        )

    rooms = [
        Room({"room_id": i, "capacity": 200, "room_type": "classroom",
              "building": "M8", "dept_id": 1, "room_num": str(i)})
        for i in range(1, 21)
    ]

    DAY_STARTS = ("08:00:00", "09:30:00", "11:00:00",
                  "12:30:00", "14:00:00", "15:30:00")
    EVE_STARTS = ("17:00:00", "18:15:00", "19:30:00",
                  "20:45:00", "22:00:00", "23:15:00")

    def plus_75(start):
        hh, mm = int(start[:2]), int(start[3:5])
        mm += 75
        return f"{(hh + mm // 60) % 24:02d}:{mm % 60:02d}:00"

    timeslots, tid = [], 1
    # Lectures want MW/TR at 1h15m. 12 per day pattern, half of them evening.
    for day in ("MW", "TR"):
        for start in DAY_STARTS + EVE_STARTS:
            timeslots.append(Timeslot({"timeslot_id": tid, "day": day,
                                       "start_time": start,
                                       "end_time": plus_75(start)}))
            tid += 1
    # Supervision, thesis and clinical want a single day. 6 per day, half evening.
    for day in ("M", "T", "W", "R"):
        for start in DAY_STARTS[:3] + EVE_STARTS[:3]:
            timeslots.append(Timeslot({"timeslot_id": tid, "day": day,
                                       "start_time": start,
                                       "end_time": plus_75(start)}))
            tid += 1

    return sections, timeslots, rooms


@pytest.fixture(scope="module")
def preference_dataset():
    return _preference_dataset()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_honours_the_evening_preference(algorithm, preference_dataset):
    """Every algorithm must honour it, not only the ones that walk the list.

    Ranking get_valid_timeslots() was enough for greedy and genetic, which take
    the first FREE slot they are handed. GRASP, PSO and hybrid SAMPLE the
    room x timeslot space and keep the cheapest candidate, so until their
    candidate scorers priced the time axis they could not see it. On the full
    dataset those three placed 424-473 undergraduate classes in the evening and
    52-82 supervision sections in the daytime, while the fitness they were
    judged by charged for every one -- the schedules looked fine and scored
    badly for reasons nothing reported.
    """
    from backend.Optimization.constraints import (
        is_evening_timeslot, prefers_daytime, prefers_evening,
    )

    sections, timeslots, rooms = preference_dataset
    schedule = _run(algorithm, sections, timeslots, rooms)
    timeslot_map = {ts.id: ts for ts in timeslots}

    misplaced = []
    for item in schedule:
        ts = timeslot_map.get(item.timeslot_id)
        if ts is None:
            continue
        wrong = (
            (prefers_daytime(item) and is_evening_timeslot(ts))
            or (prefers_evening(item) and not is_evening_timeslot(ts))
        )
        if wrong:
            misplaced.append(
                f"{item.course_id} ({item.course_type}) at {ts.start}"
            )

    assert not misplaced, f"{algorithm} ignored the time preference: {misplaced}"


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_the_preference_never_costs_a_placement(algorithm, preference_dataset):
    """A preference must not strand a section -- the point of ranking not filtering.

    Unscheduled costs 2.0 against at most 0.3 of soft penalty, so an algorithm
    that runs out of preferred slots has to spill into the rest rather than give
    up. The rule this replaced hard-filtered supervision to slots at/after 15:00
    and could not spill at all.
    """
    from backend.Optimization.constraints import section_needs

    sections, timeslots, rooms = preference_dataset
    schedule = _run(algorithm, sections, timeslots, rooms)

    expected = {
        (str(s.course.id), str(s.no))
        for s in sections
        if section_needs(s).time
    }
    placed = {
        (str(item.course_id), str(item.section))
        for item in schedule
        if item.timeslot_id is not None
    }
    assert expected - placed == set(), f"{algorithm} left these unplaced"
