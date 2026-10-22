"""The registrar practice (constraints/registrar.py): learned from a base schedule, off unless loaded.

Small hand-made base schedule; each test checks one rule against what the same call returns with no
practice loaded, so a hook that leaks into the default model fails here too.
"""
import pytest

from backend.models.models import Room, Section, Timeslot
from backend.Optimization import constraints as C
from backend.Optimization.evaluation import build_timeslot_guideline_cache


def slot(tid, day, start, end):
    return Timeslot({"timeslot_id": tid, "day": day, "start_time": start, "end_time": end})


# Ids far from the other test modules' fixtures (the slot registry is shared).
MW_0800 = slot(8101, "MW", "08:00:00", "09:15:00")      # standard period
MW_0805 = slot(8102, "MW", "08:05:00", "09:20:00")      # near-duplicate of it
MW_0830 = slot(8103, "MW", "08:30:00", "09:45:00")      # off-grid one-off
TR_0930 = slot(8104, "TR", "09:30:00", "10:45:00")
M_0800_LAB = slot(8105, "M", "08:00:00", "10:50:00")    # lab block
T_1400_TUT = slot(8106, "T", "14:00:00", "14:50:00")
T_1700_TUT = slot(8107, "T", "17:00:00", "17:50:00")
TIMESLOTS = [MW_0800, MW_0805, MW_0830, TR_0930, M_0800_LAB, T_1400_TUT, T_1700_TUT]

ROOMS = [
    Room({"room_id": 1, "capacity": 40, "room_type": "classroom", "building": "M8", "dept_id": 1, "room_num": "1"}),
    Room({"room_id": 2, "capacity": 40, "room_type": "classroom", "building": "M8", "dept_id": 1, "room_num": "2"}),
    Room({"room_id": 3, "capacity": 40, "room_type": "lab", "building": "M8", "dept_id": 2, "room_num": "3"}),
]


def section(course_id, no, course_type="Lecture Undergraduate", level="Undergraduate", dept=1):
    row = {"courses": {"course_id": course_id, "name": f"Course {course_id}", "course_type": course_type,
                       "dept_id": dept, "level": level, "course_class": None},
           "instructor_id": None, "section": no, "sec_capacity": 20}
    return Section(row, instructor_ids=[])


# (section, manual slot, manual room). Course 7001's lab sections ("61L", "62L") sit in dept 2's lab,
# twice -- the registrar's practice for that course.
BASE = [
    (section(7001, "61"), MW_0800.id, 1),
    (section(7001, "62"), MW_0805.id, 2),
    (section(7001, "61L"), M_0800_LAB.id, 3),
    (section(7001, "62L"), M_0800_LAB.id, 3),
    (section(7002, "61"), TR_0930.id, 1),
    (section(7002, "62"), TR_0930.id, 2),
    (section(7003, "61T"), T_1700_TUT.id, 1),
    (section(7004, "61"), None, None),                   # TBA in the base schedule
    (section(7005, "61"), MW_0830.id, None),             # a time, no room
]
SECTIONS = [s for s, _, _ in BASE]


@pytest.fixture
def practice():
    C.activate_registrar_practice(None)
    learned = C.learn_registrar_practice(SECTIONS, TIMESLOTS, ROOMS, [t for _, t, _ in BASE],
                                         [r for _, _, r in BASE])
    yield C.RegistrarPractice(learned)
    C.activate_registrar_practice(None)


def _with(practice, fn):
    C.activate_registrar_practice(practice)
    try:
        return fn()
    finally:
        C.activate_registrar_practice(None)


def ids(slots):
    return {t.id for t in slots}


def test_nothing_changes_without_a_practice():
    assert C.active_registrar_practice() is None
    lecture = SECTIONS[0]
    assert MW_0830.id in ids(C.get_valid_timeslots_for_section(lecture, TIMESLOTS))
    assert C.section_needs(SECTIONS[7]) == C.Needs(room=True, time=True)


def test_slots_are_the_registrar_grid(practice):
    lecture = SECTIONS[0]
    allowed = _with(practice, lambda: ids(C.get_valid_timeslots_for_section(lecture, TIMESLOTS)))
    assert MW_0800.id in allowed
    assert MW_0830.id not in allowed            # the one-off
    assert MW_0805.id not in allowed            # its near-duplicate: one slot per family


def test_a_near_duplicate_is_not_charged(practice):
    cache = _with(practice, lambda: build_timeslot_guideline_cache(SECTIONS, TIMESLOTS))
    assert MW_0805.id in cache[(7001, "62")]


def test_an_L_section_of_a_lecture_course_is_a_lab_in_its_course_lab(practice):
    lab_section = SECTIONS[2]
    assert 3 not in {r.id for r in C.get_viable_rooms(lab_section, ROOMS)}       # production: a lecture
    rooms = _with(practice, lambda: [r.id for r in C.get_viable_rooms(lab_section, ROOMS)])
    assert rooms == [3]                         # its lab, another department's: the course's practice


def test_room_type_is_not_charged(practice):
    lecture = SECTIONS[0]
    parts = _with(practice, lambda: C.room_soft_penalty_parts(lecture, ROOMS[2]))
    assert parts["room_type"] == 0.0


def test_demand_follows_the_base_schedule(practice):
    tba, roomless = SECTIONS[7], SECTIONS[8]
    assert _with(practice, lambda: C.section_needs(tba)) == C.Needs(room=False, time=False)
    assert _with(practice, lambda: C.section_needs(roomless)) == C.Needs(room=False, time=True)


def test_tutorials_meet_once_and_prefer_the_evening(practice):
    tutorial = SECTIONS[6]
    allowed = _with(practice, lambda: ids(C.get_valid_timeslots_for_section(tutorial, TIMESLOTS)))
    assert allowed <= {T_1400_TUT.id, T_1700_TUT.id}
    day = _with(practice, lambda: C.time_soft_penalty(tutorial, T_1400_TUT))
    evening = _with(practice, lambda: C.time_soft_penalty(tutorial, T_1700_TUT))
    assert evening == 0.0 < day


def test_the_practice_round_trips_through_json(practice):
    import json
    again = C.RegistrarPractice(json.loads(json.dumps(practice.data)))
    lecture = SECTIONS[0]
    first = _with(practice, lambda: ids(C.get_valid_timeslots_for_section(lecture, TIMESLOTS)))
    second = _with(again, lambda: ids(C.get_valid_timeslots_for_section(lecture, TIMESLOTS)))
    assert first == second
