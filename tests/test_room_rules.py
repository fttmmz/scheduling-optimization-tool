"""The hard room rules (constraints/rooms.py, 2026-09-30).

Three placements are forbidden outright: a class in the wrong KIND of room, a
lab in another department's lab, and either side crossing the medical-campus
line. Plus two registrar facts: branch-campus sections get a time and no room,
and a room must seat the larger of planned capacity and actual enrolment.

The department ids and campus codes are institution facts from
institution_data.py; tests read them from there rather than restating them,
and skip when an older copy of that file does not define them.
"""
import json
import pathlib

import pytest

from backend.models.models import Room, ScheduleItem, Section, Timeslot
from backend.Optimization.constraints import (
    get_viable_rooms,
    is_room_allowed,
    room_rule_violations,
    seats_needed,
    section_needs,
)
from backend.Optimization.constraints import rooms as room_rules

needs_rule_data = pytest.mark.skipif(
    not (room_rules.CAMPUS_CODE_TO_CAMPUS and room_rules.LAB_IN_CLASSROOM_DEPTS
         and room_rules.LAB_SHARING_DEPT_GROUPS and room_rules.BRANCH_CAMPUS_CODES),
    reason="institution_data.py predates the room rules",
)

MEDICAL_CODE = next((code for code, campus in room_rules.CAMPUS_CODE_TO_CAMPUS.items()
                     if campus == "MEDICAL"), None)
MAIN_CODE = next((code for code, campus in room_rules.CAMPUS_CODE_TO_CAMPUS.items()
                  if campus == "MAIN"), None)
BRANCH_CODE = next(iter(sorted(room_rules.BRANCH_CAMPUS_CODES)), None)

# A department that must use a lab room, and one whose labs may use a classroom.
EQUIPMENT_DEPT = 900001
NO_EQUIPMENT_DEPT = next(iter(sorted(room_rules.LAB_IN_CLASSROOM_DEPTS)), None)
FAMILY = sorted(next(iter(room_rules.LAB_SHARING_DEPT_GROUPS), ()))


def room(room_id=1, room_type="classroom", building="M8", dept_id=None, capacity=40):
    return Room({"room_id": room_id, "capacity": capacity, "room_type": room_type,
                 "building": building, "dept_id": dept_id, "room_num": str(room_id)})


def section(course_type="Lecture Undergraduate", dept=1, campus=None, enrolment=None,
            capacity=30, section_no="61", level="Undergraduate", course_id=1):
    row = {
        "courses": {"course_id": course_id, "name": f"C{course_id}",
                    "course_type": course_type, "dept_id": dept,
                    "level": level, "course_class": None},
        "instructor_id": None, "section": section_no, "sec_capacity": capacity,
    }
    return Section(row, info={"campus": campus, "enrolment": enrolment})


LAB = "Laboratory"


# ── room_type ────────────────────────────────────────────────────────────────

def test_lecture_may_not_use_a_lab():
    lecture = section()
    assert room_rule_violations(lecture, room(room_type="lab")) == ("room_type",)
    assert is_room_allowed(lecture, room(room_type="classroom"))


@needs_rule_data
def test_lab_needing_equipment_may_not_use_a_classroom():
    assert room_rule_violations(section(LAB, dept=EQUIPMENT_DEPT),
                                room(room_type="classroom")) == ("room_type",)


@needs_rule_data
def test_lab_without_equipment_may_use_a_classroom():
    assert is_room_allowed(section(LAB, dept=NO_EQUIPMENT_DEPT), room(room_type="classroom"))


PRACTICAL = next(iter(sorted(room_rules.PRACTICAL_LECTURE_COURSE_IDS)), None)
needs_practical = pytest.mark.skipif(PRACTICAL is None, reason="no practical course list")


@needs_practical
def test_practical_lecture_may_use_its_own_departments_lab():
    """Soft, not forbidden: the registrar teaches these courses in labs."""
    lecture = section(course_id=PRACTICAL, dept=55)
    assert is_room_allowed(lecture, room(room_type="lab", dept_id=55))
    assert is_room_allowed(lecture, room(room_type="lab", dept_id=None))
    # ...still at the soft room-type cost, so a classroom wins when one is free.
    from backend.Optimization.constraints import room_soft_penalty
    assert room_soft_penalty(lecture, room(room_type="lab", dept_id=55)) > \
        room_soft_penalty(lecture, room(room_type="classroom", dept_id=55))


@needs_practical
def test_practical_lecture_may_not_use_another_departments_lab():
    lecture = section(course_id=PRACTICAL, dept=55)
    assert room_rule_violations(lecture, room(room_type="lab", dept_id=54)) == ("room_type",)


@needs_rule_data
def test_lab_room_grant_is_limited_to_its_campus():
    """Pharmacy may use the CS computer labs on the medical campus, not elsewhere."""
    if not room_rules.LAB_ROOM_GRANTS:
        pytest.skip("no lab room grants")
    depts, owners, _campus = room_rules.LAB_ROOM_GRANTS[0]
    dept, owner = min(depts), min(owners)
    lab = section(LAB, dept=dept)
    assert is_room_allowed(lab, room(room_type="lab", building="M23", dept_id=owner))
    assert room_rule_violations(lab, room(room_type="lab", building="M10", dept_id=owner)) \
        == ("lab_department",)


def test_course_types_with_no_room_type_are_unconstrained():
    tutorial = section(course_type="Tutorial")
    assert is_room_allowed(tutorial, room(room_type="lab"))
    assert is_room_allowed(tutorial, room(room_type="classroom"))


# ── lab_department ───────────────────────────────────────────────────────────

def test_lab_may_not_use_another_departments_lab():
    """No Physics lab in a Chemistry lab."""
    physics = section(LAB, dept=55)
    assert room_rule_violations(physics, room(room_type="lab", dept_id=54)) == (
        "lab_department",)


def test_lab_may_use_its_own_or_an_open_lab():
    physics = section(LAB, dept=55)
    assert is_room_allowed(physics, room(room_type="lab", dept_id=55))
    assert is_room_allowed(physics, room(room_type="lab", dept_id=None))


@needs_rule_data
def test_lab_may_use_a_lab_of_its_sharing_family():
    member, owner = FAMILY[0], FAMILY[-1]
    assert is_room_allowed(section(LAB, dept=member), room(room_type="lab", dept_id=owner))


def test_department_is_still_only_soft_for_lectures():
    """The department rule is hard for LABS only; a lecture in another
    department's classroom is the soft cost it always was."""
    assert is_room_allowed(section(dept=1), room(dept_id=2))


# ── medical_campus ───────────────────────────────────────────────────────────

@needs_rule_data
def test_non_medical_section_may_not_use_a_medical_building():
    assert room_rule_violations(section(campus=MAIN_CODE), room(building="M23")) == (
        "medical_campus",)


@needs_rule_data
def test_medical_section_must_use_a_medical_building():
    medical = section(campus=MEDICAL_CODE)
    assert is_room_allowed(medical, room(building="M23"))
    assert room_rule_violations(medical, room(building="M10")) == ("medical_campus",)


@needs_rule_data
def test_medical_lab_may_use_the_main_campus_lab_buildings():
    for building in room_rules.MEDICAL_LAB_BUILDINGS:
        assert is_room_allowed(section(LAB, dept=EQUIPMENT_DEPT, campus=MEDICAL_CODE),
                               room(room_type="lab", building=building))
        # ...but a medical LECTURE may not follow it there.
        assert not is_room_allowed(section(campus=MEDICAL_CODE), room(building=building))


def test_unknown_campus_code_does_not_engage_the_medical_rule():
    """Without section_info the section number decides the campus, and it never
    says MEDICAL -- enforcing the rule then would empty the medical buildings."""
    assert is_room_allowed(section(campus=None), room(building="M23"))


@needs_rule_data
def test_items_are_judged_like_sections():
    """ScheduleItem.from_section must carry everything the rules read."""
    medical = section(campus=MEDICAL_CODE)
    item = ScheduleItem.from_section(medical)
    for candidate in (room(building="M23"), room(building="M10"), room(room_type="lab")):
        assert room_rule_violations(item, candidate) == room_rule_violations(medical, candidate)


# ── branch campuses and seats ────────────────────────────────────────────────

@needs_rule_data
def test_branch_campus_section_gets_a_time_but_no_room():
    needs = section_needs(section(campus=BRANCH_CODE))
    assert needs.time and not needs.room


def test_seats_needed_is_the_larger_of_capacity_and_enrolment():
    assert seats_needed(section(capacity=30, enrolment=45)) == 45
    assert seats_needed(section(capacity=30, enrolment=12)) == 30
    assert seats_needed(section(capacity=30, enrolment=None)) == 30


def test_viable_rooms_drop_forbidden_rooms_and_keep_the_rest():
    rooms = [room(1, room_type="lab"), room(2), room(3, dept_id=2)]
    assert [r.id for r in get_viable_rooms(section(dept=1), rooms)] == [2, 3]


# ── every algorithm respects the rules ───────────────────────────────────────

ALGORITHMS = ["greedy", "hybrid", "genetic", "grasp", "pso"]


def _rule_dataset():
    """Every allowed room is too SMALL, and every tempting forbidden room is big.

    Overflowing a room costs 6.0 soft; a forbidden room here costs at most 5.0
    soft (room type 4 + department 1). So on soft cost alone each section
    prefers a room it may not use, and only the hard filter keeps it out.
    Verified: with the filter disabled, the soft-ranked algorithms fail this.
    """
    rooms = [
        room(1, building="M23", capacity=10),                              # medical classroom
        room(2, building="M10", capacity=10),                              # main classroom
        room(3, room_type="lab", building="M12", dept_id=55, capacity=10),  # physics lab
        room(4, room_type="lab", building="M12", dept_id=54, capacity=60),  # chemistry lab
        room(5, room_type="lab", building="M23", dept_id=None, capacity=60),  # medical open lab
        room(6, room_type="lab", building="M10", dept_id=57, capacity=60),  # biology lab
    ]
    sections = [
        section(course_id=1, campus=MEDICAL_CODE),                 # 1 (not lab 5)
        section(course_id=2, campus=MAIN_CODE),                    # 2 (not lab 6)
        section(LAB, course_id=3, dept=55, campus=MAIN_CODE),      # 3 (not biology 6)
        section(LAB, course_id=4, dept=54, campus=MAIN_CODE),      # 4
        section(LAB, course_id=5, dept=55, campus=MEDICAL_CODE),   # 3 or 5
        section(course_id=6, campus=BRANCH_CODE),                  # time only
    ]
    for s in sections:
        s.instructor_ids = (f"I{s.course.id}",)
    timeslots = [
        Timeslot({"timeslot_id": tid, "day": day, "start_time": start, "end_time": end})
        for tid, (day, start, end) in enumerate(
            [(d, s, e) for d in ("MW", "TR", "M", "T", "W", "R")
             for s, e in (("08:00:00", "09:15:00"), ("10:00:00", "12:50:00"))], start=1)
    ]
    return sections, timeslots, rooms


@needs_rule_data
@pytest.mark.parametrize("name", ALGORITHMS)
def test_algorithms_never_use_a_forbidden_room(name):
    from test_algorithms import _run
    from backend.Optimization.evaluation import (
        count_hard_conflicts, count_room_rule_violations, count_scheduled_sections,
    )

    sections, timeslots, rooms = _rule_dataset()
    schedule = _run(name, sections, timeslots, rooms)

    assert count_room_rule_violations(schedule, rooms) == 0, name
    assert count_hard_conflicts(schedule, rooms) == 0, name
    assert count_scheduled_sections(schedule, sections) == len(sections), name

    by_course = {item.course_id: item for item in schedule}
    assert by_course[6].room_id is None and by_course[6].timeslot_id is not None, name


# ── the real dataset (skips without the local snapshot) ──────────────────────

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _real_data():
    paths = {n: FIXTURES / n for n in ("schedule_1.json", "rooms.json", "section_info.json")}
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        pytest.skip(f"missing fixtures {missing} -- run tests/snapshot_refresh.py")
    from backend.models.models import build_sections, section_info_index
    rows = json.loads(paths["schedule_1.json"].read_text(encoding="utf-8"))
    info = json.loads(paths["section_info.json"].read_text(encoding="utf-8"))
    if not info:
        pytest.skip("section_info snapshot is empty -- load it, then refresh")
    rooms = [Room(r) for r in json.loads(paths["rooms.json"].read_text(encoding="utf-8"))]
    return rows, build_sections(rows, section_info_index(info)), rooms


@needs_rule_data
def test_every_section_that_needs_a_room_has_an_allowed_one():
    """If this fails, some section CANNOT be scheduled under the room rules --
    a data or policy problem that no algorithm can fix. Lists them."""
    from backend.Optimization.constraints import sections_without_allowed_room

    _rows, sections, rooms = _real_data()
    stuck = sections_without_allowed_room(sections, rooms,
                                          lambda s: section_needs(s).room)
    assert not stuck, [(s.course.id, s.no, s.course.type, s.course.dept, s.campus)
                       for s in stuck[:20]]


@needs_rule_data
def test_manual_schedule_keeps_non_medical_sections_out_of_medical_buildings():
    """The registrar fact the medical rule rests on (import dry run, 2026-09-30)."""
    from backend.Optimization.constraints import get_building_campus, section_campus

    _rows, sections, rooms = _real_data()
    room_by_id = {r.id: r for r in rooms}
    from backend.models.models import group_rows_by_pattern
    info = {(s.course.id, s.no): s for s in sections}
    crossing = []
    for key, patterns in group_rows_by_pattern(_rows).items():
        sec = info.get(key)
        for pattern in patterns:
            r = room_by_id.get(pattern[0])
            if sec is None or r is None or sec.campus is None:
                continue
            if get_building_campus(r.building) == "MEDICAL" and section_campus(sec) != "MEDICAL":
                crossing.append((key, r.building, sec.campus))
    assert not crossing, crossing[:20]


@needs_practical
def test_practical_lecture_courses_match_the_manual():
    """PRACTICAL_LECTURE_COURSE_IDS is mined, not chosen: exactly the lecture
    courses the registrar placed in a lab their department may use. If the data
    changes, regenerate the list rather than editing it by hand."""
    from backend.Optimization.constraints import get_required_room_type
    from backend.models.models import group_rows_by_pattern

    rows, sections, rooms = _real_data()
    room_by_id = {r.id: r for r in rooms}
    by_id = {s.id: s for s in sections}
    mined = set()
    for (course_id, section_no), patterns in group_rows_by_pattern(rows).items():
        for index, pattern in enumerate(patterns):
            s, r = by_id[(course_id, section_no, index)], room_by_id.get(pattern[0])
            if (r is not None and r.type == "lab" and section_needs(s).room
                    and get_required_room_type(s.course.type) == "classroom"
                    and (not r.dept_id or room_rules._shares_labs(s.course.dept, r.dept_id))):
                mined.add(course_id)
    assert mined == set(room_rules.PRACTICAL_LECTURE_COURSE_IDS)
