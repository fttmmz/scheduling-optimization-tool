"""Supervision vs teaching (constraints/classification.py).

The instructor column mixes two relationships. TEACHING is physical -- one body,
one room, one hour. SUPERVISION is an attachment between a staff member and a
student's thesis or project; twenty-two of them at one nominal hour is a roster,
not twenty-two people in a room.

These tests pin the detector and the one behaviour that depends on it: a
supervision row must not create instructor occupancy.
"""
import pytest

from backend.Optimization.constraints import (
    SUPERVISION_COURSE_IDS,
    SUPERVISION_COURSE_TYPES,
    instructor_occupancy_id,
    is_supervision,
    passes_hard_constraints,
)
from backend.Optimization.evaluation import (
    count_hard_conflicts,
    count_instructor_conflicts,
)
from backend.models.models import Room, ScheduleItem, Section, Timeslot

TS = 100


def make_item(course_id=1, course_type="Lecture Undergraduate",
              instructor_id="1556", section="61", timeslot_id=TS, room_id=None):
    return ScheduleItem(
        course_id=course_id, course_name="Test", course_type=course_type,
        course_dept=1, capacity=30, instructor_id=instructor_id,
        room_id=room_id, timeslot_id=timeslot_id, section=section,
        level="Undergraduate",
    )


def make_section(course_id=1, course_type="Lecture Undergraduate",
                 instructor_id="1556", section="61"):
    return Section({
        "courses": {"course_id": course_id, "name": "Test",
                    "course_type": course_type, "dept_id": 1,
                    "level": "Undergraduate", "course_class": None},
        "instructor_id": instructor_id, "section": section, "sec_capacity": 30,
    })


# ── Detection ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("course_type", sorted(SUPERVISION_COURSE_TYPES))
def test_supervision_course_types_are_detected(course_type):
    assert is_supervision(make_item(course_type=course_type)) is True
    assert is_supervision(make_section(course_type=course_type)) is True


@pytest.mark.parametrize("course_type", [
    "Lecture Undergraduate", "Lecture Graduate", "Laboratory",
    "Clinical Practice", "Studio Undergraduate", "Tutorial",
])
def test_teaching_course_types_are_not_supervision(course_type):
    assert is_supervision(make_item(course_type=course_type)) is False


def test_miscategorised_courses_are_detected_by_id():
    """1501497 'Senior Project in Bioinfor.' is typed as an ordinary lecture.

    Caught by course id, because its type is wrong and its name alone is not
    evidence -- see the next test.
    """
    assert 1501497 in SUPERVISION_COURSE_IDS
    item = make_item(course_id=1501497, course_type="Lecture Undergraduate")
    assert is_supervision(item) is True


def test_project_management_is_a_real_course_not_supervision():
    """The reason name matching alone was rejected.

    'Project Management' contains a supervision keyword and is an ordinary
    taught lecture with a room and a meeting time. Detecting on name would have
    silently stopped counting its instructor's clashes.
    """
    for course_id in (1503431, 405561, 404438, 307530):
        item = make_item(course_id=course_id, course_type="Lecture Undergraduate")
        assert is_supervision(item) is False
        assert instructor_occupancy_id(item) == "1556"


def test_intensive_english_team_teaching_is_not_supervision():
    """Five instructors on one class, in a real room -- genuine team teaching.

    This is why roster WIDTH was rejected as a signal: width 5 covers both this
    and an M.Sc. thesis roster.
    """
    item = make_item(course_id=11100, course_type="Lecture Undergraduate",
                     room_id=316)
    assert is_supervision(item) is False


# ── Occupancy identity ───────────────────────────────────────────────────────

def test_teaching_yields_the_instructor_id():
    assert instructor_occupancy_id(make_item()) == "1556"


def test_supervision_yields_no_occupancy():
    item = make_item(course_type="Senior Project Supervision")
    assert item.instructor_id == "1556", "the item still REPORTS its instructor"
    assert instructor_occupancy_id(item) is None


def test_missing_instructor_yields_no_occupancy():
    assert instructor_occupancy_id(make_item(instructor_id=None)) is None


# ── The behaviour that matters ───────────────────────────────────────────────

def test_supervisors_on_one_project_are_not_a_double_booking():
    """22 supervisors on one senior project at one hour: zero conflicts.

    Post section-model rework this is ONE entity carrying 22 instructor_ids --
    the 22 source rows collapse. Under the old row-per-instructor model it was
    22 entities scoring 21 hard conflicts, and the hard tier stopped meaning
    'physically impossible'.
    """
    item = make_item(course_id=1501494, course_type="Senior Project Supervision",
                     section="61")
    item.instructor_ids = tuple(str(1556 + n) for n in range(22))

    assert count_instructor_conflicts([item]) == 0
    assert count_hard_conflicts([item]) == 0


def test_one_lecturer_teaching_two_classes_at_once_is_still_a_conflict():
    """The exemption must not leak into ordinary teaching."""
    schedule = [
        make_item(course_id=1, instructor_id="1556", section="61"),
        make_item(course_id=2, instructor_id="1556", section="61"),
    ]
    assert count_instructor_conflicts(schedule) == 1


def test_a_supervisor_may_also_teach_at_the_same_hour():
    """Supervision does not block the supervisor's real teaching.

    The same person supervises a thesis and teaches a lecture at one nominal
    hour. Only the lecture is a physical commitment, so this is not a clash --
    and previously it was, which cost the schedule a genuine placement.
    """
    schedule = [
        make_item(course_id=1, course_type="Thesis / Dissertation Master",
                  instructor_id="1556"),
        make_item(course_id=2, course_type="Lecture Undergraduate",
                  instructor_id="1556"),
    ]
    assert count_instructor_conflicts(schedule) == 0


def test_supervision_still_double_books_a_ROOM():
    """Only the instructor axis is exempt. Rooms remain physical.

    Two supervision groups genuinely cannot share one room at one hour, and the
    room check is what still catches the group-session cases the instructor
    exemption gives up on.
    """
    schedule = [
        make_item(course_id=1, course_type="Senior Project Supervision",
                  section="61", room_id=7),
        make_item(course_id=2, course_type="Senior Project Supervision",
                  section="61", room_id=7),
    ]
    assert count_hard_conflicts(schedule) == 1


def test_passes_hard_constraints_lets_supervision_share_an_hour():
    """The shared gate agrees with the scorer -- they must never disagree."""
    timeslot = Timeslot({"timeslot_id": TS, "day": "M",
                         "start_time": "19:00:00", "end_time": "21:30:00"})
    room = Room({"room_id": 7, "capacity": 50, "room_type": "classroom",
                 "building": "M8", "dept_id": 1, "room_num": "7"})
    occupied = {("1556", TS)}

    supervision = make_section(course_type="Senior Project Supervision",
                               instructor_id="1556")
    teaching = make_section(course_type="Lecture Undergraduate",
                            instructor_id="1556")

    assert passes_hard_constraints(
        supervision, room, timeslot,
        occupied_instructors=occupied, occupied_rooms=set(),
    ) is True
    assert passes_hard_constraints(
        teaching, room, timeslot,
        occupied_instructors=occupied, occupied_rooms=set(),
    ) is False
