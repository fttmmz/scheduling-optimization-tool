"""Pins the section-classification table and its helpers (constraints.py).

The classification fix (commit 21ba364) is level-aware: the SAME course_type
needs different things at different levels. That is easy to undo by accident --
it already silently regressed twice when `level` was dropped while copying a
Section into a ScheduleItem (see HANDOVER.md section 6). These tests fail loudly
when it does.
"""
import pytest

from backend.Optimization.constraints import (
    DEFAULT_NEEDS,
    NEEDS_NOTHING,
    NEEDS_ROOM_AND_TIME,
    NEEDS_TIME_ONLY,
    Needs,
    capacity_penalty,
    classify_section,
    combined_group_key,
    get_building_campus,
    get_required_room_type,
    get_section_campus,
    is_placeholder_timeslot,
    needs_for,
    occupancy_key,
)
from backend.models.models import Section, Timeslot


def make_section(course_type, level, section_no="61", capacity=30,
                 course_id=1, instructor_id=None, name="Test Course"):
    return Section({
        "courses": {
            "course_id": course_id,
            "name": name,
            "course_type": course_type,
            "dept_id": 1,
            "level": level,
            "course_class": None,
        },
        "instructor_id": instructor_id,
        "section": section_no,
        "sec_capacity": capacity,
    })


# ── Level-awareness is the whole point ───────────────────────────────────────

def test_laboratory_needs_depend_on_level():
    """An undergraduate lab needs a real room; a graduate 'lab' is research."""
    assert needs_for("Laboratory", "Undergraduate") == Needs(room=True, time=True)
    assert needs_for("Laboratory", "Master") == Needs(room=False, time=True)


def test_thesis_needs_nothing_at_doctorate_level():
    assert needs_for("Thesis / Dissertation Doctorat", "Doctorate") == Needs(False, False)
    assert needs_for("Thesis", "Doctorate") == Needs(False, False)
    # ...but a Master's 'Thesis' still meets at a scheduled hour.
    assert needs_for("Thesis", "Master") == Needs(False, True)


def test_clinical_practice_needs_a_time_but_no_room():
    """The fourth category added by commit 21ba364.

    Corroborated by the data: 226 of 232 Clinical Practice rows have no room,
    yet most carry a real meeting time (test_data_invariants covers the counts).
    """
    for level in ("Master", "Undergraduate"):
        assert needs_for("Clinical Practice", level) == Needs(room=False, time=True)


def test_unknown_combinations_fall_back_to_room_and_time():
    """Safe default: an unrecognised section is scheduled, never silently dropped."""
    assert needs_for("Some Brand New Type", "Undergraduate") == DEFAULT_NEEDS
    assert needs_for("Laboratory", None) == DEFAULT_NEEDS
    assert DEFAULT_NEEDS == Needs(room=True, time=True)


def test_missing_level_does_not_grant_a_graduate_exemption():
    """A dropped `level` must fail SAFE -- toward scheduling, not away from it.

    This is the regression that bit twice: an item detached from its Section
    classifies as an ordinary lecture. That over-schedules (wasteful but
    visible) rather than under-schedules (silent data loss).
    """
    assert needs_for("Laboratory", "Master").room is False
    assert needs_for("Laboratory", None).room is True


@pytest.mark.parametrize("course_type,level,expected", [
    ("Lecture Undergraduate", "Undergraduate", NEEDS_ROOM_AND_TIME),
    ("Clinical Practice", "Undergraduate", NEEDS_TIME_ONLY),
    ("Thesis / Dissertation Master", "Master", NEEDS_NOTHING),
    ("Office Hours", "Undergraduate", NEEDS_TIME_ONLY),
])
def test_classify_section_labels(course_type, level, expected):
    assert classify_section(make_section(course_type, level)) == expected


# ── Placeholder timeslots (two encodings, both mean "no fixed time") ─────────

def test_placeholder_timeslot_detection():
    def ts(day, start, end):
        return Timeslot({"timeslot_id": 1, "day": day,
                         "start_time": start, "end_time": end})

    assert is_placeholder_timeslot(ts(None, "08:00:00", "09:15:00")) is True   # TBA
    assert is_placeholder_timeslot(ts("M", "08:00:00", "08:00:00")) is True    # zero span
    assert is_placeholder_timeslot(ts("M", "08:00:00", "09:15:00")) is False


def test_multi_day_timeslot_is_not_a_placeholder():
    """'MTR' is one row covering three days -- a real slot (HANDOVER.md 4.5)."""
    ts = Timeslot({"timeslot_id": 207, "day": "MTR",
                   "start_time": "08:00:00", "end_time": "11:59:00"})
    assert is_placeholder_timeslot(ts) is False


# ── Combined / cross-listed groups ───────────────────────────────────────────

def test_combined_courses_share_an_occupancy_key():
    assert combined_group_key(405341) == combined_group_key(405342)
    assert occupancy_key(405341, "61") == occupancy_key(405342, "61")


def test_unrelated_courses_do_not_share_an_occupancy_key():
    assert combined_group_key(401314) != combined_group_key(405341)
    assert occupancy_key(1, "61") != occupancy_key(2, "61")


def test_different_sections_of_one_course_stay_distinct():
    """The collapse must not become a licence to double-book."""
    assert occupancy_key(405341, "61") == occupancy_key(405341, "62")  # same group key
    # ...but _count_occupants disambiguates by course within the group; see
    # test_evaluation.test_combined_group_does_not_hide_a_real_clash.
    assert occupancy_key(999999, "61") != occupancy_key(999999, "62")


def test_uncorroborated_colocations_are_not_combined():
    """Recorded for transparency, deliberately NOT excused."""
    assert combined_group_key(306620) is None
    assert combined_group_key(306622) is None


# ── Room-type mapping and capacity pricing ───────────────────────────────────

def test_required_room_type_mapping():
    assert get_required_room_type("Laboratory") == "lab"
    assert get_required_room_type("Lecture Undergraduate") == "classroom"
    assert get_required_room_type("Clinical Practice") is None


def test_zero_capacity_room_is_unknown_not_free():
    """capacity 0 means 'unknown', not 'a room with no seats'.

    It must never outrank a real room by scoring as a perfect fit.
    """
    assert capacity_penalty(30, 0) > 0
    assert capacity_penalty(30, 50) == 0.0
    assert capacity_penalty(30, 0) >= capacity_penalty(30, 29)


def test_capacity_penalty_is_graded():
    """5 seats short is not 50 short."""
    assert capacity_penalty(30, 25) < capacity_penalty(30, 10)


# ── Campus derivation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("section_no,expected", [
    ("05", "MEN"), ("35", "WOMEN"), ("61", "MAIN"),
    ("05T", "MEN"), ("35X", "WOMEN"),
])
def test_section_campus(section_no, expected):
    assert get_section_campus(section_no) == expected


@pytest.mark.parametrize("building,expected", [
    ("M3", "MEN"), ("W3", "WOMEN"), ("M8", "MAIN"), ("W8", "MAIN"),
    ("M23", "MEDICAL"),
])
def test_building_campus(building, expected):
    assert get_building_campus(building) == expected


def test_annex_buildings_follow_their_base_building():
    """'M3A' is the M3 annex -- MEN, not a separate medical campus.

    Before the fix int('3A') raised and every annex silently became MEDICAL.
    """
    assert get_building_campus("M3A") == get_building_campus("M3") == "MEN"
    assert get_building_campus("W7A") == get_building_campus("W7") == "MAIN"
