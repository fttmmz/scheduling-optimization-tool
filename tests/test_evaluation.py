"""Pins the tiered scorer (evaluation.py).

The hard/soft split is the main result of the constraint rework: HARD means
physically impossible, SOFT means imperfect but runnable. The failure mode these
tests exist to catch is the tiers quietly collapsing back into one flat count --
which is what `count_all_violations_flat` still does, and why it must never gate anything.
"""
import pytest

from backend.Optimization.evaluation import (
    HARD_CONFLICT_WEIGHT,
    SOFT_PENALTY_WEIGHT,
    UNSCHEDULED_WEIGHT,
    calculate_fitness,
    count_all_violations_flat,
    count_hard_conflicts,
    count_instructor_conflicts,
    count_room_conflicts,
    count_room_rule_violations,
    soft_violation_counts,
    total_soft_penalty,
)
from backend.models.models import Room, ScheduleItem

TS = 100          # an arbitrary timeslot id
OTHER_TS = 101


def item(course_id=1, section="61", instructor_id=None, room_id=None,
         timeslot_id=TS, capacity=30, course_type="Lecture Undergraduate",
         course_dept=1, level="Undergraduate"):
    return ScheduleItem(
        course_id=course_id,
        course_name="Test Course",
        course_type=course_type,
        course_dept=course_dept,
        capacity=capacity,
        instructor_id=instructor_id,
        room_id=room_id,
        timeslot_id=timeslot_id,
        section=section,
        level=level,
    )


def room(room_id=1, capacity=50, room_type="classroom", building="M8", dept_id=1):
    return Room({
        "room_id": room_id, "capacity": capacity, "room_type": room_type,
        "building": building, "dept_id": dept_id, "room_num": str(room_id),
    })


# ── Room double-booking ──────────────────────────────────────────────────────

def test_two_classes_in_one_room_at_one_time_is_one_conflict():
    schedule = [
        item(course_id=1, room_id=7, timeslot_id=TS),
        item(course_id=2, room_id=7, timeslot_id=TS),
    ]
    assert count_room_conflicts(schedule) == 1


def test_three_classes_in_one_room_is_two_conflicts():
    schedule = [item(course_id=i, room_id=7, timeslot_id=TS) for i in (1, 2, 3)]
    assert count_room_conflicts(schedule) == 2


def test_same_room_different_times_is_fine():
    schedule = [
        item(course_id=1, room_id=7, timeslot_id=TS),
        item(course_id=2, room_id=7, timeslot_id=OTHER_TS),
    ]
    assert count_room_conflicts(schedule) == 0


def test_unassigned_room_never_conflicts():
    """A section with no room cannot double-book one."""
    schedule = [item(course_id=i, room_id=None, timeslot_id=TS) for i in range(50)]
    assert count_room_conflicts(schedule) == 0


# ── Instructor double-booking, and the NULL-instructor majority ──────────────

def test_one_instructor_in_two_places_is_a_conflict():
    schedule = [
        item(course_id=1, instructor_id="1556", timeslot_id=TS),
        item(course_id=2, instructor_id="1556", timeslot_id=TS),
    ]
    assert count_instructor_conflicts(schedule) == 1


def test_null_instructors_never_conflict():
    """56% of real rows have no instructor (HANDOVER.md 4.3).

    If None were treated as a person, those 2,614 rows would collapse into one
    phantom instructor double-booked thousands of times and the hard tier would
    become meaningless. This is the single most load-bearing guard in the file.
    """
    schedule = [item(course_id=i, instructor_id=None, timeslot_id=TS)
                for i in range(100)]
    assert count_instructor_conflicts(schedule) == 0
    assert count_hard_conflicts(schedule) == 0


def test_null_timeslot_never_conflicts():
    schedule = [item(course_id=i, instructor_id="1556", timeslot_id=None)
                for i in range(10)]
    assert count_instructor_conflicts(schedule) == 0


# ── Cross-listed collapse, and its limits ────────────────────────────────────

def test_combined_courses_sharing_a_room_are_not_a_conflict():
    """405341/405342 are one class taught under two codes."""
    schedule = [
        item(course_id=405341, section="61", room_id=7, timeslot_id=TS),
        item(course_id=405342, section="61", room_id=7, timeslot_id=TS),
    ]
    assert count_room_conflicts(schedule) == 0


def test_combined_group_does_not_hide_a_real_clash():
    """Two SECTIONS of the same course in one room is still a double-booking.

    Without this the collapse would be a licence to hide any clash by finding a
    cross-listed course code to hang it on.
    """
    schedule = [
        item(course_id=405341, section="61", room_id=7, timeslot_id=TS),
        item(course_id=405341, section="62", room_id=7, timeslot_id=TS),
    ]
    assert count_room_conflicts(schedule) == 1


def test_uncombined_courses_sharing_a_room_still_conflict():
    schedule = [
        item(course_id=306620, section="61", room_id=7, timeslot_id=TS),
        item(course_id=306622, section="61", room_id=7, timeslot_id=TS),
    ]
    assert count_room_conflicts(schedule) == 1


# ── The tiers must not collapse into each other ──────────────────────────────

def test_soft_rules_are_not_hard():
    """A wrong-campus, wrong-department, over-capacity placement is NOT hard.

    It is exactly the case the old flat scorer treated as equivalent to a
    double-booking, which is what produced the unscheduled ceiling.
    """
    rooms = [room(room_id=7, capacity=10, room_type="classroom", building="W3",
                  dept_id=99)]
    schedule = [item(course_id=1, section="05", room_id=7, capacity=200,
                     course_type="Lecture Undergraduate", course_dept=1)]

    assert count_hard_conflicts(schedule, rooms) == 0
    assert sum(soft_violation_counts(schedule, rooms).values()) >= 3
    assert total_soft_penalty(schedule, rooms) > 0


def test_lecture_in_a_lab_is_hard():
    """Since 2026-09-30 a lecture may not be taught in a lab room at all."""
    rooms = [room(room_id=7, capacity=50, room_type="lab", building="M8", dept_id=1)]
    schedule = [item(course_id=1, section="61", room_id=7)]

    assert count_room_rule_violations(schedule, rooms) == 1
    assert count_hard_conflicts(schedule, rooms) == 1
    # Without rooms the room rules cannot be checked -- callers must pass them.
    assert count_hard_conflicts(schedule) == 0


def test_soft_violations_are_counted_per_rule():
    rooms = [room(room_id=7, capacity=10, room_type="lab", building="W3", dept_id=99)]
    schedule = [item(course_id=1, section="05", room_id=7, capacity=200)]

    counts = soft_violation_counts(schedule, rooms)
    assert counts["room_type"] == 1     # classroom course in a lab
    assert counts["campus"] == 1        # MEN section in a WOMEN building
    assert counts["department"] == 1    # dept 1 course in a dept 99 room
    assert counts["capacity"] == 1      # 200 into 10


def test_perfect_placement_has_no_soft_penalty():
    rooms = [room(room_id=7, capacity=50, room_type="classroom",
                  building="M8", dept_id=1)]
    schedule = [item(course_id=1, section="61", room_id=7, capacity=30)]
    assert total_soft_penalty(schedule, rooms) == 0.0


def test_count_all_violations_flat_flattens_the_tiers_and_is_display_only():
    """Documents the trap rather than the intent (HANDOVER.md section 6).

    count_all_violations_flat sums hard and soft at equal weight, so it cannot distinguish
    an impossible schedule from a merely imperfect one. Retained for the UI;
    never a feasibility test.
    """
    rooms = [
        room(room_id=7, capacity=50, room_type="classroom", building="M8", dept_id=1),
        room(room_id=8, capacity=10, room_type="classroom", building="W3", dept_id=99),
    ]
    # One section in a thoroughly wrong room: runnable, just bad.
    imperfect = [item(course_id=1, section="05", room_id=8, capacity=200)]
    # Two sections in one room at one time, otherwise flawless: cannot be run.
    impossible = [
        item(course_id=1, section="61", room_id=7, timeslot_id=TS),
        item(course_id=2, section="61", room_id=7, timeslot_id=TS),
    ]

    assert count_hard_conflicts(imperfect) == 0
    assert count_hard_conflicts(impossible) == 1
    assert total_soft_penalty(impossible, rooms) == 0.0

    # ...yet the flat count rates the runnable schedule as three times worse.
    assert count_all_violations_flat(imperfect, rooms) > count_all_violations_flat(impossible, rooms)

    # The tiered fitness gets the ordering right, which is the whole point.
    assert calculate_fitness(impossible, rooms, total_sections=2) < \
        calculate_fitness(imperfect, rooms, total_sections=1)


# ── Fitness ordering: the tiers must be priced so they never trade wrongly ───

def test_tier_weights_are_ordered():
    assert HARD_CONFLICT_WEIGHT > UNSCHEDULED_WEIGHT > SOFT_PENALTY_WEIGHT


def test_a_double_booking_costs_more_than_an_imperfect_room():
    """The core of the rework: never break a room to satisfy a soft rule."""
    rooms = [
        room(room_id=7, capacity=50, room_type="classroom", building="M8", dept_id=1),
        room(room_id=8, capacity=10, room_type="classroom", building="W3", dept_id=99),
    ]
    clash = [
        item(course_id=1, section="61", room_id=7, timeslot_id=TS),
        item(course_id=2, section="61", room_id=7, timeslot_id=TS),
    ]
    compromise = [
        item(course_id=1, section="61", room_id=7, timeslot_id=TS),
        item(course_id=2, section="05", room_id=8, timeslot_id=TS, capacity=200),
    ]

    assert calculate_fitness(clash, rooms, total_sections=2) < \
        calculate_fitness(compromise, rooms, total_sections=2)


def test_an_unscheduled_section_costs_more_than_an_imperfect_room():
    """A missing class is worse than a compromised one, but better than a clash."""
    rooms = [
        room(room_id=7, capacity=50, room_type="classroom", building="M8", dept_id=1),
        room(room_id=8, capacity=10, room_type="classroom", building="W3", dept_id=99),
    ]
    placed = [
        item(course_id=1, section="61", room_id=7, timeslot_id=TS),
        item(course_id=2, section="05", room_id=8, timeslot_id=TS, capacity=200),
    ]
    dropped = [
        item(course_id=1, section="61", room_id=7, timeslot_id=TS),
        item(course_id=2, section="05", room_id=None, timeslot_id=None),
    ]

    assert calculate_fitness(dropped, rooms, total_sections=2) < \
        calculate_fitness(placed, rooms, total_sections=2)


def test_a_clean_schedule_scores_near_one():
    rooms = [room(room_id=i, capacity=50, room_type="classroom",
                  building="M8", dept_id=1) for i in range(1, 4)]
    schedule = [item(course_id=i, section="61", room_id=i, timeslot_id=TS)
                for i in range(1, 4)]
    assert calculate_fitness(schedule, rooms, total_sections=3) == pytest.approx(1.0)


def test_empty_schedule_does_not_divide_by_zero():
    assert calculate_fitness([], [], total_sections=0) == 0.0
