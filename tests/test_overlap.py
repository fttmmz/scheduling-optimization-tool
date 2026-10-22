"""Real-time clashes: two DIFFERENT slots that overlap are a clash.

Occupancy used to be keyed by slot ID alone, so a room at `M 09:30-10:45` was "free" for
`MW 09:30-10:45`. On the real snapshot every algorithm produced hundreds of such clashes while the
evaluator reported 0 (PAPER_RESULTS_DRAFT.md section 00). These tests pin the rule, the evaluator's counts,
and every algorithm on a dataset where the slot-ID check WOULD double-book.
"""
import pytest

from backend.models.models import Room, ScheduleItem, Section, Timeslot
from backend.Optimization.constraints import (
    OverlapCounter, OverlapSet, OwnerMap, MULTIPLE_OWNERS, overlap_minutes, slots_clash,
)
from backend.Optimization.evaluation import (
    count_hard_conflicts, count_instructor_conflicts, count_room_conflicts, count_sibling_conflicts,
)
from test_algorithms import ALGORITHMS, _run


def slot(tid, day, start, end):
    return Timeslot({"timeslot_id": tid, "day": day, "start_time": start, "end_time": end})


# Ids far from the other test modules' fixtures, so the shared slot registry never mixes them.
MW_0930 = slot(9001, "MW", "09:30:00", "10:45:00")
M_0930 = slot(9002, "M", "09:30:00", "10:45:00")
MW_0935 = slot(9003, "MW", "09:35:00", "10:50:00")
TR_0930 = slot(9004, "TR", "09:30:00", "10:45:00")
MW_1045 = slot(9005, "MW", "10:45:00", "12:00:00")      # touches MW_0930's end
TR_1400 = slot(9006, "TR", "14:00:00", "15:15:00")


def test_the_clash_rule():
    assert slots_clash(MW_0930.id, MW_0930.id)
    assert slots_clash(MW_0930.id, M_0930.id)            # shares Monday 09:30-10:45
    assert slots_clash(MW_0930.id, MW_0935.id)            # near-duplicate: 70 shared minutes
    assert overlap_minutes(MW_0930.id, MW_0935.id) == 70
    assert not slots_clash(MW_0930.id, TR_0930.id)        # no common day
    assert not slots_clash(MW_0930.id, MW_1045.id)        # touching ends do not clash
    assert not slots_clash(MW_0930.id, 987654)            # unknown slot: only itself
    assert slots_clash(987654, 987654)


def test_occupancy_containers():
    busy = OverlapSet()
    busy.add(("room", MW_0930.id))
    assert ("room", MW_0930.id) in busy
    assert ("room", M_0930.id) in busy
    assert ("room", TR_0930.id) not in busy
    assert ("other room", M_0930.id) not in busy
    assert ("room", M_0930.id) not in busy - {("room", MW_0930.id)}
    busy.discard(("room", MW_0930.id))
    assert ("room", M_0930.id) not in busy

    counts = OverlapCounter()
    counts[("room", MW_0930.id)] += 1
    counts[("room", MW_0935.id)] += 1
    assert counts[("room", M_0930.id)] == 0               # indexing stays exact, like a Counter
    assert counts.overlapping(("room", M_0930.id)) == 2

    owners = OwnerMap()
    owners[("room", MW_0930.id)] = 7
    assert owners.get(("room", M_0930.id)) == 7
    owners[("room", MW_0935.id)] = 8
    assert owners.get(("room", M_0930.id)) is MULTIPLE_OWNERS
    owners.pop(("room", MW_0935.id))
    assert owners.get(("room", M_0930.id)) == 7


def _section(course_id, section_no="61", instructors=("1",), course_type="Lecture Undergraduate"):
    row = {
        "courses": {"course_id": course_id, "name": f"Course {course_id}", "course_type": course_type,
                    "dept_id": 1, "level": "Undergraduate", "course_class": None},
        "instructor_id": instructors[0] if instructors else None, "section": section_no, "sec_capacity": 20,
    }
    return Section(row, instructor_ids=list(instructors))


def test_evaluator_counts_overlapping_slots():
    a, b = _section(101, instructors=("1",)), _section(102, instructors=("2",))
    room_clash = [ScheduleItem.from_section(a, room_id=1, timeslot_id=MW_0930.id),
                  ScheduleItem.from_section(b, room_id=1, timeslot_id=M_0930.id)]
    assert count_room_conflicts(room_clash) == 1
    assert count_hard_conflicts(room_clash) == 1

    c, d = _section(103, instructors=("5",)), _section(104, instructors=("5",))
    teacher_clash = [ScheduleItem.from_section(c, room_id=1, timeslot_id=MW_0930.id),
                     ScheduleItem.from_section(d, room_id=2, timeslot_id=MW_0935.id)]
    assert count_instructor_conflicts(teacher_clash) == 1

    apart = [ScheduleItem.from_section(a, room_id=1, timeslot_id=MW_0930.id),
             ScheduleItem.from_section(b, room_id=1, timeslot_id=MW_1045.id)]
    assert count_hard_conflicts(apart) == 0

    blocks = []
    for pattern, ts in ((0, MW_0930), (1, M_0930)):
        row = {"courses": {"course_id": 105, "name": "Clinical", "course_type": "Clinical Practice",
                           "dept_id": 1, "level": "Master", "course_class": None},
               "instructor_id": None, "section": "71", "sec_capacity": 10}
        s = Section(row, instructor_ids=[], pattern_index=pattern, pattern_count=2)
        blocks.append(ScheduleItem.from_section(s, timeslot_id=ts.id))
    assert count_sibling_conflicts(blocks) == 1


def overlap_dataset():
    """One classroom; lecture slots that overlap pairwise on Mondays/Wednesdays. Judged by slot ID,
    MW_0930 and MW_0935 are two free hours; in real time they are one. Two lecture pairs share a teacher."""
    sections = [
        _section(201, instructors=("11",)),
        _section(202, instructors=("12",)),
        _section(203, instructors=("13",)),
        _section(204, instructors=("13",)),       # same teacher as 203
    ]
    timeslots = [MW_0930, MW_0935, TR_0930, TR_1400, MW_1045]
    rooms = [Room({"room_id": 1, "capacity": 40, "room_type": "classroom", "building": "M8",
                   "dept_id": 1, "room_num": "1"}),
             Room({"room_id": 2, "capacity": 40, "room_type": "classroom", "building": "M8",
                   "dept_id": 1, "room_num": "2"})]
    return sections, timeslots, rooms


@pytest.mark.parametrize("name", ALGORITHMS)
def test_no_algorithm_double_books_overlapping_slots(name):
    sections, timeslots, rooms = overlap_dataset()
    schedule = _run(name, sections, timeslots, rooms)
    assert count_room_conflicts(schedule) == 0, name
    assert count_instructor_conflicts(schedule) == 0, name
    assert count_hard_conflicts(schedule, rooms) == 0, name
