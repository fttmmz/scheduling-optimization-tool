"""Two meetings of one section in the same room at the same time must survive a save and reload.

Regression: saved schedules are read back by grouping rows on (course, section, room, timeslot,
capacity), so two identical meetings came back as ONE. A greedy schedule with one sibling clash
(both blocks of a studio section at W 11:00 in room 8) was saved with 1 hard conflict and read back
with 0, and the meeting was gone. Two unplaced blocks of one section merged the same way.
OFFLINE: Supabase is replaced by a fake that records what save_schedule() inserts.
"""
import os

os.environ.setdefault("NEXT_PUBLIC_SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "test-key")

import pytest  # noqa: E402

import backend.database.loader as loader  # noqa: E402
from backend.models.models import ScheduleItem, group_rows_by_pattern, meeting_blocks  # noqa: E402


class _Insert:
    def __init__(self, db, table, records):
        self.db, self.table, self.records = db, table, records

    def execute(self):
        if self.table == "schedule_detailes" and self.db.no_block_column and any("block" in r for r in self.records):
            raise RuntimeError("column schedule_detailes.block does not exist")
        if self.table == "schedule":
            return type("R", (), {"data": [{"schedule_id": 77, **self.records}]})()
        self.db.rows.extend(self.records)
        return type("R", (), {"data": list(self.records)})()


class _Table:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def insert(self, records):
        return _Insert(self.db, self.name, records)


class FakeDB:
    def __init__(self, no_block_column=False):
        self.rows, self.no_block_column = [], no_block_column

    def table(self, name):
        return _Table(self, name)


def item(section, room, slot, instructors=(5,), course=404421, capacity=18):
    return ScheduleItem(course_id=course, course_name="Studio", course_type="Studio", course_dept=1,
                        capacity=capacity, instructor_id=instructors[0] if instructors else None,
                        room_id=room, timeslot_id=slot, section=section, instructor_ids=instructors)


SCHEDULE = [
    item("61S", 8, 26),                  # the clash: two blocks of 61S, same room, same time
    item("61S", 8, 26),
    item("62S", 9, 26, instructors=(6, 7)),
    item("71", None, None, instructors=(), course=1002769, capacity=30),   # two unplaced blocks
    item("71", None, None, instructors=(), course=1002769, capacity=30),
]


def _save(monkeypatch, db):
    monkeypatch.setattr(loader, "supabase", db)
    return loader.save_schedule(SCHEDULE, {"sch_name": "t", "alg": "greedy", "user_id": 1})


def _meetings(rows):
    return sum(len(patterns) for patterns in group_rows_by_pattern(rows).values())


def test_meeting_blocks_number_only_the_repeats():
    assert meeting_blocks(SCHEDULE) == [0, 1, 0, 0, 1]


def test_identical_meetings_survive_a_save_and_reload(monkeypatch):
    db = FakeDB()
    result = _save(monkeypatch, db)
    assert result["blocks_saved"] is True
    assert _meetings(db.rows) == len(SCHEDULE)                       # 5 meetings, not 3
    grouped = group_rows_by_pattern(db.rows)
    assert len(grouped[(404421, "61S")]) == 2 and len(grouped[(1002769, "71")]) == 2
    # The team-taught section is still one meeting with both teachers on it.
    (rows_62,) = grouped[(404421, "62S")].values()
    assert sorted(r["instructor_id"] for r in rows_62) == [6, 7]


def test_registrar_rows_without_a_block_group_exactly_as_before():
    """The base schedule's exact repeats (one row per instructor, and true duplicates) still collapse."""
    rows = [
        {"course_id": 1, "section": "31", "room_id": 4, "timeslot_id": 9, "sec_capacity": 40, "instructor_id": 2},
        {"course_id": 1, "section": "31", "room_id": 4, "timeslot_id": 9, "sec_capacity": 40, "instructor_id": 2},
        {"course_id": 1, "section": "31", "room_id": 4, "timeslot_id": 9, "sec_capacity": 40, "instructor_id": 3},
        {"course_id": 1, "section": "31", "room_id": 4, "timeslot_id": 9, "sec_capacity": 40, "instructor_id": 3, "block": 0},
    ]
    assert _meetings(rows) == 1


def test_a_schedule_without_repeats_does_not_need_the_block_column(monkeypatch):
    db = FakeDB(no_block_column=True)
    monkeypatch.setattr(loader, "supabase", db)
    result = loader.save_schedule(SCHEDULE[2:3], {"sch_name": "t"})
    assert result["blocks_saved"] is True and all("block" not in r for r in db.rows)


def test_without_the_block_column_the_save_still_succeeds_and_says_so(monkeypatch):
    db = FakeDB(no_block_column=True)
    result = _save(monkeypatch, db)
    assert result["blocks_saved"] is False
    assert len(db.rows) == 6 and all("block" not in r for r in db.rows)   # every row saved
