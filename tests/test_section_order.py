"""Sections must come out in one order, however the database returns the rows.

Several algorithms depend on the order they are given, so this is what lets two devices get the same
result. The first tests need no fixtures, so they also run on a fresh clone.
"""
import json
import pathlib
import random

import pytest

from backend.models.models import (
    SCHEDULE_ROW_ORDER,
    build_sections,
    group_rows_by_pattern,
    schedule_row_sort_key,
    section_info_index,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _row(course, section, room=1, slot=1, instructor=1, capacity=30):
    return {"course_id": course, "section": section, "room_id": room, "timeslot_id": slot,
            "instructor_id": instructor, "sec_capacity": capacity}


def test_key_uses_the_snapshot_columns_in_order():
    assert SCHEDULE_ROW_ORDER == ("course_id", "section", "room_id", "timeslot_id", "instructor_id", "sec_capacity")


def test_key_sorts_as_text_to_match_the_snapshot():
    rows = [_row(9, "01"), _row(10, "01")]
    assert [r["course_id"] for r in sorted(rows, key=schedule_row_sort_key)] == [10, 9]


def test_key_accepts_empty_columns():
    rows = [_row(1, "01", room=None), _row(1, "01", room=5)]
    assert [r["room_id"] for r in sorted(rows, key=schedule_row_sort_key)] == [5, None]


def test_shuffled_rows_group_identically():
    rows = [_row(c, s, instructor=i) for c in (9, 10, 11) for s in ("01", "02") for i in (1, 2)]
    expected = group_rows_by_pattern(sorted(rows, key=schedule_row_sort_key))
    for seed in range(5):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        grouped = group_rows_by_pattern(sorted(shuffled, key=schedule_row_sort_key))
        assert list(grouped) == list(expected)
        assert grouped == expected


@pytest.mark.skipif(not (FIXTURES / "schedule_1.json").exists(), reason="snapshot fixtures not present")
def test_shuffled_snapshot_builds_the_same_sections():
    rows = json.loads((FIXTURES / "schedule_1.json").read_text(encoding="utf-8"))
    info = section_info_index(json.loads((FIXTURES / "section_info.json").read_text(encoding="utf-8")))

    def describe(sections):
        return [(s.id, s.pattern_index, tuple(s.instructor_ids or ())) for s in sections]

    expected = describe(build_sections(rows, info))
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    assert describe(build_sections(sorted(shuffled, key=schedule_row_sort_key), info)) == expected
