"""Shared test fixtures.

Two rules for this suite:

  * Tests are OFFLINE. They read a LOCAL snapshot in fixtures/, never
    Supabase. It is real university data, so it is gitignored and never
    committed; create or refresh it with snapshot_refresh.py. Tests that need
    it skip when it is absent.
  * Nothing here imports backend.database.db -- that module opens a Supabase
    client at import time, so importing it would make the whole suite require
    network and credentials.
"""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load(name):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"missing fixture {name} -- run tests/snapshot_refresh.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def raw_rows():
    """schedule_detailes for schedule_id=1, one dict per row, courses nested."""
    return _load("schedule_1.json")


@pytest.fixture(scope="session")
def raw_rooms():
    return _load("rooms.json")


@pytest.fixture(scope="session")
def raw_timeslots():
    return _load("timeslots.json")


@pytest.fixture(scope="session")
def timeslots_by_id(raw_timeslots):
    return {t["timeslot_id"]: t for t in raw_timeslots}


# ── Keys used throughout the invariant tests (HANDOVER.md section 4) ──────────

def full_key(row):
    """Every scheduling column INCLUDING instructor. Distinct count = 3,921."""
    return (row["course_id"], row["section"], row["room_id"],
            row["timeslot_id"], row["instructor_id"], row["sec_capacity"])


def class_key(row):
    """Every scheduling column EXCLUDING instructor. Distinct count = 3,270.

    This is the granularity the section model should use: one entity per
    (course, section, meeting pattern). See HANDOVER.md section 4.5 -- keying on
    (course, section) alone would collapse real clinical meeting blocks.
    """
    return (row["course_id"], row["section"], row["room_id"],
            row["timeslot_id"], row["sec_capacity"])


def section_key(row):
    """(course_id, section) -- what Section.id currently returns. 3,254 distinct."""
    return (row["course_id"], row["section"])


def is_real_timeslot(ts):
    """Mirror of constraints.is_placeholder_timeslot, on a raw dict.

    Kept as a plain dict check so the invariant tests describe the DATA and do
    not silently follow a change in the placeholder rule -- test_classification
    covers that rule separately.
    """
    return bool(ts and ts.get("day") and ts.get("start_time") != ts.get("end_time"))
