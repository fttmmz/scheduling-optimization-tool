"""Regenerate the offline test fixtures from Supabase.

The invariant tests in test_data_invariants.py assert facts about the real
dataset (see HANDOVER.md section 4). Hitting the database from every test run
would be slow, network-dependent, and would silently change what the tests
assert whenever the data changed -- which is exactly the failure we want to
catch, not absorb.

So the tests run against a committed snapshot, and this script refreshes it.
When the dataset legitimately changes: run this, inspect the diff, and update
the expected numbers in test_data_invariants.py deliberately.

    $env:PYTHONPATH='D:\\uniDB\\scheduling-optimization-tool\\src'
    .\\.venv311\\Scripts\\python.exe tests\\snapshot_refresh.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from backend.database.db import supabase  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
PAGE = 1000

# Only the columns the tests actually use -- keeps the snapshot small and makes
# an unrelated schema addition a non-event.
DETAIL_COLS = ("course_id", "section", "room_id", "timeslot_id",
               "instructor_id", "sec_capacity")
COURSE_COLS = ("course_id", "name", "course_type", "dept_id", "level", "course_class")
ROOM_COLS = ("room_id", "capacity", "room_type", "building", "dept_id", "room_num")
TIMESLOT_COLS = ("timeslot_id", "day", "start_time", "end_time")


def _page(table, select, **eq):
    rows, offset = [], 0
    while True:
        query = supabase.table(table).select(select)
        for key, value in eq.items():
            query = query.eq(key, value)
        res = query.range(offset, offset + PAGE - 1).execute()
        rows.extend(res.data)
        if len(res.data) < PAGE:
            return rows
        offset += PAGE


def _pick(row, cols):
    return {col: row.get(col) for col in cols}


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)

    details = _page("schedule_detailes", "*, courses(*)", schedule_id=1)
    snapshot = []
    for row in details:
        record = _pick(row, DETAIL_COLS)
        record["courses"] = _pick(row.get("courses") or {}, COURSE_COLS)
        snapshot.append(record)

    # Stable ordering so the committed file diffs cleanly between refreshes.
    snapshot.sort(key=lambda r: tuple(
        (r[c] is None, str(r[c])) for c in DETAIL_COLS
    ))

    payloads = {
        "schedule_1.json": snapshot,
        "rooms.json": [_pick(r, ROOM_COLS) for r in _page("room", "*")],
        "timeslots.json": [_pick(t, TIMESLOT_COLS) for t in _page("timeslot", "*")],
    }

    for name, payload in payloads.items():
        path = FIXTURES / name
        path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str),
                        encoding="utf-8")
        print(f"wrote {path.name:<20} {len(payload):>5} records  "
              f"{path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
