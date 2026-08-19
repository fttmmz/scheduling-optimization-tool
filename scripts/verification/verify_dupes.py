"""Section 4 verification: is schedule_detailes one row per class, or one row per instructor?

Counts, for each schedule_id of interest:
  total rows
  distinct FullKey   = every scheduling column INCLUDING instructor
  distinct ClassKey  = every scheduling column EXCLUDING instructor
  distinct SectionKey= (course_id, section) only -- the identity Section.id uses
"""
from collections import Counter, defaultdict

from backend.database.db import supabase

PAGE = 1000


def fetch(schedule_id):
    rows, offset = [], 0
    while True:
        res = (
            supabase.table("schedule_detailes")
            .select("*")
            .eq("schedule_id", schedule_id)
            .range(offset, offset + PAGE - 1)
            .execute()
        )
        rows.extend(res.data)
        if len(res.data) < PAGE:
            return rows
        offset += PAGE


def report(schedule_id):
    rows = fetch(schedule_id)
    if not rows:
        print(f"\n=== schedule_id={schedule_id}: NO ROWS ===")
        return

    print(f"\n=== schedule_id={schedule_id} ===")
    print("columns:", sorted(rows[0].keys()))

    def full_key(r):
        return (r["course_id"], r["section"], r["room_id"],
                r["timeslot_id"], r["instructor_id"], r["sec_capacity"])

    def class_key(r):
        return (r["course_id"], r["section"], r["room_id"],
                r["timeslot_id"], r["sec_capacity"])

    def section_key(r):
        return (r["course_id"], r["section"])

    total = len(rows)
    full = Counter(map(full_key, rows))
    cls = Counter(map(class_key, rows))
    sec = Counter(map(section_key, rows))

    print(f"total rows              {total:>6}")
    print(f"distinct FullKey        {len(full):>6}")
    print(f"distinct ClassKey       {len(cls):>6}")
    print(f"distinct SectionKey     {len(sec):>6}   <- (course_id, section)")
    print(f"true duplicate rows     {total - len(full):>6}   = total - FullKey")
    print(f"team-teaching rows      {len(full) - len(cls):>6}   = FullKey - ClassKey")

    # how many instructors per class, ignoring exact-duplicate rows
    instr_per_class = defaultdict(set)
    for r in rows:
        instr_per_class[class_key(r)].add(r["instructor_id"])
    multi = {k: v for k, v in instr_per_class.items() if len(v) > 1}
    print(f"classes w/ >1 instructor{len(multi):>6}")
    if multi:
        widest = sorted(multi.items(), key=lambda kv: -len(kv[1]))[:5]
        print("  widest:", [(k[0], k[1], len(v)) for k, v in widest])

    # meeting patterns per (course, section) -- distinct room/time combos
    patt = defaultdict(set)
    for r in rows:
        patt[section_key(r)].add((r["room_id"], r["timeslot_id"]))
    multi_patt = {k: v for k, v in patt.items() if len(v) > 1}
    print(f"sections w/ >1 room/time{len(multi_patt):>6}")

    # what does instructor_id actually look like?
    ids = [r["instructor_id"] for r in rows]
    print(f"distinct instructor_id  {len(set(ids)):>6}  sample={sorted(set(map(str, ids)))[:5]}")
    print(f"  null instructor_id    {sum(1 for i in ids if i is None):>6}")


for sid in (1, 10):
    report(sid)
