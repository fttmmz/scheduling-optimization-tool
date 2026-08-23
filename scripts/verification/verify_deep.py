"""Follow-ups: (a) are schedule 1 and 10 the same data? (b) what do NULL instructors do
to the team-teaching count? (c) what are the 11 multi-pattern sections?"""
from collections import Counter, defaultdict

from backend.database.db import supabase

PAGE = 1000


def fetch(schedule_id):
    rows, offset = [], 0
    while True:
        res = (supabase.table("schedule_detailes").select("*")
               .eq("schedule_id", schedule_id)
               .range(offset, offset + PAGE - 1).execute())
        rows.extend(res.data)
        if len(res.data) < PAGE:
            return rows
        offset += PAGE


def full_key(r):
    return (r["course_id"], r["section"], r["room_id"],
            r["timeslot_id"], r["instructor_id"], r["sec_capacity"])


def class_key(r):
    return (r["course_id"], r["section"], r["room_id"],
            r["timeslot_id"], r["sec_capacity"])


s1, s10 = fetch(1), fetch(10)

print("--- (a) is schedule_id=1 the same data as schedule_id=10? ---")
c1, c10 = Counter(map(full_key, s1)), Counter(map(full_key, s10))
print("identical as multisets:", c1 == c10)
only1 = c1 - c10
only10 = c10 - c1
print("rows only in 1 :", sum(only1.values()))
print("rows only in 10:", sum(only10.values()))

print("\n--- (b) team teaching, NULL instructors excluded ---")
for name, rows in (("schedule_id=1", s1), ("schedule_id=10", s10)):
    instr = defaultdict(set)
    for r in rows:
        if r["instructor_id"] is not None:
            instr[class_key(r)].add(r["instructor_id"])
    multi = {k: v for k, v in instr.items() if len(v) > 1}
    print(f"{name}: classes with >1 REAL instructor = {len(multi)}")
    if multi:
        widest = sorted(multi.items(), key=lambda kv: -len(kv[1]))[:5]
        for k, v in widest:
            print(f"    course={k[0]} sec={k[1]} -> {len(v)} instructors")
    extra = sum(len(v) - 1 for v in multi.values())
    print(f"    extra real-instructor assignments = {extra}")

    # classes that mix NULL and non-NULL
    mixed = 0
    allinstr = defaultdict(set)
    for r in rows:
        allinstr[class_key(r)].add(r["instructor_id"])
    for k, v in allinstr.items():
        if None in v and len(v) > 1:
            mixed += 1
    print(f"    classes mixing NULL and real instructor = {mixed}")

print("\n--- (c) sections with >1 room/timeslot pattern ---")
patt = defaultdict(set)
for r in s1:
    patt[(r["course_id"], r["section"])].add((r["room_id"], r["timeslot_id"]))
for k, v in sorted(patt.items(), key=lambda kv: -len(kv[1]))[:15]:
    if len(v) > 1:
        print(f"    course={k[0]} sec={k[1]} -> {len(v)} patterns {sorted(v)}")

print("\n--- (d) how many DISTINCT sections carry a real instructor at all? ---")
for name, rows in (("schedule_id=1", s1), ("schedule_id=10", s10)):
    secs = defaultdict(set)
    for r in rows:
        secs[(r["course_id"], r["section"])].add(r["instructor_id"])
    with_real = sum(1 for v in secs.values() if v - {None})
    print(f"{name}: {with_real} of {len(secs)} distinct sections have >=1 real instructor")
