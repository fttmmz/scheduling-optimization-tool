"""Decisive test: do exact duplicates land on SCHEDULABLE rows (real room + real time)
or on rows that need nothing scheduled? Deduping is only risky for the former."""
from collections import Counter, defaultdict

from backend.database.db import supabase

PAGE = 1000


def fetch(table, **eq):
    rows, offset = [], 0
    while True:
        q = supabase.table(table).select("*, courses(*)" if table == "schedule_detailes" else "*")
        for k, v in eq.items():
            q = q.eq(k, v)
        res = q.range(offset, offset + PAGE - 1).execute()
        rows.extend(res.data)
        if len(res.data) < PAGE:
            return rows
        offset += PAGE


rows = fetch("schedule_detailes", schedule_id=1)
timeslots = {t["timeslot_id"]: t for t in fetch("timeslot")}


def full_key(r):
    return (r["course_id"], r["section"], r["room_id"],
            r["timeslot_id"], r["instructor_id"], r["sec_capacity"])


def real_time(tid):
    ts = timeslots.get(tid)
    return bool(ts and ts.get("day") and ts.get("start_time") != ts.get("end_time"))


full = Counter(map(full_key, rows))

print("--- (A) redundant rows by what the row actually needs ---")
buckets = Counter()
for k, n in full.items():
    if n > 1:
        buckets[(k[2] is not None, real_time(k[3]))] += n - 1
print(f"    {'has room':<10} {'has real time':<14} {'redundant rows':>14}")
for (hr, ht), n in sorted(buckets.items(), key=lambda kv: -kv[1]):
    print(f"    {str(hr):<10} {str(ht):<14} {n:>14}")
print(f"    TOTAL {sum(buckets.values())}")

print("\n--- (B) the risky bucket: duplicates on fully-schedulable rows ---")
risky = [(k, n) for k, n in full.items() if n > 1 and k[2] is not None and real_time(k[3])]
print(f"    {len(risky)} distinct rows, {sum(n - 1 for _, n in risky)} redundant")
ctype = {r["course_id"]: (r["courses"] or {}).get("course_type") for r in rows}
for k, n in sorted(risky, key=lambda kv: -kv[1])[:15]:
    ts = timeslots.get(k[3])
    print(f"      course={k[0]} sec={k[1]} room={k[2]} day={ts['day'] if ts else None} "
          f"copies={n} type={ctype.get(k[0])}")

print("\n--- (C) the user's example, 1002702 sec 71 ---")
for r in sorted([r for r in rows if r["course_id"] == 1002702 and r["section"] == "71"],
                key=lambda r: (r["timeslot_id"] or 0)):
    ts = timeslots.get(r["timeslot_id"])
    print(f"      room={r['room_id']} ts={r['timeslot_id']} "
          f"day={ts['day'] if ts else None} {ts['start_time'] if ts else ''}-"
          f"{ts['end_time'] if ts else ''} instr={r['instructor_id']} cap={r['sec_capacity']}")

print("\n--- (D) the 22-instructor classes: what are they? ---")
for cid in (1501393, 1501494):
    sub = [r for r in rows if r["course_id"] == cid]
    if sub:
        c = sub[0]["courses"] or {}
        print(f"      course={cid} name={c.get('name')!r} type={c.get('course_type')!r} "
              f"dept={c.get('dept_id')} level={c.get('level')!r} rows={len(sub)}")

print("\n--- (E) do multi-pattern sections span DIFFERENT parts of the day? ---")
pat = defaultdict(set)
for r in rows:
    pat[(r["course_id"], r["section"])].add(r["timeslot_id"])
for k, v in sorted(pat.items(), key=lambda kv: -len(kv[1])):
    if len(v) > 1:
        desc = []
        for tid in sorted(v, key=lambda t: t or 0):
            ts = timeslots.get(tid)
            desc.append(f"{ts['day']} {ts['start_time']}-{ts['end_time']}" if ts else str(tid))
        print(f"      course={k[0]} sec={k[1]} type={str(ctype.get(k[0]))[:20]:<20} -> {desc}")
