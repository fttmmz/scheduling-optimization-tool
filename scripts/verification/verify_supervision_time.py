"""Check: do Senior Project Supervision rows sit later in the day than lectures?"""
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


def start_hour(tid):
    ts = timeslots.get(tid)
    if not ts or not ts.get("day") or ts.get("start_time") == ts.get("end_time"):
        return None
    return int(str(ts["start_time"])[:2])


by_type = defaultdict(lambda: {"rows": 0, "no_room": 0, "no_time": 0, "hours": []})
for r in rows:
    t = (r["courses"] or {}).get("course_type")
    b = by_type[t]
    b["rows"] += 1
    if r["room_id"] is None:
        b["no_room"] += 1
    h = start_hour(r["timeslot_id"])
    if h is None:
        b["no_time"] += 1
    else:
        b["hours"].append(h)

print(f"{'course_type':<32} {'rows':>5} {'noRoom':>7} {'noTime':>7} {'medStart':>9} {'>=17h':>7}")
for t, b in sorted(by_type.items(), key=lambda kv: -kv[1]["rows"]):
    hs = sorted(b["hours"])
    med = hs[len(hs) // 2] if hs else None
    late = sum(1 for h in hs if h >= 17)
    latepct = f"{100*late/len(hs):.0f}%" if hs else "-"
    print(f"{str(t):<32} {b['rows']:>5} {b['no_room']:>7} {b['no_time']:>7} "
          f"{str(med):>9} {latepct:>7}")

print("\n--- Senior Project Supervision: start-hour histogram ---")
sp = by_type.get("Senior Project Supervision", {}).get("hours", [])
for h, n in sorted(Counter(sp).items()):
    print(f"    {h:02d}:00  {'#' * n} ({n})")

print("\n--- Lecture Undergraduate: start-hour histogram (for contrast) ---")
lec = by_type.get("Lecture Undergraduate", {}).get("hours", [])
for h, n in sorted(Counter(lec).items()):
    print(f"    {h:02d}:00  {'#' * min(n, 60)} ({n})")
