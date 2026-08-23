"""Are the 777 'exact duplicates' really redundant, or are they distinct meetings
that lost their distinguishing attribute? Hypothesis from the reg site: clinical
practice courses have several meeting blocks per section.
"""
from collections import Counter, defaultdict

from backend.database.db import supabase

PAGE = 1000


def fetch_details(schedule_id=1):
    rows, offset = [], 0
    while True:
        res = (supabase.table("schedule_detailes").select("*, courses(*)")
               .eq("schedule_id", schedule_id)
               .range(offset, offset + PAGE - 1).execute())
        rows.extend(res.data)
        if len(res.data) < PAGE:
            return rows
        offset += PAGE


def fetch_all(table):
    rows, offset = [], 0
    while True:
        res = supabase.table(table).select("*").range(offset, offset + PAGE - 1).execute()
        rows.extend(res.data)
        if len(res.data) < PAGE:
            return rows
        offset += PAGE


rows = fetch_details(1)
timeslots = {t["timeslot_id"]: t for t in fetch_all("timeslot")}


def full_key(r):
    return (r["course_id"], r["section"], r["room_id"],
            r["timeslot_id"], r["instructor_id"], r["sec_capacity"])


print("--- (1) duplicate multiplicity: how many copies do repeated rows come in? ---")
full = Counter(map(full_key, rows))
mult = Counter(full.values())
for k in sorted(mult):
    print(f"    {k} copies: {mult[k]:>5} distinct rows  ({(k-1)*mult[k]:>4} redundant)")

print("\n--- (2) which course types produce the exact duplicates? ---")
dup_by_type = Counter()
all_by_type = Counter()
ctype = {}
for r in rows:
    t = (r["courses"] or {}).get("course_type")
    ctype[r["course_id"]] = t
    all_by_type[t] += 1
for k, n in full.items():
    if n > 1:
        dup_by_type[ctype.get(k[0])] += n - 1
print(f"    {'course_type':<28} {'redundant':>9} {'total rows':>11}")
for t, n in dup_by_type.most_common():
    print(f"    {str(t):<28} {n:>9} {all_by_type[t]:>11}")

print("\n--- (3) course types that never duplicate ---")
clean = [t for t in all_by_type if t not in dup_by_type]
print("   ", clean)

print("\n--- (4) does multiplicity match capacity? (rotation-group theory) ---")
sample = [(k, n) for k, n in full.items() if n > 1][:12]
for k, n in sample:
    ts = timeslots.get(k[3])
    day = ts["day"] if ts else None
    print(f"    course={k[0]} sec={k[1]} type={str(ctype.get(k[0]))[:22]:<22} "
          f"copies={n} cap={k[5]} day={day} room={k[2]}")

print("\n--- (5) day encoding: are multi-day patterns one timeslot row? ---")
daypat = Counter(t["day"] for t in timeslots.values())
print("    distinct day strings:", len(daypat))
print("    sample:", [d for d, _ in daypat.most_common(12)])
multi_day = [d for d in daypat if d and len(str(d)) > 1]
print(f"    multi-character day strings: {len(multi_day)} e.g. {multi_day[:10]}")

print("\n--- (6) clinical practice specifically ---")
for t in all_by_type:
    if t and "clinic" in str(t).lower():
        sub = [r for r in rows if ctype.get(r["course_id"]) == t]
        secs = {(r["course_id"], r["section"]) for r in sub}
        pat = defaultdict(set)
        for r in sub:
            pat[(r["course_id"], r["section"])].add((r["room_id"], r["timeslot_id"]))
        print(f"    type={t!r}: {len(sub)} rows, {len(secs)} distinct sections, "
              f"{sum(1 for v in pat.values() if len(v) > 1)} sections with >1 pattern")
        print(f"      rows per section: {Counter(len(v) for v in pat.values())}")
        null_room = sum(1 for r in sub if r["room_id"] is None)
        print(f"      rows with NULL room: {null_room}/{len(sub)}")
