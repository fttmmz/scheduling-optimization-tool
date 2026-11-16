"""How well a timetable met each instructor's saved preferences: the report the admin reads.

The report judges a class the way the optimizer's penalty does
(Optimization/InstructorPreferences/preference_penalty.py), so the page and the search agree on what a miss is:

  * a class's time band comes from its start: before 12:00 morning, before 17:00 afternoon, else evening;
  * days are M T W R (R = Thursday); an avoided day is a miss, and so is a day outside a preferred list;
  * "fewer teaching days" is met at FEWER_DAYS_TARGET days or fewer, where the penalty starts counting;
  * "max classes per day" is checked day by day;
  * "minimize gaps" counts a day as gappy when two of its classes are more than GAP_TOLERANCE_MIN apart.
    The penalty charges every idle minute; the tolerance only stops the report calling the usual
    15-minute change-over a gap.

`cost` is the optimizer's own penalty for that instructor (before the run's on/off weight), so the
list can be sorted by who the timetable treats worst. Preferences are read as they are now, which is
not necessarily what they were when the schedule was made.
"""
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional

from backend.Optimization.InstructorPreferences.preference_penalty import InstructorPreferenceTracker, _cost

FEWER_DAYS_TARGET = 2
GAP_TOLERANCE_MIN = 30
MAX_MISSES_LISTED = 6

DAY_NAMES = {"M": "Mon", "T": "Tue", "W": "Wed", "R": "Thu"}
DAY_ORDER = "MTWR"
CHECK_ORDER = ("days", "time", "fewer_days", "max_classes", "gaps")


def _band(start_minute: int) -> str:
    if start_minute < 12 * 60:
        return "morning"
    return "afternoon" if start_minute < 17 * 60 else "evening"


def _clock(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _duration(minutes: int) -> str:
    hours, rest = divmod(int(round(minutes)), 60)
    if hours and rest:
        return f"{hours} h {rest} min"
    return f"{hours} h" if hours else f"{rest} min"


def _day_list(days: Iterable[str]) -> str:
    return ", ".join(DAY_NAMES[d] for d in DAY_ORDER if d in set(days))


def _check(key, priority, met, total, summary, misses=()):
    status = "met" if met >= total else ("missed" if met == 0 else "partly")
    misses = list(misses)
    return {
        "key": key,
        "priority": priority,
        "met": met,
        "total": total,
        "status": status,
        "summary": summary,
        "misses": misses[:MAX_MISSES_LISTED],
        "more_misses": max(0, len(misses) - MAX_MISSES_LISTED),
    }


def _checks(events, pref) -> List[dict]:
    """One entry per preference the instructor set, judged against their meetings."""
    daily = defaultdict(list)
    for days, start, end in events:
        for day in days:
            daily[day].append((start, end))
    checks = []

    if pref.get("days_priority"):
        preferred = set(pref.get("preferred_days") or ())
        avoided = set(pref.get("avoid_days") or ())
        met = total = 0
        bad_days = {}
        for days, _, _ in events:
            for day in days:
                total += 1
                if day in avoided:
                    bad_days[day] = "avoided"
                elif preferred and day not in preferred:
                    bad_days.setdefault(day, "not preferred")
                else:
                    met += 1
        wanted = []
        if preferred:
            wanted.append(f"prefers {_day_list(preferred)}")
        if avoided:
            wanted.append(f"avoids {_day_list(avoided)}")
        misses = [f"Teaches {DAY_NAMES[d]} ({bad_days[d]})" for d in DAY_ORDER if d in bad_days]
        checks.append(_check("days", pref["days_priority"], met, total,
                             f"{met} of {total} class days fit ({'; '.join(wanted)})", misses))

    if pref.get("time_priority"):
        wanted = set(pref.get("preferred_times") or ())
        met = 0
        misses = []
        for days, start, end in sorted(events, key=lambda e: (DAY_ORDER.find(e[0][0]), e[1])):
            band = _band(start)
            if band in wanted:
                met += 1
            else:
                misses.append(f"{''.join(days)} {_clock(start)}–{_clock(end)} ({band})")
        order = ("morning", "afternoon", "evening")
        wanted_text = " or ".join(b for b in order if b in wanted)
        checks.append(_check("time", pref["time_priority"], met, len(events),
                             f"{met} of {len(events)} classes in the {wanted_text}", misses))

    if pref.get("fewer_days_priority"):
        count = len(daily)
        ok = count <= FEWER_DAYS_TARGET
        summary = f"Teaches on {count} day{'s' if count != 1 else ''} ({_day_list(daily)})"
        misses = [] if ok else [f"{count - FEWER_DAYS_TARGET} more than {FEWER_DAYS_TARGET} days"]
        checks.append(_check("fewer_days", pref["fewer_days_priority"], int(ok), 1, summary, misses))

    if pref.get("max_classes_priority") and pref.get("max_classes_per_day") is not None:
        limit = pref["max_classes_per_day"]
        over = [(d, len(daily[d])) for d in DAY_ORDER if d in daily and len(daily[d]) > limit]
        met = len(daily) - len(over)
        checks.append(_check(
            "max_classes", pref["max_classes_priority"], met, len(daily),
            f"{met} of {len(daily)} days at {limit} class{'es' if limit != 1 else ''} or fewer",
            [f"{DAY_NAMES[d]}: {n} classes" for d, n in over],
        ))

    if pref.get("gaps_priority"):
        busy = [d for d in DAY_ORDER if len(daily.get(d, ())) > 1]
        misses = []
        idle_total = 0
        for day in busy:
            previous_end = None
            idle = longest = 0
            for start, end in sorted(daily[day]):
                if previous_end is not None and start > previous_end:
                    idle += start - previous_end
                    longest = max(longest, start - previous_end)
                previous_end = max(previous_end or end, end)
            idle_total += idle
            if longest > GAP_TOLERANCE_MIN:
                misses.append(f"{DAY_NAMES[day]}: {_duration(idle)} idle between classes")
        met = len(busy) - len(misses)
        summary = (f"{met} of {len(busy)} multi-class days without long gaps"
                   f" ({_duration(idle_total)} idle in total)") if busy else "No day has two classes"
        checks.append(_check("gaps", pref["gaps_priority"], met, len(busy), summary, misses))

    return checks


def _overall(checks) -> str:
    statuses = {c["status"] for c in checks}
    if statuses <= {"met"}:
        return "met"
    return "missed" if statuses == {"missed"} else "partly"


def build_preference_report(
    schedule_items,
    timeslots,
    preferences_by_instructor: Mapping[str, dict],
    names: Optional[Mapping[str, str]] = None,
) -> Dict:
    """Per-instructor results and the totals the admin page shows, for any list of ScheduleItems."""
    names = names or {}
    tracker = InstructorPreferenceTracker(timeslots, preferences_by_instructor)
    for item in schedule_items:
        tracker.add(item, tracker.timeslot_map.get(item.timeslot_id))

    rows = []
    for inst_id, pref in preferences_by_instructor.items():
        events = tracker.events.get(inst_id, [])
        if not events:
            rows.append({"inst_id": inst_id, "name": names.get(inst_id), "status": "no_classes",
                         "classes": 0, "teaching_days": 0, "cost": 0.0, "checks": []})
            continue
        checks = _checks(events, pref)
        rows.append({
            "inst_id": inst_id,
            "name": names.get(inst_id),
            "status": _overall(checks),
            "classes": len(events),
            "teaching_days": len({d for days, _, _ in events for d in days}),
            "cost": round(_cost(events, pref), 2),
            "checks": checks,
        })

    rank = {"missed": 0, "partly": 1, "met": 2, "no_classes": 3}
    rows.sort(key=lambda r: (rank[r["status"]], -r["cost"], str(r["name"] or r["inst_id"])))

    by_check = {key: {"met": 0, "total": 0, "instructors": 0, "fully_met": 0} for key in CHECK_ORDER}
    by_priority = {p: {"checks": 0, "fully_met": 0} for p in ("high", "medium", "low")}
    for row in rows:
        for check in row["checks"]:
            agg = by_check[check["key"]]
            agg["met"] += check["met"]
            agg["total"] += check["total"]
            agg["instructors"] += 1
            agg["fully_met"] += check["status"] == "met"
            prio = by_priority[check["priority"]]
            prio["checks"] += 1
            prio["fully_met"] += check["status"] == "met"

    teaching = [r for r in rows if r["status"] != "no_classes"]
    return {
        "summary": {
            "with_preferences": len(rows),
            "teaching": len(teaching),
            "no_classes": len(rows) - len(teaching),
            "met": sum(r["status"] == "met" for r in teaching),
            "partly": sum(r["status"] == "partly" for r in teaching),
            "missed": sum(r["status"] == "missed" for r in teaching),
            "total_cost": round(sum(r["cost"] for r in teaching), 2),
        },
        "by_check": by_check,
        "by_priority": by_priority,
        "instructors": rows,
        "rules": {"fewer_days_target": FEWER_DAYS_TARGET, "gap_tolerance_min": GAP_TOLERANCE_MIN},
    }


def load_instructor_names(client, page_size: int = 1000) -> Dict[str, str]:
    """inst_id (as a string, matching schedule items) -> name, read in pages past Supabase's 1000-row cap."""
    names, offset = {}, 0
    while True:
        rows = (
            client.table("instructor").select("inst_id, name").order("inst_id")
            .range(offset, offset + page_size - 1).execute().data or []
        )
        names.update({str(r["inst_id"]): r.get("name") for r in rows})
        if len(rows) < page_size:
            return names
        offset += page_size
