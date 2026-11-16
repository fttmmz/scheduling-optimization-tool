"""Evaluate instructor preferences for a generated timetable. Calculate penalty totals, satisfaction statistics, and results by preference category without changing the schedule."""
import statistics
from .preference_config import get_preference_calculator

def preference_checks(schedule, timeslots, preferences):
    """Evaluate one binary check per configured preference category per instructor.

    Days: every scheduled meeting day must satisfy preferred/avoided days.
    Times: every scheduled meeting must be in a preferred time band.
    Gaps: no positive gaps between consecutive same-day meetings.
    Fewer days: teaching on at most two distinct days (the provisional scorer's
        zero-penalty target, NOT a user-specified number of days).
    Max classes: no day exceeds the saved maximum.

    Instructors without scheduled teaching events are excluded from the rate;
    their preferences cannot be assessed from a timetable.
    """
    from collections import defaultdict
    from .preference_penalty import (
        _event, _teaching_instructors,
    )
    slot_map = {ts.id: ts for ts in timeslots}
    events = defaultdict(list)
    for item in schedule:
        event = _event(slot_map.get(item.timeslot_id))
        if event is not None:
            for inst in _teaching_instructors(item):
                if inst in preferences:
                    events[inst].append(event)

    results = {}
    for inst, pref in preferences.items():
        meetings = events.get(str(inst), [])
        if not meetings:
            continue
        daily = defaultdict(list)
        for days, start, end in meetings:
            for day in days:
                daily[day].append((start, end))
        checks = {}
        if pref.get("days_priority"):
            preferred = set(pref.get("preferred_days") or [])
            avoided = set(pref.get("avoid_days") or [])
            checks["days"] = all(
                not (set(days) & avoided) and
                (not preferred or set(days).issubset(preferred))
                for days, _, _ in meetings
            )
        if pref.get("time_priority"):
            allowed = set(pref.get("preferred_times") or [])
            checks["times"] = all(
                ("morning" if start < 720 else "afternoon" if start < 1020 else "evening") in allowed
                for _, start, _ in meetings
            )
        if pref.get("gaps_priority"):
            no_gaps = True
            for slots in daily.values():
                previous_end = None
                for start, end in sorted(slots):
                    if previous_end is not None and start > previous_end:
                        no_gaps = False
                    previous_end = max(previous_end or end, end)
            checks["gaps"] = no_gaps
        if pref.get("fewer_days_priority"):
            checks["fewer_days"] = len(daily) <= 2
        if pref.get("max_classes_priority") and pref.get("max_classes_per_day") is not None:
            maximum = pref["max_classes_per_day"]
            checks["max_classes"] = all(len(slots) <= maximum for slots in daily.values())
        if checks:
            results[str(inst)] = checks
    return results

def gini(values):
    """Gini inequality: 0 is equal distribution, 1 approaches maximum inequality."""
    values = sorted(max(0.0, float(v)) for v in values)
    if not values or sum(values) == 0:
        return 0.0 if not values else float("nan")
    n = len(values)
    return sum((2 * i - n - 1) * v for i, v in enumerate(values, 1)) / (n * sum(values))

def satisfaction_summary(checks):
    counts = [len(v) for v in checks.values()]
    met = [sum(v.values()) for v in checks.values()]
    total = sum(counts)
    satisfied = sum(met)
    percentages = [100 * a / b for a, b in zip(met, counts)]
    return {
        "min_count": min(met) if met else 0,
        "max_count": max(met) if met else 0,
        "avg_count": statistics.mean(met) if met else 0.0,
        "gini": gini(percentages),
        "instructors": len(checks), "total": total,
        "satisfied": satisfied, "unsatisfied": total - satisfied,
        "rate": 100 * satisfied / total if total else float("nan"),
        "min": min(percentages) if percentages else float("nan"),
        "max": max(percentages) if percentages else float("nan"),
        "mean": statistics.mean(percentages) if percentages else float("nan"),
        "std": statistics.pstdev(percentages) if percentages else float("nan"),
    }

def preference_violation_counts(schedule, timeslots, preferences):
    """Count satisfied and violated instructor preferences by category.

    Preferred and avoided days share the `days` check.
    Instructors without scheduled teaching are not included.
    """
    checks = preference_checks(schedule, timeslots, {str(k): v for k, v in (preferences or {}).items()})
    counts = {}
    for category in ("days", "times", "gaps", "fewer_days", "max_classes"):
        values = [result[category] for result in checks.values() if category in result]
        counts[category] = {
            "satisfied": sum(values),
            "violated": len(values) - sum(values),
            "total": len(values),
        }
    return counts


def evaluate_preference_config(schedule, timeslots, preferences, calculator=None):
    """Calculate instructor preference metrics for an existing timetable.

    Calculates and returns preference metrics without modifying the timetable.
    """
    preferences = {str(k): v for k, v in (preferences or {}).items()}
    calculator = calculator or get_preference_calculator()
    checks = preference_checks(schedule, timeslots, preferences)
    summary = satisfaction_summary(checks)
    return {
        "preference_penalty": float(calculator(schedule, timeslots, preferences)),
        "instructors_evaluated": summary["instructors"],
        "preferences_evaluated": summary["total"],
        "preferences_satisfied": summary["satisfied"],
        "preferences_unsatisfied": summary["unsatisfied"],
        "preference_satisfaction_rate": summary["rate"],
        "mean_instructor_satisfaction": summary["mean"],
        "satisfaction_gini": summary["gini"],
        "categories": preference_violation_counts(schedule, timeslots, preferences),
    }
