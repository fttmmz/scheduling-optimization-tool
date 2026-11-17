"""Fairness across instructors for one timetable: the numbers the admin pages show.

Wraps the two fairness measures in Optimization/ without changing them, keeping only what the
pages need and making the result safe to send as JSON:

  * workload: schedule_fairness.evaluate_schedule_fairness -- how evenly teaching sections,
    weekly meetings and weekly hours are spread, as a Gini and a 0-100 score (100 = equal);
  * preferences: InstructorPreferences.preference_evaluation.evaluate_preference_config -- how
    evenly preference satisfaction is spread (its satisfaction Gini). Its checks are pass/fail
    and stricter than preference_report.py's (any gap between classes fails "minimize gaps").

Who counts for workload: every instructor teaching any item of the schedule, scheduled or not,
so a section left without a time shows as lost teaching instead of quietly dropping out.
Instructors who teach nothing this term are left out: they are the same for every schedule and
would only add the same zeros to each.
"""
import math
from typing import Dict, Mapping, Optional

from backend.Optimization.InstructorPreferences.preference_evaluation import evaluate_preference_config
from backend.Optimization.InstructorPreferences.preference_penalty import _teaching_instructors
from backend.Optimization.schedule_fairness import evaluate_schedule_fairness

HEAVIEST_LISTED = 5


def _number(value, digits: int = 4) -> Optional[float]:
    """A finite float rounded for display, or None: JSON has no NaN or infinity."""
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _metric(summary: Mapping) -> Dict:
    return {key: _number(summary.get(key)) for key in
            ("min", "max", "average", "standard_deviation", "gini", "fairness_score")}


def workload_fairness(schedule_items, timeslots, names: Optional[Mapping[str, str]] = None) -> Dict:
    names = names or {}
    teaching = sorted({str(i) for item in schedule_items for i in _teaching_instructors(item)})
    report = evaluate_schedule_fairness(schedule_items, timeslots, instructors=teaching)
    loads = report["instructor_workloads"]
    heaviest = sorted(loads.items(), key=lambda kv: (-kv[1]["hours"], kv[0]))[:HEAVIEST_LISTED]
    return {
        "instructors": report["number_of_instructors"],
        "without_scheduled_teaching": sum(1 for load in loads.values() if load["sections"] == 0),
        "sections": _metric(report["sections"]),
        "meetings": _metric(report["meetings"]),
        "hours": _metric(report["hours"]),
        "heaviest": [{"inst_id": inst_id, "name": names.get(inst_id), **load} for inst_id, load in heaviest],
    }


def preference_fairness(schedule_items, timeslots, preferences_by_instructor: Mapping[str, dict]) -> Optional[Dict]:
    """None when no instructor has saved preferences: there is nothing to spread."""
    if not preferences_by_instructor:
        return None
    result = evaluate_preference_config(schedule_items, timeslots, preferences_by_instructor)
    gini = _number(result["satisfaction_gini"])
    return {
        "instructors": result["instructors_evaluated"],
        "checks": result["preferences_evaluated"],
        "checks_met": result["preferences_satisfied"],
        "satisfaction_rate": _number(result["preference_satisfaction_rate"], 2),
        "mean_instructor_satisfaction": _number(result["mean_instructor_satisfaction"], 2),
        "gini": gini,
        # Same reading as the workload score: 100 means everyone fares alike.
        "fairness_score": None if gini is None else _number(max(0.0, (1.0 - gini) * 100.0), 2),
    }


def build_fairness_report(schedule_items, timeslots, preferences_by_instructor: Optional[Mapping[str, dict]],
                          names: Optional[Mapping[str, str]] = None) -> Dict:
    """Both measures. `preferences` is None when saved preferences couldn't be read or none exist."""
    return {
        "workload": workload_fairness(schedule_items, timeslots, names),
        "preferences": preference_fairness(schedule_items, timeslots, preferences_by_instructor or {}),
    }
