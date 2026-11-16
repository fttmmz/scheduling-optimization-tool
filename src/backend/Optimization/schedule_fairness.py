"""Workload fairness evaluation for a complete university timetable.

This module measures how teaching workload is distributed across instructors.
It is independent of instructor preferences.

Metrics:
- sections: number of scheduled sections assigned to each instructor
- meetings: number of weekly teaching meetings (multi-day slots count once per day)
- hours: total weekly teaching hours

The Gini coefficient is used as an inequality measure:
- 0.0 means an equal distribution
- values closer to 1.0 mean a more unequal distribution

Usage from the project:
    from backend.Optimization.schedule_fairness import evaluate_schedule_fairness

    report = evaluate_schedule_fairness(
        schedule=schedule,
        timeslots=data["timeslots"],
        instructors=data.get("instructors", []),  # optional, but recommended
    )

    print(report["sections"])
    print(report["instructor_workloads"])
"""

from collections import defaultdict
import statistics


# Use the same event and instructor extraction rules as the existing
# preference-penalty code, so this report counts teaching assignments
# consistently with the rest of the project.
from .InstructorPreferences.preference_penalty import _event, _teaching_instructors

def calculate_gini(values):
    """Return the Gini coefficient for a list of non-negative workloads.

    A result of 0 means everyone has the same workload.
    A result closer to 1 means the workload is more unequal.
    """
    numbers = sorted(max(0.0, float(value)) for value in values)

    if not numbers:
        return 0.0

    total = sum(numbers)
    if total == 0:
        return 0.0

    count = len(numbers)
    weighted_sum = sum(
        (2 * index - count - 1) * value
        for index, value in enumerate(numbers, start=1)
    )
    return weighted_sum / (count * total)


def _get_instructor_id(instructor):
    """Accept an instructor ID or a common instructor object/dictionary."""
    if isinstance(instructor, (str, int)):
        return str(instructor)

    if isinstance(instructor, dict):
        instructor_id = instructor.get("id")
        if instructor_id is None:
            instructor_id = instructor.get("instructor_id")
        if instructor_id is not None:
            return str(instructor_id)

    instructor_id = getattr(instructor, "id", None)
    if instructor_id is None:
        instructor_id = getattr(instructor, "instructor_id", None)
    if instructor_id is not None:
        return str(instructor_id)

    raise ValueError(
        "Each instructor must be an ID or have an 'id'/'instructor_id' field."
    )


def _summarize_metric(workloads):
    """Summarize one workload metric across all included instructors."""
    values = list(workloads.values())

    if not values:
        return {
            "min": 0,
            "max": 0,
            "average": 0.0,
            "standard_deviation": 0.0,
            "gini": 0.0,
            "fairness_score": 100.0,
            "lowest_instructors": [],
            "highest_instructors": [],
        }

    minimum = min(values)
    maximum = max(values)
    inequality = calculate_gini(values)

    return {
        "min": minimum,
        "max": maximum,
        "average": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "gini": inequality,
        # A simple, easy-to-read score: 100 means equal; lower means less equal.
        "fairness_score": max(0.0, (1.0 - inequality) * 100.0),
        "lowest_instructors": [
            instructor_id
            for instructor_id, value in workloads.items()
            if value == minimum
        ],
        "highest_instructors": [
            instructor_id
            for instructor_id, value in workloads.items()
            if value == maximum
        ],
    }


def evaluate_schedule_fairness(schedule, timeslots, instructors=None):
    """Evaluate workload fairness for the whole generated schedule.

    Args:
        schedule:
            The generated schedule returned by the SchedulingEngine.
        timeslots:
            The project's timeslot objects. Each scheduled item is expected
            to have a timeslot_id, as in the existing benchmark script.
        instructors:
            Optional iterable containing all instructors (IDs, dictionaries,
            or objects). Pass the complete instructor list to include people
            who received zero classes. If omitted, only instructors found in
            the schedule can be counted.

    Returns:
        A dictionary containing per-instructor workload and overall statistics
        for sections, meetings, and hours.

    Notes:
        - Each scheduled item counts as one section for each instructor
          assigned to teach it.
        - A timeslot covering multiple days counts as one meeting per day.
        - Teaching hours are multiplied by the number of days in the timeslot.
        - This function measures workload distribution only. It does not
          evaluate instructor preferences or decide whether an allocation is
          fair based on contract hours, course difficulty, or other policies.
    """
    timeslot_by_id = {timeslot.id: timeslot for timeslot in timeslots}

    # Start with zero workload for every instructor, including instructors
    # who have no assigned classes (when the full instructor list is supplied).
    workloads = defaultdict(
        lambda: {"sections": 0, "meetings": 0, "hours": 0.0}
    )

    if instructors is not None:
        for instructor in instructors:
            instructor_id = _get_instructor_id(instructor)
            workloads[instructor_id]  # initialize zero values

    for scheduled_item in schedule:
        timeslot = timeslot_by_id.get(scheduled_item.timeslot_id)
        event = _event(timeslot)

        # Ignore items without a valid teaching timeslot.
        if event is None:
            continue

        days, start, end = event
        duration_hours = max(0.0, (end - start) / 60.0)

        # Avoid counting the same instructor twice for one scheduled item.
        assigned_instructors = {
            str(instructor_id)
            for instructor_id in _teaching_instructors(scheduled_item)
        }

        for instructor_id in assigned_instructors:
            workloads[instructor_id]["sections"] += 1
            workloads[instructor_id]["meetings"] += len(days)
            workloads[instructor_id]["hours"] += len(days) * duration_hours

    # Convert defaultdict to a regular dictionary for a clean report.
    instructor_workloads = {
        instructor_id: {
            "sections": values["sections"],
            "meetings": values["meetings"],
            "hours": round(values["hours"], 2),
        }
        for instructor_id, values in sorted(workloads.items())
    }

    sections = {
        instructor_id: values["sections"]
        for instructor_id, values in instructor_workloads.items()
    }
    meetings = {
        instructor_id: values["meetings"]
        for instructor_id, values in instructor_workloads.items()
    }
    hours = {
        instructor_id: values["hours"]
        for instructor_id, values in instructor_workloads.items()
    }

    return {
        "number_of_instructors": len(instructor_workloads),
        "instructor_workloads": instructor_workloads,
        "sections": _summarize_metric(sections),
        "meetings": _summarize_metric(meetings),
        "hours": _summarize_metric(hours),
    }
