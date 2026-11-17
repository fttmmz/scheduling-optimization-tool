"""The fairness figures the admin pages show: who is counted, and that the result is JSON-safe."""
import json
import math
from datetime import time
from types import SimpleNamespace

from backend.fairness_report import build_fairness_report, preference_fairness, workload_fairness

SLOTS = [
    SimpleNamespace(id=1, day="MW", start=time(8, 0), end=time(9, 15)),
    SimpleNamespace(id=2, day="TR", start=time(14, 0), end=time(15, 15)),
    SimpleNamespace(id=3, day="M", start=time(18, 0), end=time(20, 30)),
]


def item(inst, slot, course_type="Lecture"):
    return SimpleNamespace(course_id="C1", course_type=course_type, instructor_id=inst,
                           instructor_ids=[inst], timeslot_id=slot)


def pref(**fields):
    base = dict(preferred_days=[], avoid_days=[], days_priority=None, preferred_times=[],
                time_priority=None, gaps_priority=None, fewer_days_priority=None,
                max_classes_per_day=None, max_classes_priority=None)
    return {**base, **fields}


def test_an_instructor_whose_sections_went_unscheduled_counts_as_zero_load():
    items = [item("1", 1), item("1", 2), item("2", None)]     # instructor 2's only section has no time
    w = workload_fairness(items, SLOTS, names={"1": "A"})
    assert w["instructors"] == 2
    assert w["without_scheduled_teaching"] == 1
    assert w["sections"]["min"] == 0 and w["sections"]["max"] == 2
    assert w["sections"]["gini"] == 0.5                         # one has everything, one nothing
    assert w["heaviest"][0] == {"inst_id": "1", "name": "A", "sections": 2, "meetings": 4, "hours": 5.0}


def test_supervision_is_not_teaching_load():
    items = [item("1", 1), item("2", 2, course_type="Senior Project Supervision")]
    assert workload_fairness(items, SLOTS)["instructors"] == 1


def test_equal_loads_score_100():
    w = workload_fairness([item("1", 1), item("2", 1)], SLOTS)
    assert w["hours"]["gini"] == 0 and w["hours"]["fairness_score"] == 100


def test_preference_fairness_reads_the_satisfaction_gini():
    prefs = {"1": pref(preferred_times=["morning"], time_priority="high"),
             "2": pref(preferred_times=["morning"], time_priority="high")}
    p = preference_fairness([item("1", 1), item("2", 3)], SLOTS, prefs)   # 1 met, 2 missed
    assert (p["instructors"], p["checks"], p["checks_met"]) == (2, 2, 1)
    assert p["satisfaction_rate"] == 50.0
    assert p["gini"] == 0.5 and p["fairness_score"] == 50.0


def test_nobody_satisfied_gives_no_score_rather_than_nan():
    prefs = {"1": pref(preferred_times=["evening"], time_priority="low")}
    p = preference_fairness([item("1", 1)], SLOTS, prefs)
    assert p["gini"] is None and p["fairness_score"] is None
    json.dumps(p, allow_nan=False)                               # what the API will send


def test_no_saved_preferences_means_no_preference_fairness():
    report = build_fairness_report([item("1", 1)], SLOTS, None)
    assert report["preferences"] is None
    assert report["workload"]["instructors"] == 1
    assert not any(isinstance(v, float) and math.isnan(v) for v in report["workload"]["hours"].values())
