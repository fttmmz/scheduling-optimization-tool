"""The admin's preference report: what counts as met, and that it agrees with the optimizer's penalty."""
from datetime import time
from types import SimpleNamespace

from backend.Optimization.InstructorPreferences.preference_penalty import calculate_instructor_preference_penalty
from backend.preference_report import build_preference_report, load_instructor_names

SLOTS = [
    SimpleNamespace(id=1, day="MW", start=time(8, 0), end=time(9, 15)),
    SimpleNamespace(id=2, day="MW", start=time(9, 30), end=time(10, 45)),
    SimpleNamespace(id=3, day="TR", start=time(14, 0), end=time(15, 15)),
    SimpleNamespace(id=4, day="M", start=time(18, 0), end=time(20, 30)),
    SimpleNamespace(id=5, day="F", start=time(8, 0), end=time(9, 0)),
]


def item(inst, slot, course="C1"):
    return SimpleNamespace(course_id=course, course_type="Lecture", instructor_id=inst,
                           instructor_ids=[inst], timeslot_id=slot)


def pref(**fields):
    base = dict(preferred_days=[], avoid_days=[], days_priority=None, preferred_times=[],
                time_priority=None, gaps_priority=None, fewer_days_priority=None,
                max_classes_per_day=None, max_classes_priority=None)
    return {**base, **fields}


def checks_of(report, inst):
    row = next(r for r in report["instructors"] if r["inst_id"] == inst)
    return row, {c["key"]: c for c in row["checks"]}


def test_preferred_time_counts_each_class_by_its_start():
    prefs = {"1": pref(preferred_times=["morning"], time_priority="high")}
    report = build_preference_report([item("1", 1), item("1", 4)], SLOTS, prefs)
    row, checks = checks_of(report, "1")
    assert (checks["time"]["met"], checks["time"]["total"]) == (1, 2)
    assert checks["time"]["status"] == "partly"
    assert "evening" in checks["time"]["misses"][0]
    assert row["status"] == "partly"


def test_avoided_day_is_a_miss_and_friday_is_ignored():
    prefs = {"1": pref(avoid_days=["W"], days_priority="medium")}
    # Slot 1 meets M and W; slot 5 meets on Friday, which preferences don't cover.
    report = build_preference_report([item("1", 1), item("1", 5)], SLOTS, prefs)
    _, checks = checks_of(report, "1")
    assert (checks["days"]["met"], checks["days"]["total"]) == (1, 2)
    assert checks["days"]["misses"] == ["Teaches Wed (avoided)"]


def test_fewer_days_is_met_at_two_days():
    prefs = {"1": pref(fewer_days_priority="low")}
    two_days = build_preference_report([item("1", 1), item("1", 2)], SLOTS, prefs)
    four_days = build_preference_report([item("1", 1), item("1", 3)], SLOTS, prefs)
    assert checks_of(two_days, "1")[1]["fewer_days"]["status"] == "met"
    assert checks_of(four_days, "1")[1]["fewer_days"]["status"] == "missed"


def test_max_classes_is_checked_per_day():
    prefs = {"1": pref(max_classes_per_day=1, max_classes_priority="high")}
    report = build_preference_report([item("1", 1), item("1", 4)], SLOTS, prefs)   # Mon has two
    _, checks = checks_of(report, "1")
    assert (checks["max_classes"]["met"], checks["max_classes"]["total"]) == (1, 2)
    assert checks["max_classes"]["misses"] == ["Mon: 2 classes"]


def test_short_change_over_is_not_a_gap_but_a_long_wait_is():
    prefs = {"1": pref(gaps_priority="medium")}
    back_to_back = build_preference_report([item("1", 1), item("1", 2)], SLOTS, prefs)
    long_wait = build_preference_report([item("1", 1), item("1", 4)], SLOTS, prefs)
    assert checks_of(back_to_back, "1")[1]["gaps"]["status"] == "met"
    gaps = checks_of(long_wait, "1")[1]["gaps"]
    assert gaps["status"] == "missed"
    assert gaps["misses"] == ["Mon: 8 h 45 min idle between classes"]


def test_instructor_without_classes_is_listed_but_not_judged():
    prefs = {"1": pref(time_priority="low", preferred_times=["evening"]), "2": pref(gaps_priority="low")}
    report = build_preference_report([item("1", 4)], SLOTS, prefs, names={"1": "A", "2": "B"})
    assert report["summary"]["teaching"] == 1
    assert report["summary"]["no_classes"] == 1
    assert report["summary"]["met"] == 1
    assert report["instructors"][-1]["status"] == "no_classes"
    assert report["instructors"][0]["name"] == "A"


def test_cost_matches_the_optimizer_penalty():
    prefs = {
        "1": pref(preferred_days=["M"], days_priority="high", preferred_times=["morning"],
                  time_priority="medium", gaps_priority="low", fewer_days_priority="high",
                  max_classes_per_day=1, max_classes_priority="medium"),
        "2": pref(avoid_days=["T"], days_priority="low"),
    }
    items = [item("1", 1), item("1", 4), item("1", 3), item("2", 3), item("2", 2)]
    report = build_preference_report(items, SLOTS, prefs)
    assert report["summary"]["total_cost"] == round(
        calculate_instructor_preference_penalty(items, SLOTS, prefs), 2)


def test_totals_add_up_by_check_and_priority():
    prefs = {"1": pref(preferred_times=["morning"], time_priority="high"),
             "2": pref(preferred_times=["morning"], time_priority="high")}
    report = build_preference_report([item("1", 1), item("2", 4)], SLOTS, prefs)
    assert report["by_check"]["time"] == {"met": 1, "total": 2, "instructors": 2, "fully_met": 1}
    assert report["by_priority"]["high"] == {"checks": 2, "fully_met": 1}
    assert report["instructors"][0]["inst_id"] == "2"          # worst first


def test_instructor_names_are_read_in_pages():
    pages = [[{"inst_id": 1, "name": "A"}, {"inst_id": 2, "name": "B"}], [{"inst_id": 3, "name": "C"}]]

    class Table:
        def select(self, *_):
            return self

        def order(self, *_):
            return self

        def range(self, *_):
            return self

        def execute(self):
            return SimpleNamespace(data=pages.pop(0))

    client = SimpleNamespace(table=lambda _: Table())
    assert load_instructor_names(client, page_size=2) == {"1": "A", "2": "B", "3": "C"}
