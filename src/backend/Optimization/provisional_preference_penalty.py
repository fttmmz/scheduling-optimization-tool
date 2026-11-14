"""Instructor preference penalty calculation for timetable optimization.

The same penalty function is used by the shared fitness function and
incremental candidate ranking. It evaluates only classes with an
assigned teaching time.

All preference penalties are soft and do not affect hard-constraint checks.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

# One unit for low priority, two for medium, three for high.
PRIORITY_WEIGHTS = {"low": 1.0, "medium": 2.0, "high": 3.0}


def _minutes(value):
    if value is None:
        return None
    if hasattr(value, "hour"):
        return value.hour * 60 + value.minute
    try:
        hour, minute = str(value).split(":")[:2]
        return int(hour) * 60 + int(minute)
    except (TypeError, ValueError):
        return None


def _event(timeslot):
    """Represent one meeting as (teaching days, start minute, end minute)."""
    if timeslot is None:
        return None
    start = _minutes(getattr(timeslot, "start", None))
    end = _minutes(getattr(timeslot, "end", None))
    days = tuple(dict.fromkeys(day for day in str(getattr(timeslot, "day", ""))
                                   if day in "MTWR"))
    if not days or start is None or end is None or end <= start:
        return None
    return (days, start, end)


def _teaching_instructors(item):
    # Use the same instructor filtering as the hard-conflict checks: a
    # supervision roster does not represent a teaching meeting.
    from backend.Optimization.constraints import instructor_occupancy_ids
    return tuple(str(value) for value in instructor_occupancy_ids(item))


def _cost(events, preferences):
    """Compute unweighted preference units for one instructor's meetings."""
    if not events or not preferences:
        return 0.0
    cost = 0.0
    daily = defaultdict(list)
    for days, start, end in events:
        band = "morning" if start < 12 * 60 else "afternoon" if start < 17 * 60 else "evening"
        time_weight = PRIORITY_WEIGHTS.get(preferences.get("time_priority"), 0)
        if time_weight and band not in preferences.get("preferred_times", ()):
            cost += 2.0 * time_weight

        day_weight = PRIORITY_WEIGHTS.get(preferences.get("days_priority"), 0)
        preferred = set(preferences.get("preferred_days", ()))
        avoided = set(preferences.get("avoid_days", ()))
        for day in days:
            if day_weight and day in avoided:
                cost += 3.0 * day_weight
            elif day_weight and preferred and day not in preferred:
                cost += 2.0 * day_weight
            daily[day].append((start, end))

    days_weight = PRIORITY_WEIGHTS.get(preferences.get("fewer_days_priority"), 0)
    if days_weight:
        cost += 1.5 * days_weight * max(0, len(daily) - 2)

    max_weight = PRIORITY_WEIGHTS.get(preferences.get("max_classes_priority"), 0)
    maximum = preferences.get("max_classes_per_day")
    if max_weight and maximum is not None:
        cost += 2.0 * max_weight * sum(max(0, len(meetings) - maximum)
                                         for meetings in daily.values())

    gap_weight = PRIORITY_WEIGHTS.get(preferences.get("gaps_priority"), 0)
    if gap_weight:
        for meetings in daily.values():
            meetings.sort()
            previous_end = None
            for start, end in meetings:
                if previous_end is not None and start > previous_end:
                    cost += gap_weight * (start - previous_end) / 60.0
                previous_end = max(previous_end or end, end)
    return cost


def calculate_instructor_preference_penalty(schedule, timeslots, preferences_by_instructor):
    """Return the temporary, nonnegative preference penalty before lambda."""
    tracker = InstructorPreferenceTracker(timeslots, preferences_by_instructor)
    for item in schedule:
        tracker.add(item, tracker.timeslot_map.get(item.timeslot_id))
    return tracker.total()


class InstructorPreferenceTracker:
    """Maintain per-instructor meetings for efficient candidate/move scoring.

    ``delta`` and ``move_delta`` return weighted *changes* in penalty.
    Negative values indicate an improvement. Inactive configurations have
    exactly zero impact on candidate ranking.
    """

    def __init__(self, timeslots, preferences_by_instructor=None, weight=1.0):
        self.timeslot_map = {ts.id: ts for ts in timeslots}
        self.preferences = preferences_by_instructor or {}
        self.weight = weight
        self.events = defaultdict(list)

    @classmethod
    def from_config(cls, timeslots, config):
        return cls(timeslots,
                   config.by_instructor if config and config.active else {},
                   config.weight if config and config.active else 0.0)

    def add(self, item, timeslot):
        event = _event(timeslot)
        if event is not None:
            for inst in _teaching_instructors(item):
                if inst in self.preferences:
                    self.events[inst].append(event)

    def delta(self, item, timeslot):
        event = _event(timeslot)
        if not self.weight or event is None:
            return 0.0
        change = 0.0
        for inst in _teaching_instructors(item):
            pref = self.preferences.get(inst)
            if pref is not None:
                old = self.events[inst]
                change += _cost(old + [event], pref) - _cost(old, pref)
        return self.weight * change

    def move_delta(self, item, old_timeslot, new_timeslot):
        old_event = _event(old_timeslot)
        new_event = _event(new_timeslot)
        if not self.weight or old_event == new_event:
            return 0.0
        change = 0.0
        for inst in _teaching_instructors(item):
            pref = self.preferences.get(inst)
            if pref is None:
                continue
            before = self.events[inst]
            after = before.copy()
            if old_event is not None:
                after.remove(old_event)
            if new_event is not None:
                after.append(new_event)
            change += _cost(after, pref) - _cost(before, pref)
        return self.weight * change

    def move(self, item, old_timeslot, new_timeslot):
        old_event = _event(old_timeslot)
        new_event = _event(new_timeslot)
        if old_event == new_event:
            return
        for inst in _teaching_instructors(item):
            if inst not in self.preferences:
                continue
            if old_event is not None:
                self.events[inst].remove(old_event)
            if new_event is not None:
                self.events[inst].append(new_event)

    def total(self):
        return sum(_cost(events, self.preferences[inst])
                   for inst, events in self.events.items())
