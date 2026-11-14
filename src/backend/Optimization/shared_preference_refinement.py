"""Shared instructor preference refinement for PSO, RL, and ML-Greedy.

Runs only when instructor preferences are enabled.
Uses a limited refinement budget to control runtime.
Preserves the initial number of hard conflicts and scheduled sections.
"""
import copy
import random

from backend.Optimization.constraints import (
    get_valid_timeslots, instructor_occupancy_ids, section_needs,
)
from backend.Optimization.evaluation import (
    count_hard_conflicts, count_scheduled_sections,
)
from backend.Optimization.instructor_preferences import preference_penalty


def refine_preferences_shared(schedule, sections, timeslots, rooms, config, *,
                              seed=1, max_evaluations=240, max_passes=2,
                              label="Optimizer"):
    if config is None or not config.active or not schedule:
        return schedule

    original = copy.deepcopy(schedule)
    working = copy.deepcopy(schedule)
    hard_limit = count_hard_conflicts(original, rooms)
    scheduled_limit = count_scheduled_sections(original, sections)
    initial = preference_penalty(original, timeslots, config)
    current = initial
    rng = random.Random(seed if seed is not None else 1)

    # Pattern index is important: multiple meeting blocks may share a course/section.
    section_by_key = {}
    for sec in sections:
        key = (str(sec.course.id), str(sec.no), str(getattr(sec, "pattern_index", 0)))
        section_by_key[key] = sec

    valid_ids = {}
    for key, sec in section_by_key.items():
        if section_needs(sec).time:
            valid_ids[key] = [ts.id for ts in get_valid_timeslots(sec, timeslots)]

    eligible = []
    for i, item in enumerate(working):
        key = (str(item.course_id), str(item.section),
               str(getattr(item, "pattern_index", 0)))
        sec = section_by_key.get(key)
        if (sec is None or item.timeslot_id is None or key not in valid_ids
                or not any(str(t) in config.by_instructor
                           for t in instructor_occupancy_ids(item))):
            continue
        eligible.append((i, key))

    evaluations = 0
    accepted = 0
    for _ in range(max_passes):
        rng.shuffle(eligible)
        improved = False
        for i, key in eligible:
            if evaluations >= max_evaluations:
                break
            item = working[i]
            old_slot = item.timeslot_id
            alternatives = [tid for tid in valid_ids[key] if tid != old_slot]
            rng.shuffle(alternatives)
            best_slot, best_penalty = None, current

            # Best improvement among candidates within the shared evaluation budget.
            for tid in alternatives:
                if evaluations >= max_evaluations:
                    break
                evaluations += 1
                item.timeslot_id = tid
                if (count_hard_conflicts(working, rooms) <= hard_limit
                        and count_scheduled_sections(working, sections) >= scheduled_limit):
                    penalty = preference_penalty(working, timeslots, config)
                    if penalty < best_penalty - 1e-9:
                        best_slot, best_penalty = tid, penalty
                item.timeslot_id = old_slot

            if best_slot is not None:
                item.timeslot_id = best_slot
                current = best_penalty
                accepted += 1
                improved = True
        if evaluations >= max_evaluations or not improved:
            break

    # Final safeguard: do not return a worse or less feasible timetable.
    if (count_hard_conflicts(working, rooms) > hard_limit
            or count_scheduled_sections(working, sections) < scheduled_limit
            or current > initial + 1e-9):
        print(f"{label} preferences: safeguard restored baseline")
        return original
    print(f"{label} preferences: {accepted} accepted moves, "
          f"{evaluations}/{max_evaluations} candidates checked; "
          f"penalty {initial:.3f} -> {current:.3f}")
    return working
