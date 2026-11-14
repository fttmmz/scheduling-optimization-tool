"""Feasibility-preserving instructor preference refinement shared by Greedy and GRASP.

Only valid single-section timeslot relocations are considered. A final exact
feasibility check protects against sibling constraints and other global rules.
"""
import copy
import random
from backend.Optimization.constraints import (
    OverlapSet, get_valid_timeslots, instructor_occupancy_ids, section_needs,
)
from backend.Optimization.evaluation import count_hard_conflicts, count_scheduled_sections
from backend.Optimization.instructor_preferences import current_preference_config, preference_penalty
from backend.Optimization.provisional_preference_penalty import InstructorPreferenceTracker


def refine_preferences(schedule, sections, timeslots, rooms, seed=1, max_items=450,
                       max_slots=24, label='Scheduler'):
    config = current_preference_config()
    if config is None or not config.active:
        return schedule
    original = schedule
    working = copy.deepcopy(schedule)
    base_hard = count_hard_conflicts(original, rooms)
    base_scheduled = count_scheduled_sections(original, sections)
    original_penalty = preference_penalty(original, timeslots, config)
    tracker = InstructorPreferenceTracker.from_config(timeslots, config)
    for item in working:
        tracker.add(item, tracker.timeslot_map.get(item.timeslot_id))
    # ScheduleItems may not be in input order, and multiple meeting blocks
    # can share a course/section identity; use the same matching convention as
    # the other preference refinement implementations.
    section_map = {}
    for section in sections:
        section_map.setdefault((section.course.id, str(section.no)), section)
    rng = random.Random(seed)
    indexes = [i for i, item in enumerate(working)
               if item.timeslot_id is not None
               and any(str(inst) in config.by_instructor
                       for inst in instructor_occupancy_ids(item))]
    rng.shuffle(indexes)
    accepted = 0
    for idx in indexes[:max_items]:
        item = working[idx]
        section = section_map.get((item.course_id, str(item.section)))
        if section is None or not section_needs(section).time:
            continue
        old_ts = tracker.timeslot_map.get(item.timeslot_id)
        if old_ts is None:
            continue
        # Build occupancy without the moved item. OverlapSet handles real-time
        # overlaps, not just identical timeslot IDs.
        busy_rooms, busy_instructors = OverlapSet(), OverlapSet()
        for j, other in enumerate(working):
            if j == idx or other.timeslot_id is None:
                continue
            if other.room_id is not None:
                busy_rooms.add((other.room_id, other.timeslot_id))
            for inst in instructor_occupancy_ids(other):
                busy_instructors.add((inst, other.timeslot_id))
        instructors = instructor_occupancy_ids(item)
        candidates = []
        for ts in get_valid_timeslots(section, timeslots):
            if ts.id == item.timeslot_id:
                continue
            delta = tracker.move_delta(item, old_ts, ts)
            if delta < -1e-8:
                candidates.append((delta, rng.random(), ts))
        candidates.sort(key=lambda entry: (entry[0], entry[1]))
        for _, _, ts in candidates[:max_slots]:
            if item.room_id is not None and (item.room_id, ts.id) in busy_rooms:
                continue
            if any((inst, ts.id) in busy_instructors for inst in instructors):
                continue
            # Exact hard-conflict validation also covers sibling/linked blocks.
            prior = item.timeslot_id
            item.timeslot_id = ts.id
            if count_hard_conflicts(working, rooms) > base_hard:
                item.timeslot_id = prior
                continue
            tracker.move(item, old_ts, ts)
            accepted += 1
            break
    new_penalty = preference_penalty(working, timeslots, config)
    if (count_hard_conflicts(working, rooms) > base_hard
            or count_scheduled_sections(working, sections) < base_scheduled
            or new_penalty >= original_penalty - 1e-8):
        print(f'{label} preferences: no feasible improvement; retaining OFF baseline')
        return original
    print(f'{label} preferences: accepted {accepted} moves; '
          f'penalty {original_penalty:.3f} -> {new_penalty:.3f} '
          f'({(original_penalty-new_penalty)/original_penalty*100:.2f}% reduction)')
    return working
