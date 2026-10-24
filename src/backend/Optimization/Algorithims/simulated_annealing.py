
"""
Simulated Annealing for University Course Timetabling.

Based on:
Sylejmani, K., Gashi, E., & Ymeri, A. (2023).
"Simulated annealing with penalization for university course timetabling."
Journal of Scheduling, 26, 497-517.
https://doi.org/10.1007/s10951-022-00747-5

Uses the project's existing constraint and fitness functions.
"""

import copy
import inspect
import math
import random

from backend.Optimization.constraints import (
    OverlapSet,
    finalize_schedule,
    get_valid_timeslots_for_section,
    get_viable_rooms,
    instructor_occupancy_ids,
    passes_hard_constraints,
    section_needs,
)
from backend.Optimization.evaluation import (
    build_timeslot_guideline_cache,
    calculate_fitness,
)
from backend.Optimization.benchmark_seeds import seed_all, seed_for_run
from backend.models.models import ScheduleItem

INITIAL_TEMPERATURE = 1.0
MIN_TEMPERATURE = 0.001
COOLING_RATE = 0.995
ITERATIONS_PER_TEMPERATURE = 5


def _make_item(section, room, timeslot):
    """Build a ScheduleItem, keeping optional metadata when the model supports it."""
    values = {
        "course_id": section.course.id,
        "course_name": section.course.name,
        "course_type": section.course.type,
        "course_dept": section.course.dept,
        "capacity": section.capacity,
        "instructor_id": section.instructor_id,
        "room_id": room.id if room is not None else None,
        "timeslot_id": timeslot.id if timeslot is not None else None,
        "section": str(section.no),
        "instructor_ids": section.instructor_ids,
        "pattern_index": section.pattern_index,
        "level": getattr(section.course, "level", None),
        "course_class": getattr(section.course, "course_class", None),
    }

    # Preserve metadata used by room rules and the registrar if ScheduleItem
    # defines corresponding constructor fields; do not assume all versions do.
    optional = {
        "campus": getattr(section, "campus", None),
        "enrolment": getattr(section, "enrolment", None),
        "sibling_key": getattr(section, "sibling_key", None),
    }
    try:
        params = inspect.signature(ScheduleItem).parameters
    except (TypeError, ValueError):
        params = {}
    for key, value in optional.items():
        if key in params:
            values[key] = value

    item = ScheduleItem(**values)
    # Also support models that expose these fields as writable attributes but
    # do not declare them as constructor parameters.
    for key, value in optional.items():
        if key not in params and hasattr(item, key):
            try:
                setattr(item, key, value)
            except (AttributeError, TypeError):
                pass
    return item


def _occupancy(schedule, excluded_index=None):
    """Collect real-time room and instructor bookings, optionally excluding one item."""
    busy_rooms = OverlapSet()
    busy_instructors = OverlapSet()
    for index, item in enumerate(schedule):
        if index == excluded_index or item.timeslot_id is None:
            continue
        if item.room_id is not None:
            busy_rooms.add((item.room_id, item.timeslot_id))
        for instructor_id in instructor_occupancy_ids(item):
            busy_instructors.add((instructor_id, item.timeslot_id))
    return busy_rooms, busy_instructors


def _try_placement(section, candidate_rooms, candidate_timeslots,
                   busy_rooms, busy_instructors):
    """Find one feasible room/time combination; None is valid for an unused axis."""
    for ts in candidate_timeslots:
        for room in candidate_rooms:
            if passes_hard_constraints(
                section, room, ts,
                occupied_rooms=busy_rooms,
                occupied_instructors=busy_instructors,
            ):
                return room, ts, True
    return None, None, False


def _build_initial_schedule(sections, timeslots, rooms):
    """Construct a starting schedule, including items that need no booking."""
    schedule = []
    occupied_rooms = OverlapSet()
    occupied_instructors = OverlapSet()

    for section in sections:
        needs = section_needs(section)
        room_candidates = (list(get_viable_rooms(section, rooms))
                           if needs.room else [None])
        time_candidates = (list(get_valid_timeslots_for_section(section, timeslots))
                           if needs.time else [None])
        random.shuffle(room_candidates)
        random.shuffle(time_candidates)

        room, ts, found = _try_placement(
            section, room_candidates, time_candidates,
            occupied_rooms, occupied_instructors,
        )
        if not found:
            room, ts = None, None

        item = _make_item(section, room, ts)
        schedule.append(item)
        if ts is not None:
            if room is not None:
                occupied_rooms.add((room.id, ts.id))
            for instructor_id in instructor_occupancy_ids(section):
                occupied_instructors.add((instructor_id, ts.id))
    return schedule


def _make_neighbor(schedule, sections, rooms, timeslots):
    """Move one section to a feasible assignment, or try to schedule a missing one."""
    if not schedule:
        return schedule.copy()

    # In our constructor one ScheduleItem is created for each Section, in order.
    # Access the Section, not needs_for(course_type, level), so registrar
    # overrides and branch-campus rules are respected.
    movable = [
        i for i, section in enumerate(sections)
        if section_needs(section).room or section_needs(section).time
    ]
    if not movable:
        return schedule.copy()

    # Prefer rescuing an unassigned section when one exists; otherwise use the
    # normal SA random move. Rescuing is still a candidate, subject to fitness.
    missing = [
        i for i in movable
        if ((section_needs(sections[i]).room and schedule[i].room_id is None)
            or (section_needs(sections[i]).time and schedule[i].timeslot_id is None))
    ]
    index = random.choice(missing if missing else movable)
    section = sections[index]
    needs = section_needs(section)
    original = schedule[index]

    neighbor = schedule.copy()
    neighbor[index] = copy.copy(original)
    selected = neighbor[index]
    busy_rooms, busy_instructors = _occupancy(schedule, excluded_index=index)

    room_by_id = {room.id: room for room in rooms}
    timeslot_by_id = {ts.id: ts for ts in timeslots}
    current_room = room_by_id.get(original.room_id)
    current_ts = timeslot_by_id.get(original.timeslot_id)

    # Missing assignments must be built as complete room/time combinations.
    # This also handles time-only and room-only sections.
    incomplete = ((needs.room and current_room is None)
                  or (needs.time and current_ts is None))
    if incomplete:
        candidates_rooms = (list(get_viable_rooms(section, rooms))
                            if needs.room else [None])
        candidates_times = (list(get_valid_timeslots_for_section(section, timeslots))
                            if needs.time else [None])
        random.shuffle(candidates_rooms)
        random.shuffle(candidates_times)
        room, ts, found = _try_placement(
            section, candidates_rooms, candidates_times,
            busy_rooms, busy_instructors,
        )
        if found:
            selected.room_id = room.id if room is not None else None
            selected.timeslot_id = ts.id if ts is not None else None
        return neighbor

    # For fully assigned sections, change exactly one axis.
    change_room = (random.random() < 0.5 if needs.room and needs.time
                   else needs.room)
    if change_room:
        candidates = list(get_viable_rooms(section, rooms))
        random.shuffle(candidates)
        for room in candidates:
            if room.id == original.room_id:
                continue
            if passes_hard_constraints(
                section, room, current_ts,
                occupied_rooms=busy_rooms,
                occupied_instructors=busy_instructors,
            ):
                selected.room_id = room.id
                break
    else:
        candidates = list(get_valid_timeslots_for_section(section, timeslots))
        random.shuffle(candidates)
        for ts in candidates:
            if ts.id == original.timeslot_id:
                continue
            if passes_hard_constraints(
                section, current_room, ts,
                occupied_rooms=busy_rooms,
                occupied_instructors=busy_instructors,
            ):
                selected.timeslot_id = ts.id
                break
    return neighbor


def simulated_annealing_schedule(sections, timeslots, rooms,
                                 valid_timeslot_cache=None):
    """Run one SA search, maximizing the project's fitness score."""
    sections = list(sections)
    timeslots = list(timeslots)
    rooms = list(rooms)
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    current = _build_initial_schedule(sections, timeslots, rooms)
    current_fitness = calculate_fitness(
        current, rooms, sections=sections, timeslots=timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    best = copy.deepcopy(current)
    best_fitness = current_fitness
    temperature = INITIAL_TEMPERATURE

    while temperature > MIN_TEMPERATURE:
        for _ in range(ITERATIONS_PER_TEMPERATURE):
            neighbor = _make_neighbor(current, sections, rooms, timeslots)
            neighbor_fitness = calculate_fitness(
                neighbor, rooms, sections=sections, timeslots=timeslots,
                valid_timeslot_cache=valid_timeslot_cache,
            )
            delta = neighbor_fitness - current_fitness
            if delta >= 0 or random.random() < math.exp(delta / temperature):
                current, current_fitness = neighbor, neighbor_fitness

            if current_fitness > best_fitness:
                best = copy.deepcopy(current)
                best_fitness = current_fitness
        temperature *= COOLING_RATE

    return finalize_schedule(
        best, sections, timeslots, valid_timeslot_cache, rooms=rooms,
    )


def simulated_annealing_runs(sections, timeslots, rooms, num_runs=30, seeds=None):
    """Run SA repeatedly with controlled seeds and return the highest-fitness run."""
    sections = list(sections)
    timeslots = list(timeslots)
    rooms = list(rooms)
    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    best_overall_schedule = None
    best_overall_fitness = float("-inf")

    print(f"\n=== Simulated Annealing: {num_runs} runs ===")
    for run in range(num_runs):
        run_seed = seed_all(seed_for_run(run, seeds))
        schedule = simulated_annealing_schedule(
            sections, timeslots, rooms,
            valid_timeslot_cache=valid_timeslot_cache,
        )
        fitness = calculate_fitness(
            schedule, rooms, sections=sections, timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )
        print(f"Run {run + 1:2d} (seed {run_seed}): fitness = {fitness:.4f}")
        if fitness > best_overall_fitness:
            best_overall_fitness = fitness
            best_overall_schedule = schedule
    return best_overall_schedule
