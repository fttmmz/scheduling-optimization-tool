
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
import math
import random

from backend.Optimization.constraints import (
    section_needs,
    get_viable_rooms,
    get_viable_rooms_for_schedule_item,
    get_valid_timeslots_for_section,
    passes_hard_constraints,
    instructor_occupancy_ids,
    finalize_schedule,
    needs_for,
)

from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
)

from backend.Optimization.benchmark_seeds import (
    seed_all,
    seed_for_run,
)

from backend.models.models import ScheduleItem


# Simulated Annealing settings
INITIAL_TEMPERATURE = 1.0
MIN_TEMPERATURE = 0.001
COOLING_RATE = 0.995
ITERATIONS_PER_TEMPERATURE = 5


def _build_initial_schedule(sections, timeslots, rooms):
    """Create one valid starting schedule."""

    schedule = []
    occupied_rooms = set()
    occupied_instructors = set()

    for section in sections:
        needs = section_needs(section)

        viable_rooms = (
            get_viable_rooms(section, rooms)
            if needs.room else [None]
        )

        valid_timeslots = (
            get_valid_timeslots_for_section(section, timeslots)
            if needs.time else [None]
        )

        room = None
        timeslot = None
        found = False

        if not needs.room and not needs.time:
            found = True

        else:
            room_candidates = list(viable_rooms)
            time_candidates = list(valid_timeslots)

            random.shuffle(room_candidates)
            random.shuffle(time_candidates)

            for ts in time_candidates:
                for rm in room_candidates:

                    if passes_hard_constraints(
                        section,
                        rm,
                        ts,
                        occupied_instructors=occupied_instructors,
                        occupied_rooms=occupied_rooms,
                    ):
                        room = rm
                        timeslot = ts
                        found = True
                        break

                if found:
                    break

        if not found:
            room = None
            timeslot = None

        item = ScheduleItem(
            course_id=section.course.id,
            course_name=section.course.name,
            course_type=section.course.type,
            course_dept=section.course.dept,
            capacity=section.capacity,
            instructor_id=section.instructor_id,
            room_id=room.id if room else None,
            timeslot_id=timeslot.id if timeslot else None,
            section=str(section.no),
            instructor_ids=section.instructor_ids,
            pattern_index=section.pattern_index,
            level=getattr(section.course, "level", None),
            course_class=getattr(
                section.course, "course_class", None
            ),
        )

        schedule.append(item)

        if room is not None and timeslot is not None:
            occupied_rooms.add((room.id, timeslot.id))

        if timeslot is not None:
            for instructor_id in instructor_occupancy_ids(section):
                occupied_instructors.add(
                    (instructor_id, timeslot.id)
                )

    return schedule


def _make_neighbor(schedule, sections, rooms, timeslots):
    """Create a neighboring schedule by changing one assignment."""

    # Copy the list without copying every ScheduleItem.
    neighbor = schedule.copy()

    if not neighbor:
        return neighbor

    movable = []

    for index, item in enumerate(neighbor):
        needs = needs_for(
            item.course_type,
            getattr(item, "level", None),
        )

        if needs.room or needs.time:
            movable.append(index)

    if not movable:
        return neighbor

    selected_index = random.choice(movable)

    # Copy only the section that will be modified.
    neighbor[selected_index] = copy.copy(
        neighbor[selected_index]
    )

    selected_item = neighbor[selected_index]

    # Original section object is needed for constraint checking.
    selected_section = sections[selected_index]

    item_needs = section_needs(selected_section)

    occupied_rooms = set()
    occupied_instructors = set()

    # Track occupied resources, excluding selected section.
    for index, item in enumerate(neighbor):

        if index == selected_index:
            continue

        if item.room_id is not None and item.timeslot_id is not None:
            occupied_rooms.add(
                (item.room_id, item.timeslot_id)
            )

        if item.timeslot_id is not None:
            for instructor_id in instructor_occupancy_ids(item):
                occupied_instructors.add(
                    (instructor_id, item.timeslot_id)
                )

    # Create lookup dictionaries.
    room_by_id = {room.id: room for room in rooms}
    timeslot_by_id = {ts.id: ts for ts in timeslots}

    current_room = room_by_id.get(selected_item.room_id)
    current_timeslot = timeslot_by_id.get(
        selected_item.timeslot_id
    )

    # Decide whether to change room or timeslot.
    if item_needs.room and item_needs.time:
        change_room = random.random() < 0.5
    else:
        change_room = item_needs.room

    # ---------- Change room ----------
    if change_room:

        viable_rooms = get_viable_rooms_for_schedule_item(
            selected_item,
            rooms,
        )

        candidates = list(viable_rooms)
        random.shuffle(candidates)

        for room in candidates:

            # Skip the current room.
            if room.id == selected_item.room_id:
                continue

            if passes_hard_constraints(
                selected_section,
                room,
                current_timeslot,
                occupied_instructors=occupied_instructors,
                occupied_rooms=occupied_rooms,
            ):
                selected_item.room_id = room.id
                break

    # ---------- Change timeslot ----------
    else:

        # Only consider timeslots valid for this section.
        valid_timeslots = get_valid_timeslots_for_section(
            selected_section,
            timeslots,
        )

        candidates = list(valid_timeslots)
        random.shuffle(candidates)

        for ts in candidates:

            # Skip the current timeslot.
            if ts.id == selected_item.timeslot_id:
                continue

            if passes_hard_constraints(
                selected_section,
                current_room,
                ts,
                occupied_instructors=occupied_instructors,
                occupied_rooms=occupied_rooms,
            ):
                selected_item.timeslot_id = ts.id
                break

    return neighbor


def simulated_annealing_schedule(
    sections,
    timeslots,
    rooms,
    valid_timeslot_cache=None,
):
    """Run one Simulated Annealing search."""

    # Ensure sections can be accessed by index.
    sections = list(sections)

    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(
            sections,
            timeslots,
        )

    # Generate initial timetable.
    current = _build_initial_schedule(
        sections,
        timeslots,
        rooms,
    )

    current_fitness = calculate_fitness(
        current,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )

    # Store best solution.
    best = copy.deepcopy(current)
    best_fitness = current_fitness

    temperature = INITIAL_TEMPERATURE

    # Main Simulated Annealing loop.
    while temperature > MIN_TEMPERATURE:

        for _ in range(ITERATIONS_PER_TEMPERATURE):

            # Generate neighboring solution.
            neighbor = _make_neighbor(
                current,
                sections,
                rooms,
                timeslots,
            )

            # Evaluate neighboring solution.
            neighbor_fitness = calculate_fitness(
                neighbor,
                rooms,
                sections=sections,
                timeslots=timeslots,
                valid_timeslot_cache=valid_timeslot_cache,
            )

            difference = neighbor_fitness - current_fitness

            # Always accept better or equal solutions.
            if difference >= 0:
                accept = True

            # Sometimes accept worse solutions.
            else:
                probability = math.exp(
                    difference / temperature
                )

                accept = random.random() < probability

            if accept:
                current = neighbor
                current_fitness = neighbor_fitness

            # Update best solution found.
            if current_fitness > best_fitness:
                best = copy.deepcopy(current)
                best_fitness = current_fitness

        # Reduce temperature.
        temperature *= COOLING_RATE

    # Finalize and return best timetable.
    return finalize_schedule(
        best,
        sections,
        timeslots,
        valid_timeslot_cache,
        rooms=rooms,
    )


def simulated_annealing_runs(
    sections,
    timeslots,
    rooms,
    num_runs=30,
    seeds=None,
):
    """Run SA multiple times and return the best schedule."""

    # Convert once to maintain stable section ordering.
    sections = list(sections)

    valid_timeslot_cache = build_timeslot_guideline_cache(
        sections,
        timeslots,
    )

    best_overall_schedule = None
    best_overall_fitness = float("-inf")

    print(
        f"\n=== Simulated Annealing: {num_runs} runs ==="
    )

    for run in range(num_runs):

        run_seed = seed_all(
            seed_for_run(run, seeds)
        )

        schedule = simulated_annealing_schedule(
            sections,
            timeslots,
            rooms,
            valid_timeslot_cache=valid_timeslot_cache,
        )

        fitness = calculate_fitness(
            schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )

        print(
            f"Run {run + 1:2d} "
            f"(seed {run_seed}): "
            f"fitness = {fitness:.4f}"
        )

        if fitness > best_overall_fitness:
            best_overall_fitness = fitness
            best_overall_schedule = schedule

    return best_overall_schedule

