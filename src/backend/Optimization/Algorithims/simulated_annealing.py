"""
Simulated Annealing for University Course Timetabling.

Based on:
Sylejmani, K., Gashi, E., & Ymeri, A. (2023).
"Simulated annealing with penalization for university course timetabling."
Journal of Scheduling, 26, 497-517.
https://doi.org/10.1007/s10951-022-00747-5

This implementation uses the project's existing constraint and fitness
functions and applies a standard simulated annealing search.
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

        # Nothing needs to be assigned.
        if not needs.room and not needs.time:
            found = True

        else:
            # Randomise equally valid possibilities so SA runs are not identical.
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

        # If nothing conflict-free exists, leave the required placement empty.
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
            course_class=getattr(section.course, "course_class", None),
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


def _make_neighbor(schedule, rooms, timeslots, movable, occupied_rooms, occupied_instructors):
    """Create a nearby schedule by changing one assignment."""

    neighbor = list(schedule)

    if not neighbor:
        return neighbor

    selected_index = random.choice(movable)

    neighbor[selected_index] = copy.copy(schedule[selected_index])
    selected_item = neighbor[selected_index]

    item_needs = needs_for(
        selected_item.course_type,
        getattr(selected_item, "level", None),
    )

    occupied_rooms = set()
    occupied_instructors = set()

    # Build occupancy without the item that is being moved.
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

    # Decide whether to change room or time.
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

        if viable_rooms:
            candidates = list(viable_rooms)
            random.shuffle(candidates)

            for room in candidates:

                if selected_item.timeslot_id is None:
                    selected_item.room_id = room.id
                    break

                if (
                    room.id,
                    selected_item.timeslot_id,
                ) not in occupied_rooms:

                    selected_item.room_id = room.id
                    break

    # ---------- Change timeslot ----------
    else:

        candidates = list(timeslots)
        random.shuffle(candidates)

        selected_instructors = instructor_occupancy_ids(
            selected_item
        )

        for ts in candidates:

            instructor_free = all(
                (instructor_id, ts.id)
                not in occupied_instructors
                for instructor_id in selected_instructors
            )

            room_free = (
                selected_item.room_id is None
                or (
                    selected_item.room_id,
                    ts.id,
                ) not in occupied_rooms
            )

            if instructor_free and room_free:
                selected_item.timeslot_id = ts.id
                break

    return neighbor, selected_index


def simulated_annealing_schedule(
    sections,
    timeslots,
    rooms,
    valid_timeslot_cache=None,
):
    """Run one Simulated Annealing search."""

    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(
            sections,
            timeslots,
        )

    current = _build_initial_schedule(
        sections,
        timeslots,
        rooms,
    )

    current = _build_initial_schedule(
    sections,
    timeslots,
    rooms,
)

occupied_rooms = set()
occupied_instructors = set()

for item in current:
    if item.room_id is not None and item.timeslot_id is not None:
        occupied_rooms.add(
            (item.room_id, item.timeslot_id)
        )

    if item.timeslot_id is not None:
        for instructor_id in instructor_occupancy_ids(item):
            occupied_instructors.add(
                (instructor_id, item.timeslot_id)
            )

movable = [
            
    movable = [
    index
    for index, item in enumerate(current)
    if (
        needs_for(
            item.course_type,
            getattr(item, "level", None),
        ).room
        or needs_for(
            item.course_type,
            getattr(item, "level", None),
        ).time
    )
]

    current_fitness = calculate_fitness(
        current,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )

    best = copy.deepcopy(current)
    best_fitness = current_fitness

    temperature = INITIAL_TEMPERATURE

    while temperature > MIN_TEMPERATURE:

        for _ in range(ITERATIONS_PER_TEMPERATURE):

            neighbor, selected_index = _make_neighbor(
                current,
                rooms,
                timeslots,
                movable,
                occupied_rooms,
                occupied_instructors,
            )

            neighbor_fitness = calculate_fitness(
                neighbor,
                rooms,
                sections=sections,
                timeslots=timeslots,
                valid_timeslot_cache=valid_timeslot_cache,
            )

            difference = neighbor_fitness - current_fitness

            # Always accept an improvement.
            if difference >= 0:
                accept = True

            # Sometimes accept a worse solution.
            else:
                probability = math.exp(
                    difference / temperature
                )
                accept = random.random() < probability

            if accept:
                current = neighbor
                current_fitness = neighbor_fitness

            # Keep the best solution found during the whole search.
            if current_fitness > best_fitness:
                best = copy.deepcopy(current)
                best_fitness = current_fitness

        # Cool the temperature.
        temperature *= COOLING_RATE

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
    """Run SA several times and return the best schedule."""

    valid_timeslot_cache = build_timeslot_guideline_cache(
        sections,
        timeslots,
    )

    best_overall_schedule = None
    best_overall_fitness = -1.0

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
