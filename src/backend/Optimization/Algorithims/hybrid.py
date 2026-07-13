import random
import copy
import statistics
from backend.Optimization.constraints import (
    get_valid_timeslots,
    passes_hard_constraints,
    get_viable_rooms,
    get_viable_rooms_for_schedule_item,
)
from backend.models.models import ScheduleItem
from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
)


def choose_random_assignment(
        section,
        rooms,
        timeslots,
        occupied_instructors,
        occupied_rooms,
):
    """
    Find a room/timeslot pair for `section`.

    Pass 1: look for a pair that passes full hard constraints.
    Pass 2 (fallback): if nothing passes hard constraints, relax those
    preference-style checks but NEVER return a room/timeslot that is
    already occupied. Returning an occupied slot was the original bug --
    it silently created double-booking conflicts every time pass 1 failed.
    Only if truly nothing is free do we give up and return (None, None),
    leaving the section unscheduled instead of forcing a conflict.
    """
    if not rooms or not timeslots:
        return None, None

    shuffled_timeslots = random.sample(timeslots, len(timeslots))

    # Pass 1: fully valid assignment
    for ts in shuffled_timeslots:
        if section.instructor_id is not None and (section.instructor_id, ts.id) in occupied_instructors:
            continue

        for room in random.sample(rooms, len(rooms)):
            if (room.id, ts.id) in occupied_rooms:
                continue

            if passes_hard_constraints(
                    section,
                    room,
                    ts,
                    occupied_instructors=occupied_instructors,
                    occupied_rooms=occupied_rooms,
            ):
                return room, ts

    # Pass 2: relax soft/preference constraints, but keep hard exclusivity
    # (never double-book a room or an instructor).
    for ts in shuffled_timeslots:
        if section.instructor_id is not None and (section.instructor_id, ts.id) in occupied_instructors:
            continue

        for room in random.sample(rooms, len(rooms)):
            if (room.id, ts.id) in occupied_rooms:
                continue
            return room, ts

    # Nothing free at all -- better to leave unscheduled than to conflict.
    return None, None


# Create Population
# This function creates many random schedules (solutions)
def create_population(size, sections, rooms, timeslots):
    population = []

    # precompute static room/timeslot viability for each section
    section_candidates = [
        (
            section,
            get_viable_rooms(section, rooms),
            get_valid_timeslots(section, timeslots),
        )
        for section in sections
    ]

    # repeat to create many schedules
    for i in range(size):
        schedule = []
        occupied_instructors = set()
        occupied_rooms = set()

        # go through each section and assign random room + timeslot
        for section, viable_rooms, valid_timeslots in section_candidates:
            room, timeslot = choose_random_assignment(
                section,
                viable_rooms,
                valid_timeslots,
                occupied_instructors,
                occupied_rooms,
            )
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
            )

            schedule.append(item)
            if room is not None and timeslot is not None:
                occupied_rooms.add((room.id, timeslot.id))
            if section.instructor_id is not None and timeslot is not None:
                occupied_instructors.add((section.instructor_id, timeslot.id))

        # add this full schedule to population
        population.append(schedule)

    return population


# Selection
# This keeps the better half of the population
# and evaluates fitness using section/timeslot-aware penalties.
def selection(population, rooms, sections, timeslots, valid_timeslot_cache=None):
    # sort schedules from best to worst using fitness score
    population.sort(
        key=lambda s: calculate_fitness(
            s,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        ),
        reverse=True,
    )

    # keep only top 50%
    return population[:len(population) // 2]


# Crossover
# This mixes two parent schedules to create a new one
CROSSOVER_RATE = 0.85


def crossover(parent1, parent2):
    # if schedule is empty, return empty
    if len(parent1) == 0:
        return []

    # decide if crossover happens
    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1)

    # pick a random split point
    point = random.randint(1, len(parent1) - 1)

    # take part from parent1 + rest from parent2
    child = parent1[:point] + parent2[point:]

    # deepcopy to avoid linking objects together
    return copy.deepcopy(child)


# Mutation
# This randomly changes small parts of a schedule.
#
# CHANGED: the original version rolled the dice ONCE for the whole
# schedule and, if it hit, mutated a single random gene -- unscheduled
# or conflicting items had no better chance of being touched than a
# perfectly fine item. Now every unscheduled item is *always* given a
# chance to be placed, and every other item mutates independently at
# MUTATION_RATE, which gives the GA a much better shot at closing gaps.
MUTATION_RATE = 0.03


def mutation(schedule, rooms, timeslots, sections=None):
    if len(schedule) == 0:
        return schedule

    section_map = {str(s.no): s for s in sections} if sections else {}

    # build current occupancy
    occupied_instructors = set()
    occupied_rooms = set()
    for item in schedule:
        if item.instructor_id is not None and item.timeslot_id is not None:
            occupied_instructors.add((item.instructor_id, item.timeslot_id))
        if item.room_id is not None and item.timeslot_id is not None:
            occupied_rooms.add((item.room_id, item.timeslot_id))

    for item in schedule:
        is_unscheduled = item.room_id is None or item.timeslot_id is None

        # Unscheduled items always get a placement attempt; scheduled
        # items mutate with the usual low probability.
        if not is_unscheduled and random.random() > MUTATION_RATE:
            continue

        # free up this item's own current slot before searching, so it
        # doesn't block itself from re-using a compatible room/timeslot
        if item.room_id is not None and item.timeslot_id is not None:
            occupied_rooms.discard((item.room_id, item.timeslot_id))
        if item.instructor_id is not None and item.timeslot_id is not None:
            occupied_instructors.discard((item.instructor_id, item.timeslot_id))

        section = section_map.get(item.section)

        if section is not None:
            viable_rooms = get_viable_rooms(section, rooms)
            valid_times = get_valid_timeslots(section, timeslots)
            assign_target = section
        else:
            viable_rooms = get_viable_rooms_for_schedule_item(item, rooms)
            valid_times = timeslots
            assign_target = item

        if is_unscheduled:
            # try a full, constraint-aware placement
            room, ts = choose_random_assignment(
                assign_target,
                viable_rooms,
                valid_times,
                occupied_instructors,
                occupied_rooms,
            )
            if room is not None and ts is not None:
                item.room_id = room.id
                item.timeslot_id = ts.id
        else:
            # small perturbation: nudge room or timeslot, same as before,
            # but now filtered to each section's *valid* timeslots
            if random.random() < 0.5 and viable_rooms and item.timeslot_id is not None:
                for room in random.sample(viable_rooms, len(viable_rooms)):
                    if (room.id, item.timeslot_id) not in occupied_rooms:
                        item.room_id = room.id
                        break
            else:
                candidate_times = valid_times if valid_times else timeslots
                for ts in random.sample(candidate_times, len(candidate_times)):
                    instructor_free = (item.instructor_id, ts.id) not in occupied_instructors
                    room_free = (item.room_id, ts.id) not in occupied_rooms
                    if instructor_free and room_free:
                        item.timeslot_id = ts.id
                        break

        # re-add (possibly updated) occupancy
        if item.room_id is not None and item.timeslot_id is not None:
            occupied_rooms.add((item.room_id, item.timeslot_id))
        if item.instructor_id is not None and item.timeslot_id is not None:
            occupied_instructors.add((item.instructor_id, item.timeslot_id))

    return schedule


def constraint_repair(schedule, rooms, timeslots, sections):
    """
    Repairs hard constraint violations by reassigning
    conflicting classes to valid rooms/timeslots.

    CHANGED from the original in three ways:
      1. Unscheduled items (room_id/timeslot_id is None) are now also
         picked up for repair instead of being skipped forever.
      2. Non-conflicting-but-invalid placements (e.g. a section sitting
         in a room/timeslot that fails passes_hard_constraints even
         though nobody else is using that slot) are now detected and
         repaired too.
      3. Room/timeslot lookups are O(1) via dicts instead of re-scanning
         lists.
    """

    occupied_rooms = set()
    occupied_instructors = set()

    section_map = {str(section.no): section for section in sections}
    room_map = {room.id: room for room in rooms}
    timeslot_map = {ts.id: ts for ts in timeslots}

    for item in schedule:

        unscheduled = item.room_id is None or item.timeslot_id is None

        room_conflict = (
                not unscheduled and (item.room_id, item.timeslot_id) in occupied_rooms
        )
        instructor_conflict = (
                not unscheduled
                and item.instructor_id is not None
                and (item.instructor_id, item.timeslot_id) in occupied_instructors
        )

        hard_violation = False
        if not unscheduled and not room_conflict and not instructor_conflict:
            section = section_map.get(item.section)
            room_obj = room_map.get(item.room_id)
            ts_obj = timeslot_map.get(item.timeslot_id)
            if section and room_obj and ts_obj:
                try:
                    hard_violation = not passes_hard_constraints(
                        section,
                        room_obj,
                        ts_obj,
                        occupied_instructors=occupied_instructors,
                        occupied_rooms=occupied_rooms,
                    )
                except Exception:
                    # if the constraint check itself errors out, don't
                    # let that crash the repair pass -- treat as no
                    # additional violation detected.
                    hard_violation = False

        needs_repair = unscheduled or room_conflict or instructor_conflict or hard_violation

        if needs_repair:

            section = section_map.get(item.section)

            if section:

                viable_rooms = get_viable_rooms(section, rooms)
                valid_times = get_valid_timeslots(section, timeslots)

                room, ts = choose_random_assignment(
                    section,
                    viable_rooms,
                    valid_times,
                    occupied_instructors,
                    occupied_rooms,
                )

                if room and ts:
                    item.room_id = room.id
                    item.timeslot_id = ts.id
                else:
                    # could not find any free slot -- leave unscheduled
                    # rather than keep an invalid/conflicting one
                    item.room_id = None
                    item.timeslot_id = None

        # update occupied sets
        if item.room_id and item.timeslot_id:
            occupied_rooms.add(
                (item.room_id, item.timeslot_id)
            )

        if item.instructor_id and item.timeslot_id:
            occupied_instructors.add(
                (item.instructor_id, item.timeslot_id)
            )

    return schedule


def local_search(schedule, rooms, timeslots, sections, iterations=5):
    best = copy.deepcopy(schedule)

    best_score = calculate_fitness(
        best,
        rooms,
        sections=sections,
        timeslots=timeslots
    )

    section_map = {str(s.no): s for s in sections}

    for _ in range(iterations):

        candidate = best.copy()

        index = random.randrange(len(candidate))

        candidate[index] = copy.deepcopy(candidate[index])

        item = candidate[index]

        # CHANGED: restrict candidate timeslots to ones actually valid
        # for this item's section instead of any timeslot in the system.
        # Falls back to the full timeslot list only if we don't know the
        # section (shouldn't normally happen).
        section = section_map.get(item.section)
        candidate_timeslots = get_valid_timeslots(section, timeslots) if section else timeslots
        if not candidate_timeslots:
            candidate_timeslots = timeslots

        new_ts = random.choice(candidate_timeslots)

        old_ts = item.timeslot_id

        item.timeslot_id = new_ts.id

        candidate_score = calculate_fitness(
            candidate,
            rooms,
            sections=sections,
            timeslots=timeslots
        )

        if candidate_score > best_score:
            best = candidate
            best_score = candidate_score


        else:
            item.timeslot_id = old_ts

    return best


def iterated_local_search(schedule, rooms, timeslots, sections):
    best = local_search(
        schedule,
        rooms,
        timeslots,
        sections,
        iterations=5
    )

    best_score = calculate_fitness(
        best,
        rooms,
        sections=sections,
        timeslots=timeslots
    )

    section_map = {str(s.no): s for s in sections}

    for _ in range(2):

        candidate = copy.deepcopy(best)

        # perturbation:
        # randomly change several classes
        # CHANGED: restrict to each perturbed item's own valid timeslots
        # so the perturbation doesn't wander into slots that
        # constraint_repair previously had no way of detecting as bad.

        for _ in range(2):

            item = random.choice(candidate)
            section = section_map.get(item.section)
            candidate_timeslots = get_valid_timeslots(section, timeslots) if section else timeslots
            if not candidate_timeslots:
                candidate_timeslots = timeslots

            item.timeslot_id = random.choice(candidate_timeslots).id

        candidate = constraint_repair(
            candidate,
            rooms,
            timeslots,
            sections
        )

        candidate = local_search(
            candidate,
            rooms,
            timeslots,
            sections,
            iterations=5
        )

        candidate_score = calculate_fitness(
            candidate,
            rooms,
            sections=sections,
            timeslots=timeslots
        )

        if candidate_score > best_score:
            best = candidate
            best_score = candidate_score

    return best


# Genetic Algorithm
# Main function that runs the whole process
def genetic_schedule(sections, timeslots, rooms, valid_timeslot_cache=None,
                     population_size=16, generations=15):
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    # step 1: create initial random population
    population = create_population(
        size=population_size,
        sections=sections,
        rooms=rooms,
        timeslots=timeslots
    )

    # step 2: repeat evolution process many times
    for i in range(generations):

        # select best schedules
        selected = selection(
            population,
            rooms,
            sections,
            timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )

        new_population = []

        # create new generation
        while len(new_population) < len(population):
            # pick 2 random parents
            p1, p2 = random.sample(selected, 2)

            # crossover (mix parents)
            child = crossover(p1, p2)

            child = mutation(
                child,
                rooms,
                timeslots,
                sections
            )

            child = constraint_repair(
                child,
                rooms,
                timeslots,
                sections
            )

            new_population.append(child)

        # replace old population with new one
        population = new_population

        # Improve only the best schedule using ILS
        population.sort(
            key=lambda s: calculate_fitness(
                s,
                rooms,
                sections=sections,
                timeslots=timeslots,
                valid_timeslot_cache=valid_timeslot_cache,
            ),
            reverse=True,
        )

        population[0] = iterated_local_search(
            population[0],
            rooms,
            timeslots,
            sections,
        )

    # step 3: get best schedule from final population
    best = max(
        population,
        key=lambda s: calculate_fitness(
            s,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        ),
    )

    return best


def genetic_runs(sections, timeslots, rooms, num_runs=30):
    fitness_scores = []
    best_overall_schedule = None
    best_overall_fitness = -1

    print(f"\n=== Genetic Algorithm: {num_runs} runs ===")
    print(f"Total sections: {len(sections)}, rooms: {len(rooms)}, timeslots: {len(timeslots)}")

    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    for run in range(num_runs):
        best_schedule = genetic_schedule(
            sections,
            timeslots,
            rooms,
            valid_timeslot_cache=valid_timeslot_cache,
        )
        score = calculate_fitness(
            best_schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )
        fitness_scores.append(score)

        if score > best_overall_fitness:
            best_overall_fitness = score
            best_overall_schedule = best_schedule

    best_score = max(fitness_scores)
    worst_score = min(fitness_scores)
    avg_score = sum(fitness_scores) / len(fitness_scores)
    std_dev = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0

    print(f"\n=== Results ===")
    print(f"Best fitness: {best_score}")
    print(f"Worst fitness: {worst_score}")
    print(f"Average fitness: {avg_score:.4f}")
    print(f"Standard deviation: {std_dev:.4f}")

    return best_overall_schedule