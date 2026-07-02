import random
import copy

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

# =========================
# PARAMETERS
# =========================
POPULATION_SIZE = 15
GENERATIONS = 15

CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.08

# Tabu Search
TABU_ITERATIONS = 5
TABU_TENURE = 7

# Apply Tabu Search to top 20%
TS_APPLY_RATE = 0.2

# Keep the best schedules each generation
ELITE_SIZE = 2


# =========================
# RANDOM ASSIGNMENT
# =========================
def choose_random_assignment(
        section,
        rooms,
        timeslots,
        occupied_instructors,
        occupied_rooms,
):
    if not rooms or not timeslots:
        return None, None

    for ts in timeslots:

        if (
                section.instructor_id is not None
                and (section.instructor_id, ts.id) in occupied_instructors
        ):
            continue

        for room in rooms:

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

    return None, None


# =========================
# CREATE POPULATION
# =========================
def create_population(size, sections, rooms, timeslots):
    population = []

    section_candidates = [
        (
            idx,
            section,
            get_viable_rooms(section, rooms),
            get_valid_timeslots(section, timeslots),
        )
        for idx, section in enumerate(sections)
    ]

    for i in range(size):

        schedule = [None] * len(sections)

        occupied_instructors = set()
        occupied_rooms = set()

        order = section_candidates[:]
        random.shuffle(order)

        for idx, section, viable_rooms, valid_timeslots in order:

            shuffled_rooms = viable_rooms[:]
            shuffled_timeslots = valid_timeslots[:]
            random.shuffle(shuffled_rooms)
            random.shuffle(shuffled_timeslots)

            room, ts = choose_random_assignment(
                section,
                shuffled_rooms,
                shuffled_timeslots,
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
                timeslot_id=ts.id if ts else None,
                section=str(section.no),
            )

            schedule[idx] = item

            if room and ts:
                occupied_rooms.add((room.id, ts.id))

            if section.instructor_id and ts:
                occupied_instructors.add(
                    (section.instructor_id, ts.id)
                )

        repair_schedule(schedule, sections, rooms, timeslots)

        population.append(schedule)

    return population


# =========================
# CONFLICT REPAIR
# =========================
def repair_schedule(schedule, sections, rooms, timeslots):
    occupied_instructors = set()
    occupied_rooms = set()

    order = list(range(len(schedule)))
    random.shuffle(order)

    for idx in order:

        item = schedule[idx]
        section = sections[idx] if idx < len(sections) else None

        has_conflict = False

        if item.instructor_id is not None and item.timeslot_id is not None:
            if (item.instructor_id, item.timeslot_id) in occupied_instructors:
                has_conflict = True

        if item.room_id is not None and item.timeslot_id is not None:
            if (item.room_id, item.timeslot_id) in occupied_rooms:
                has_conflict = True

        needs_assignment = (
                item.room_id is None
                or item.timeslot_id is None
                or has_conflict
        )

        if not needs_assignment:
            if item.instructor_id is not None and item.timeslot_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )
            if item.room_id is not None and item.timeslot_id is not None:
                occupied_rooms.add((item.room_id, item.timeslot_id))
            continue

        viable_rooms = (
            get_viable_rooms(section, rooms)
            if section is not None
            else get_viable_rooms_for_schedule_item(item, rooms)
        )
        valid_timeslots = (
            get_valid_timeslots(section, timeslots)
            if section is not None
            else timeslots
        )

        if not viable_rooms or not valid_timeslots:
            item.room_id = None
            item.timeslot_id = None
            continue

        shuffled_rooms = viable_rooms[:]
        shuffled_timeslots = valid_timeslots[:]
        random.shuffle(shuffled_rooms)
        random.shuffle(shuffled_timeslots)

        assigned = False

        for ts in shuffled_timeslots:

            if (
                    item.instructor_id is not None
                    and (item.instructor_id, ts.id) in occupied_instructors
            ):
                continue

            for room in shuffled_rooms:

                if (room.id, ts.id) in occupied_rooms:
                    continue

                if section is not None and not passes_hard_constraints(
                        section,
                        room,
                        ts,
                        occupied_instructors=occupied_instructors,
                        occupied_rooms=occupied_rooms,
                ):
                    continue

                item.room_id = room.id
                item.timeslot_id = ts.id
                assigned = True
                break

            if assigned:
                break

        if assigned:
            if item.instructor_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )
            occupied_rooms.add((item.room_id, item.timeslot_id))
        else:
            item.room_id = None
            item.timeslot_id = None

    return schedule


# =========================
# SELECTION
# =========================
def selection(population, rooms, sections, timeslots, cache):
    population.sort(
        key=lambda s: calculate_fitness(
            s,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        ),
        reverse=True,
    )

    return population[: len(population) // 2]


def tournament_selection(
        population,
        rooms,
        sections,
        timeslots,
        cache,
        tournament_size=3,
):
    competitors = random.sample(
        population,
        tournament_size,
    )

    return max(
        competitors,
        key=lambda s: calculate_fitness(
            s,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        ),
    )


# =========================
# CROSSOVER
# =========================
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1)

    child = []

    for i in range(len(parent1)):

        if random.random() < 0.5:
            child.append(copy.deepcopy(parent1[i]))
        else:
            child.append(copy.deepcopy(parent2[i]))

    return child


# =========================
# MUTATION
# =========================
def mutation(schedule, rooms, timeslots):
    if len(schedule) == 0:
        return schedule

    if random.random() > MUTATION_RATE:
        return schedule

    unscheduled = [
        item for item in schedule
        if item.room_id is None or item.timeslot_id is None
    ]

    selected_item = (
        random.choice(unscheduled) if unscheduled else random.choice(schedule)
    )

    occupied_instructors = set()
    occupied_rooms = set()

    for item in schedule:

        if item != selected_item:

            if item.instructor_id and item.timeslot_id:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )

            if item.room_id and item.timeslot_id:
                occupied_rooms.add(
                    (item.room_id, item.timeslot_id)
                )

    if selected_item.room_id is None or selected_item.timeslot_id is None:

        viable_rooms = get_viable_rooms_for_schedule_item(selected_item, rooms)

        if viable_rooms and timeslots:

            shuffled_rooms = random.sample(
                viable_rooms, min(20, len(viable_rooms))
            )
            shuffled_timeslots = random.sample(
                timeslots, min(30, len(timeslots))
            )

            assigned = False

            for ts in shuffled_timeslots:

                if (
                        selected_item.instructor_id is not None
                        and (selected_item.instructor_id, ts.id) in occupied_instructors
                ):
                    continue

                for room in shuffled_rooms:

                    if (room.id, ts.id) in occupied_rooms:
                        continue

                    selected_item.room_id = room.id
                    selected_item.timeslot_id = ts.id
                    assigned = True
                    break

                if assigned:
                    break

        return schedule

    if random.random() < 0.5:

        viable_rooms = get_viable_rooms_for_schedule_item(
            selected_item,
            rooms,
        )

        if viable_rooms and selected_item.timeslot_id:

            for room in random.sample(
                    viable_rooms,
                    min(20, len(viable_rooms)),
            ):

                if (
                        room.id,
                        selected_item.timeslot_id,
                ) not in occupied_rooms:
                    selected_item.room_id = room.id
                    break

    else:

        for ts in random.sample(
                timeslots,
                min(30, len(timeslots)),
        ):

            instructor_free = (
                                  selected_item.instructor_id,
                                  ts.id,
                              ) not in occupied_instructors

            room_free = (
                            selected_item.room_id,
                            ts.id,
                        ) not in occupied_rooms

            if instructor_free and room_free:
                selected_item.timeslot_id = ts.id
                break

    return schedule


# =========================
# TABU SEARCH
# This improves good solutions produced by GA
# =========================
def tabu_search(
        schedule,
        rooms,
        timeslots,
        sections,
        cache,
):
    current = copy.deepcopy(schedule)
    best = copy.deepcopy(schedule)

    best_score = calculate_fitness(
        best,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=cache,
    )

    tabu_list = []

    for i in range(TABU_ITERATIONS):

        # choose one random course
        candidate_indices = random.sample(
            range(len(current)),
            min(20, len(current)),
        )

        idx = random.choice(candidate_indices)

        neighbor = copy.deepcopy(current)
        selected_item = neighbor[idx]

        # build occupied instructor and room sets
        occupied_instructors = set()
        occupied_rooms = set()

        for j, item in enumerate(neighbor):

            if j == idx:
                continue

            if item.instructor_id is not None and item.timeslot_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )

            if item.room_id is not None and item.timeslot_id is not None:
                occupied_rooms.add(
                    (item.room_id, item.timeslot_id)
                )

        # --------------------------------
        # Change timeslot
        # --------------------------------
        if random.random() < 0.5:

            valid_times = get_valid_timeslots(
                sections[idx],
                timeslots,
            )

            for ts in random.sample(
                    valid_times,
                    min(30, len(valid_times)),
            ):

                instructor_free = (
                                      selected_item.instructor_id,
                                      ts.id,
                                  ) not in occupied_instructors

                room_free = (
                                selected_item.room_id,
                                ts.id,
                            ) not in occupied_rooms

                if instructor_free and room_free:
                    selected_item.timeslot_id = ts.id
                    break

            move = (
                "ts",
                idx,
                selected_item.timeslot_id,
            )

        # --------------------------------
        # Change room
        # --------------------------------
        else:

            valid_rooms = get_viable_rooms_for_schedule_item(
                selected_item,
                rooms,
            )

            for room in random.sample(
                    valid_rooms,
                    min(20, len(valid_rooms)),
            ):

                if (
                        room.id,
                        selected_item.timeslot_id,
                ) not in occupied_rooms:
                    selected_item.room_id = room.id
                    break

            move = (
                "room",
                idx,
                selected_item.room_id,
            )

        # skip tabu moves
        if move in tabu_list:
            continue

        repair_schedule(neighbor, sections, rooms, timeslots)

        score = calculate_fitness(
            neighbor,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        # accept better neighbor
        if score > best_score:
            best_score = score
            best = copy.deepcopy(neighbor)
            current = neighbor

        # add move to tabu list
        tabu_list.append(move)

        # maintain tabu tenure
        if len(tabu_list) > TABU_TENURE:
            tabu_list.pop(0)

    return best


# =========================
# HYBRID GA + TABU SEARCH
# =========================
def genetic_schedule(sections, timeslots, rooms, cache=None):
    if cache is None:
        cache = build_timeslot_guideline_cache(
            sections,
            timeslots,
        )

    population = create_population(
        POPULATION_SIZE,
        sections,
        rooms,
        timeslots,
    )

    best_generation_score = 0
    stagnation = 0
    NO_IMPROVEMENT_LIMIT = 5

    for i in range(GENERATIONS):

        selected = selection(
            population,
            rooms,
            sections,
            timeslots,
            cache,
        )

        elites = copy.deepcopy(selected[:ELITE_SIZE])

        new_population = elites

        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_selection(
                selected,
                rooms,
                sections,
                timeslots,
                cache,
            )

            p2 = tournament_selection(
                selected,
                rooms,
                sections,
                timeslots,
                cache,
            )

            child = crossover(p1, p2)

            repair_schedule(child, sections, rooms, timeslots)

            child = mutation(
                child,
                rooms,
                timeslots,
            )

            repair_schedule(child, sections, rooms, timeslots)

            new_population.append(child)

        new_population.sort(
            key=lambda s: calculate_fitness(
                s,
                rooms,
                sections=sections,
                timeslots=timeslots,
                valid_timeslot_cache=cache,
            ),
            reverse=True,
        )

        current_best = calculate_fitness(
            new_population[0],
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        if current_best > best_generation_score:
            best_generation_score = current_best
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= NO_IMPROVEMENT_LIMIT:
            top_k = max(
                1,
                int(len(new_population) * TS_APPLY_RATE)
            )

            for j in range(top_k):
                new_population[j] = tabu_search(
                    new_population[j],
                    rooms,
                    timeslots,
                    sections,
                    cache,
                )

            population = new_population
            break

        # Apply Tabu Search only during the final generation
        if i == GENERATIONS - 1:
            top_k = max(
                1,
                int(len(new_population) * TS_APPLY_RATE)
            )

            for j in range(top_k):
                new_population[j] = tabu_search(
                    new_population[j],
                    rooms,
                    timeslots,
                    sections,
                    cache,
                )

        population = new_population

    best = max(
        population,
        key=lambda s: calculate_fitness(
            s,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        ),
    )

    repair_schedule(best, sections, rooms, timeslots)

    return best


# =========================
# MULTIPLE RUNS
# =========================
def genetic_runs(
        sections,
        timeslots,
        rooms,
        num_runs=30,
):
    fitness_scores = []

    best_overall_schedule = None
    best_overall_fitness = -1

    print(
        f"\n=== Hybrid GA + Tabu Search: {num_runs} runs ==="
    )

    cache = build_timeslot_guideline_cache(
        sections,
        timeslots,
    )

    for i in range(num_runs):

        best_schedule = genetic_schedule(
            sections,
            timeslots,
            rooms,
            cache,
        )

        score = calculate_fitness(
            best_schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        fitness_scores.append(score)

        if score > best_overall_fitness:
            best_overall_fitness = score
            best_overall_schedule = best_schedule

    print("\n=== Results ===")
    print(f"Best fitness: {max(fitness_scores):.4f}")
    print(f"Worst fitness: {min(fitness_scores):.4f}")
    print(f"Average fitness: {sum(fitness_scores) / len(fitness_scores):.4f}")

    return best_overall_schedule