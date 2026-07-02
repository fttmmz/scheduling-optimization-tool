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


def build_viability_cache(sections, rooms, timeslots):
    cache = {}

    for i, section in enumerate(sections):
        cache[i] = {
            "rooms": get_viable_rooms(section, rooms),
            "timeslots": get_valid_timeslots(section, timeslots),
        }

    return cache


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

        viability_cache = build_viability_cache(sections, rooms, timeslots)
        repair_schedule(schedule, sections, rooms, timeslots, viability_cache)

        population.append(schedule)

    return population


# =========================
# CONFLICT REPAIR
# =========================
def repair_schedule(schedule, sections, rooms, timeslots, viability_cache=None):
    occupied_instructors = set()
    occupied_rooms = set()

    order = sorted(
        range(len(schedule)),
        key=lambda i: (
            len(viability_cache[i]["rooms"]) if viability_cache else 0
        )
    )

    for idx in order:

        item = schedule[idx]
        section = sections[idx]

        if item.room_id and item.timeslot_id:

            if (
                    (item.instructor_id, item.timeslot_id) in occupied_instructors
                    or (item.room_id, item.timeslot_id) in occupied_rooms
            ):
                item.room_id = None
                item.timeslot_id = None
            else:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))
                occupied_rooms.add((item.room_id, item.timeslot_id))
                continue

        rooms_viable = viability_cache[idx]["rooms"] if viability_cache else get_viable_rooms(section, rooms)
        times_viable = viability_cache[idx]["timeslots"] if viability_cache else get_valid_timeslots(section, timeslots)

        assigned = False

        for ts in random.sample(times_viable, min(len(times_viable), 20)):

            if section.instructor_id and (section.instructor_id, ts.id) in occupied_instructors:
                continue

            for room in random.sample(rooms_viable, min(len(rooms_viable), 20)):

                if (room.id, ts.id) in occupied_rooms:
                    continue

                if not passes_hard_constraints(
                        section,
                        room,
                        ts,
                        occupied_instructors,
                        occupied_rooms,
                ):
                    continue

                item.room_id = room.id
                item.timeslot_id = ts.id

                occupied_rooms.add((room.id, ts.id))
                if section.instructor_id:
                    occupied_instructors.add((section.instructor_id, ts.id))

                assigned = True
                break

            if assigned:
                break

        if not assigned:
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
def crossover(parent1, parent2, sections, viability_cache):
    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1)

    child = [None] * len(parent1)

    order = sorted(
        range(len(sections)),
        key=lambda i: len(viability_cache[i]["rooms"]) *
                      len(viability_cache[i]["timeslots"])
    )

    for i in order:

        p1_gene = parent1[i]
        p2_gene = parent2[i]

        assigned = False

        # Try parent1 gene
        if (
                p1_gene.room_id is not None and
                p1_gene.timeslot_id is not None
        ):
            if passes_hard_constraints(
                    sections[i],
                    p1_gene.room_id,
                    p1_gene.timeslot_id,
                    set(),
                    set()
            ):
                child[i] = copy.deepcopy(p1_gene)
                assigned = True

        # Try parent2 gene
        if (
                not assigned and
                p2_gene.room_id is not None and
                p2_gene.timeslot_id is not None
        ):
            if passes_hard_constraints(
                    sections[i],
                    p2_gene.room_id,
                    p2_gene.timeslot_id,
                    set(),
                    set()
            ):
                child[i] = copy.deepcopy(p2_gene)
                assigned = True

        # If neither works → fallback
        if not assigned:
            child[i] = copy.deepcopy(p1_gene if random.random() < 0.5 else p2_gene)

    return child


# =========================
# MUTATION
# =========================
def mutation(schedule, rooms, timeslots):
    if len(schedule) == 0:
        return schedule

    if random.random() > MUTATION_RATE:
        return schedule

    selected_item = random.choice(schedule)

    # build conflict map
    occupied_instructors = set()
    occupied_rooms = set()

    for item in schedule:
        if item != selected_item:

            if item.instructor_id and item.timeslot_id:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))

            if item.room_id and item.timeslot_id:
                occupied_rooms.add((item.room_id, item.timeslot_id))

    if selected_item.room_id is None or selected_item.timeslot_id is None:

        viable_rooms = get_viable_rooms_for_schedule_item(selected_item, rooms)

        for room in random.sample(viable_rooms, min(20, len(viable_rooms))):
            for ts in random.sample(timeslots, min(20, len(timeslots))):

                if (
                        (room.id, ts.id) not in occupied_rooms and
                        (selected_item.instructor_id, ts.id) not in occupied_instructors
                ):
                    selected_item.room_id = room.id
                    selected_item.timeslot_id = ts.id
                    return schedule

        return schedule

    if random.random() < 0.5:

        viable_rooms = get_viable_rooms_for_schedule_item(selected_item, rooms)

        for room in random.sample(viable_rooms, min(20, len(viable_rooms))):

            if (room.id, selected_item.timeslot_id) not in occupied_rooms:
                selected_item.room_id = room.id
                break

    else:

        for ts in random.sample(timeslots, min(30, len(timeslots))):

            if (
                    (selected_item.instructor_id, ts.id) not in occupied_instructors and
                    (selected_item.room_id, ts.id) not in occupied_rooms
            ):
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

        repair_schedule(neighbor, sections, rooms, timeslots, cache)

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
        cache = build_timeslot_guideline_cache(sections, timeslots)

    viability_cache = build_viability_cache(sections, rooms, timeslots)

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
            p1 = tournament_selection(selected, rooms, sections, timeslots, cache)
            p2 = tournament_selection(selected, rooms, sections, timeslots, cache)

            child = crossover(p1, p2, sections, viability_cache)
            repair_schedule(child, sections, rooms, timeslots, viability_cache)

            child = mutation(child, rooms, timeslots)
            repair_schedule(child, sections, rooms, timeslots, viability_cache)

            new_population.append(child)

        for s in new_population:
            repair_schedule(s, sections, rooms, timeslots, viability_cache)

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

        if stagnation >= NO_IMPROVEMENT_LIMIT or i == GENERATIONS - 1:

            top_k = max(1, int(len(new_population) * TS_APPLY_RATE))

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

    repair_schedule(best, sections, rooms, timeslots, viability_cache)

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