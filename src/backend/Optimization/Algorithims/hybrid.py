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
POPULATION_SIZE = 10
GENERATIONS = 10

CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.03

# Tabu Search parameters
TABU_ITERATIONS = 3
TABU_TENURE = 5

# Apply TS to top 10% of solutions
TS_APPLY_RATE = 0.1


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

    if rooms and timeslots:
        return random.choice(rooms), random.choice(timeslots)

    return None, None


# =========================
# CREATE POPULATION
# =========================
def create_population(size, sections, rooms, timeslots):

    population = []

    section_candidates = [
        (
            section,
            get_viable_rooms(section, rooms),
            get_valid_timeslots(section, timeslots),
        )
        for section in sections
    ]

    for i in range(size):

        schedule = []

        occupied_instructors = set()
        occupied_rooms = set()

        for section, viable_rooms, valid_timeslots in section_candidates:

            room, ts = choose_random_assignment(
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
                timeslot_id=ts.id if ts else None,
                section=str(section.no),
            )

            schedule.append(item)

            if room and ts:
                occupied_rooms.add((room.id, ts.id))

            if section.instructor_id and ts:
                occupied_instructors.add(
                    (section.instructor_id, ts.id)
                )

        population.append(schedule)

    return population


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


# =========================
# CROSSOVER
# =========================
def crossover(parent1, parent2):

    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1)

    point = random.randint(1, len(parent1) - 1)

    child = parent1[:point] + parent2[point:]

    return copy.deepcopy(child)


# =========================
# MUTATION
# =========================
def mutation(schedule, rooms, timeslots):

    if len(schedule) == 0:
        return schedule

    if random.random() > MUTATION_RATE:
        return schedule

    selected_item = random.choice(schedule)

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
        idx = random.randint(0, len(current) - 1)

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

    for i in range(GENERATIONS):

        selected = selection(
            population,
            rooms,
            sections,
            timeslots,
            cache,
        )

        new_population = []

        while len(new_population) < POPULATION_SIZE:

            p1, p2 = random.sample(selected, 2)

            child = crossover(p1, p2)

            child = mutation(
                child,
                rooms,
                timeslots,
            )

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
    print(f"Average fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")

    return best_overall_schedule