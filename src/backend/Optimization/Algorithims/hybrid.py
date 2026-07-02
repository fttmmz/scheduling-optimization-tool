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
POPULATION_SIZE = 20          # was 10 -> more diversity to select from
GENERATIONS = 20              # was 10 -> more time to converge

CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.02          # now applied PER GENE (see mutation()), not per chromosome

# Elitism: best individuals are copied unchanged into the next generation.
# This guarantees fitness never regresses generation-to-generation.
ELITISM_COUNT = 2

# Random immigrants: brand new random individuals injected each generation
# to fight premature convergence / loss of diversity.
RANDOM_IMMIGRANT_RATE = 0.10

# Tournament selection
TOURNAMENT_SIZE = 3

# Tabu Search parameters
TABU_TENURE = 7

# Tabu search is now run EVERY generation on the elites (memetic algorithm),
# with a light budget, plus a much deeper pass at the very end.
TS_ITERATIONS_PER_GEN = 12
TS_ITERATIONS_FINAL = 60

# Apply per-generation TS to this fraction of the (sorted) population
TS_APPLY_RATE = 0.2


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


def create_single_random_individual(sections, rooms, timeslots):
    """Used for random-immigrant diversity injection."""
    return create_population(1, sections, rooms, timeslots)[0]


# =========================
# FITNESS HELPERS
# =========================
def _score(schedule, rooms, sections, timeslots, cache):
    return calculate_fitness(
        schedule,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=cache,
    )


def _sorted_by_fitness(population, rooms, sections, timeslots, cache):
    scored = [
        (ind, _score(ind, rooms, sections, timeslots, cache))
        for ind in population
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


# =========================
# SELECTION (tournament)
# =========================
def tournament_select(scored_population, k=TOURNAMENT_SIZE):
    """
    scored_population: list of (individual, fitness) tuples.
    Picks k at random, returns the fittest of that sample.
    Keeps selection pressure without fully discarding weaker
    individuals the way pure truncation does.
    """
    sample = random.sample(scored_population, min(k, len(scored_population)))
    sample.sort(key=lambda pair: pair[1], reverse=True)
    return sample[0][0]


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
# MUTATION (per-gene)
# =========================
def _mutate_single_item(selected_item, schedule, rooms, timeslots):

    occupied_instructors = set()
    occupied_rooms = set()

    for item in schedule:

        if item is selected_item:
            continue

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


def mutation(schedule, rooms, timeslots):
    """
    Per-gene mutation: every item in the schedule independently has a
    MUTATION_RATE chance of being reassigned. This gives far more
    exploration than mutating a single random gene per individual,
    while MUTATION_RATE stays low enough not to destroy good schedules.
    """

    if len(schedule) == 0:
        return schedule

    for item in schedule:
        if random.random() < MUTATION_RATE:
            _mutate_single_item(item, schedule, rooms, timeslots)

    return schedule


# =========================
# TABU SEARCH
# Real tabu search: accepts sideways/worse moves to escape local
# optima, uses an attribute-based tabu list (tabu the *reverting*
# move) with an aspiration criterion that overrides tabu status
# whenever a move produces a new best-ever solution.
# =========================
def tabu_search(
    schedule,
    rooms,
    timeslots,
    sections,
    cache,
    iterations=TS_ITERATIONS_PER_GEN,
):

    current = copy.deepcopy(schedule)
    best = copy.deepcopy(schedule)

    current_score = _score(current, rooms, sections, timeslots, cache)
    best_score = current_score

    # move_key -> iteration index at which the tabu status expires
    tabu_list = {}

    for it in range(iterations):

        idx = random.randint(0, len(current) - 1)

        neighbor = copy.deepcopy(current)
        selected_item = neighbor[idx]

        old_ts = selected_item.timeslot_id
        old_room = selected_item.room_id

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

            candidates = (
                random.sample(valid_times, min(30, len(valid_times)))
                if valid_times
                else []
            )

            for ts in candidates:

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

            # tabu the move that would UNDO this change (revert to old_ts)
            move_key = ("ts", idx, old_ts)

        # --------------------------------
        # Change room
        # --------------------------------
        else:

            valid_rooms = get_viable_rooms_for_schedule_item(
                selected_item,
                rooms,
            )

            candidates = (
                random.sample(valid_rooms, min(20, len(valid_rooms)))
                if valid_rooms
                else []
            )

            for room in candidates:

                if (
                    room.id,
                    selected_item.timeslot_id,
                ) not in occupied_rooms:

                    selected_item.room_id = room.id
                    break

            # tabu the move that would UNDO this change (revert to old_room)
            move_key = ("room", idx, old_room)

        # nothing actually changed (no free slot found) -> skip
        if (
            selected_item.timeslot_id == old_ts
            and selected_item.room_id == old_room
        ):
            continue

        score = _score(neighbor, rooms, sections, timeslots, cache)

        is_tabu = tabu_list.get(move_key, -1) > it
        aspiration = score > best_score  # override tabu if new global best

        if is_tabu and not aspiration:
            continue

        # accept the move regardless of whether it improves on `current`
        # (this is what lets tabu search escape local optima, unlike
        # a plain greedy hill-climb)
        current = neighbor
        current_score = score

        tabu_list[move_key] = it + TABU_TENURE

        if score > best_score:
            best_score = score
            best = copy.deepcopy(neighbor)

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

    # track the best individual ever seen, independent of what
    # survives into the final generation's population
    global_best = None
    global_best_score = float("-inf")

    for gen in range(GENERATIONS):

        scored = _sorted_by_fitness(
            population, rooms, sections, timeslots, cache
        )

        if scored[0][1] > global_best_score:
            global_best_score = scored[0][1]
            global_best = copy.deepcopy(scored[0][0])

        is_final_gen = gen == GENERATIONS - 1

        # ---- Memetic step: polish a slice of the fitter individuals
        # every generation (not just the last one), so improvements
        # compound across generations instead of being a one-shot
        # end-of-run polish.
        ts_budget = TS_ITERATIONS_FINAL if is_final_gen else TS_ITERATIONS_PER_GEN
        top_k = max(1, int(len(scored) * TS_APPLY_RATE))

        for j in range(top_k):
            ind, _ = scored[j]
            improved = tabu_search(
                ind, rooms, timeslots, sections, cache, iterations=ts_budget
            )
            improved_score = _score(improved, rooms, sections, timeslots, cache)
            scored[j] = (improved, improved_score)

        scored.sort(key=lambda pair: pair[1], reverse=True)

        if scored[0][1] > global_best_score:
            global_best_score = scored[0][1]
            global_best = copy.deepcopy(scored[0][0])

        if is_final_gen:
            population = [ind for ind, _ in scored]
            break

        # ---- Build next generation ----
        new_population = []

        # Elitism: carry the best individuals over untouched
        elite_count = min(ELITISM_COUNT, len(scored))
        for j in range(elite_count):
            new_population.append(copy.deepcopy(scored[j][0]))

        # Random immigrants: fresh random individuals for diversity
        n_immigrants = int(POPULATION_SIZE * RANDOM_IMMIGRANT_RATE)
        for _ in range(n_immigrants):
            if len(new_population) >= POPULATION_SIZE:
                break
            new_population.append(
                create_single_random_individual(sections, rooms, timeslots)
            )

        # Fill the rest via tournament selection + crossover + mutation
        while len(new_population) < POPULATION_SIZE:

            p1 = tournament_select(scored)
            p2 = tournament_select(scored)

            child = crossover(p1, p2)
            child = mutation(child, rooms, timeslots)

            new_population.append(child)

        population = new_population

    best_final = max(
        population,
        key=lambda s: _score(s, rooms, sections, timeslots, cache),
    )
    best_final_score = _score(best_final, rooms, sections, timeslots, cache)

    # return whichever is truly best: the final population's best,
    # or the best ever observed during the run
    if global_best_score > best_final_score:
        return global_best

    return best_final


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
    best_overall_fitness = float("-inf")

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