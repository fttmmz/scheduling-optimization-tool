import random

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
# FAST CLONING
# =========================
# copy.deepcopy() is generic (reflection/pickle-based) and was one of the
# hotter spots in profiling -- it gets called a lot (every child, every
# tabu neighbor, every elite). ScheduleItem is a plain data holder, so a
# hand-written clone that just copies its fields is much cheaper and
# behaves identically.
def clone_item(item):
    return ScheduleItem(
        course_id=item.course_id,
        course_name=item.course_name,
        course_type=item.course_type,
        course_dept=item.course_dept,
        capacity=item.capacity,
        instructor_id=item.instructor_id,
        room_id=item.room_id,
        timeslot_id=item.timeslot_id,
        section=item.section,
    )


def clone_schedule(schedule):
    return [clone_item(item) for item in schedule]


# =========================
# OPTION CACHE
# =========================
def build_option_cache(sections, rooms, timeslots):
    """
    Precomputes each section's viable rooms and valid timeslots ONCE per
    run, instead of every function (create_population, repair_schedule,
    mutation, tabu_search) recalculating them from scratch every time
    they touch a section. Also gives a cheap "how constrained is this
    section" score (len(viable_rooms) * len(valid_timeslots)) that
    repair/crossover use to place the tightest sections first.
    """

    cache = {}

    for idx, section in enumerate(sections):
        viable_rooms = get_viable_rooms(section, rooms)
        valid_timeslots = get_valid_timeslots(section, timeslots)
        cache[idx] = (viable_rooms, valid_timeslots)

    return cache


def _constraint_score(idx, option_cache):
    """Lower = more constrained = should be placed first."""

    if option_cache and idx in option_cache:
        viable_rooms, valid_timeslots = option_cache[idx]
        if viable_rooms and valid_timeslots:
            return len(viable_rooms) * len(valid_timeslots)
        return 0  # no viable options at all -- treat as maximally constrained

    return 1_000_000  # unknown section (no cache entry) -- place last


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
    """
    Tries every viable room/timeslot combination (in whatever order they're
    given -- callers should shuffle for diversity) and returns the first one
    that passes hard constraints and doesn't collide with what's already
    occupied. Never falls back to a blind pick -- an unplaceable section is
    left (None, None) for repair_schedule()/mutation() to retry later.
    """

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
def create_population(size, sections, rooms, timeslots, option_cache=None):

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    population = []

    section_candidates = [
        (idx, section, option_cache[idx][0], option_cache[idx][1])
        for idx, section in enumerate(sections)
    ]

    for i in range(size):

        schedule = [None] * len(sections)

        occupied_instructors = set()
        occupied_rooms = set()

        # Most-constrained-first, with a random tiebreak so individuals
        # in the population still differ from each other.
        order = sorted(
            section_candidates,
            key=lambda c: (
                len(c[2]) * len(c[3]) if c[2] and c[3] else 0,
                random.random(),
            ),
        )

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

        repair_schedule(schedule, sections, rooms, timeslots, option_cache=option_cache)

        population.append(schedule)

    return population


# =========================
# CONFLICT REPAIR
# =========================
def repair_schedule(schedule, sections, rooms, timeslots, option_cache=None):
    """
    Detects room/instructor double-bookings and unassigned sections, and
    re-places them (most-constrained-first) without colliding with
    anything already accepted in this pass. Unplaceable items are left
    unscheduled (None, None) rather than kept in a conflicting state.
    """

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    occupied_instructors = set()
    occupied_rooms = set()

    order = sorted(
        range(len(schedule)),
        key=lambda i: (_constraint_score(i, option_cache), random.random()),
    )

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

        if idx in option_cache:
            viable_rooms, valid_timeslots = option_cache[idx]
        elif section is not None:
            viable_rooms = get_viable_rooms(section, rooms)
            valid_timeslots = get_valid_timeslots(section, timeslots)
        else:
            viable_rooms = get_viable_rooms_for_schedule_item(item, rooms)
            valid_timeslots = timeslots

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
    """Kept for external/backwards-compat callers. The hot path inside
    genetic_schedule() no longer calls this -- it does the scoring itself
    once per generation and reuses the results (see genetic_schedule)."""

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


def tournament_selection(population, fitness_lookup, tournament_size=3):
    """
    fitness_lookup maps id(schedule) -> fitness, precomputed once per
    generation by the caller. This used to call calculate_fitness on
    every competitor on every call (2 calls/child x tournament_size=3 ==
    6 fitness evaluations per child, ~78/generation) -- now it's a dict
    lookup, so selection itself is effectively free.
    """

    competitors = random.sample(
        population,
        min(tournament_size, len(population)),
    )

    return max(competitors, key=lambda s: fitness_lookup[id(s)])


# =========================
# CROSSOVER
# =========================
def crossover(parent1, parent2, option_cache=None):
    """
    Conflict-aware crossover. Builds the child most-constrained-section-
    first, picks a preferred parent 50/50 for diversity, and falls back
    to the other parent's slot if the preferred one already collides
    with something placed earlier in the child. Only unscheduled if
    neither parent's slot is free.
    """

    if random.random() > CROSSOVER_RATE:
        return clone_schedule(parent1)

    child = [None] * len(parent1)

    occupied_instructors = set()
    occupied_rooms = set()

    order = sorted(
        range(len(parent1)),
        key=lambda i: (_constraint_score(i, option_cache), random.random()),
    )

    for idx in order:

        item1 = parent1[idx]
        item2 = parent2[idx]

        if random.random() < 0.5:
            preferred, other = item1, item2
        else:
            preferred, other = item2, item1

        chosen = None

        for candidate in (preferred, other):

            if candidate.room_id is None or candidate.timeslot_id is None:
                continue

            instructor_conflict = (
                candidate.instructor_id is not None
                and (candidate.instructor_id, candidate.timeslot_id)
                in occupied_instructors
            )

            room_conflict = (
                candidate.room_id,
                candidate.timeslot_id,
            ) in occupied_rooms

            if not instructor_conflict and not room_conflict:
                chosen = candidate
                break

        if chosen is not None:

            new_item = clone_item(chosen)
            child[idx] = new_item

            if new_item.instructor_id is not None:
                occupied_instructors.add(
                    (new_item.instructor_id, new_item.timeslot_id)
                )
            occupied_rooms.add((new_item.room_id, new_item.timeslot_id))

        else:
            new_item = clone_item(preferred)
            new_item.room_id = None
            new_item.timeslot_id = None
            child[idx] = new_item

    return child


# =========================
# MUTATION
# =========================
def mutation(schedule, rooms, timeslots, option_cache=None):
    """
    Returns (schedule, changed) so the caller can skip the follow-up
    repair_schedule() pass when mutation didn't actually touch anything
    (MUTATION_RATE = 0.08, so that's ~92% of calls -- skipping repair on
    those is one of the bigger wins here, since repair_schedule sorts
    and walks the whole schedule).
    """

    if len(schedule) == 0:
        return schedule, False

    if random.random() > MUTATION_RATE:
        return schedule, False

    indexed = list(enumerate(schedule))

    unscheduled = [
        (i, item) for i, item in indexed
        if item.room_id is None or item.timeslot_id is None
    ]

    idx, selected_item = (
        random.choice(unscheduled) if unscheduled else random.choice(indexed)
    )

    occupied_instructors = set()
    occupied_rooms = set()

    for i, item in indexed:

        if i == idx:
            continue

        if item.instructor_id and item.timeslot_id:
            occupied_instructors.add(
                (item.instructor_id, item.timeslot_id)
            )

        if item.room_id and item.timeslot_id:
            occupied_rooms.add(
                (item.room_id, item.timeslot_id)
            )

    if option_cache and idx in option_cache:
        cached_rooms, cached_timeslots = option_cache[idx]
    else:
        cached_rooms = get_viable_rooms_for_schedule_item(selected_item, rooms)
        cached_timeslots = timeslots

    # Case 1: the selected item is unscheduled -> try to fully place it.
    if selected_item.room_id is None or selected_item.timeslot_id is None:

        if cached_rooms and cached_timeslots:

            shuffled_rooms = random.sample(
                cached_rooms, min(20, len(cached_rooms))
            )
            shuffled_timeslots = random.sample(
                cached_timeslots, min(30, len(cached_timeslots))
            )

            for ts in shuffled_timeslots:

                if (
                    selected_item.instructor_id is not None
                    and (selected_item.instructor_id, ts.id) in occupied_instructors
                ):
                    continue

                placed = False

                for room in shuffled_rooms:

                    if (room.id, ts.id) in occupied_rooms:
                        continue

                    selected_item.room_id = room.id
                    selected_item.timeslot_id = ts.id
                    placed = True
                    break

                if placed:
                    return schedule, True

        return schedule, False

    # Case 2: normal mutation of an already-scheduled item.
    changed = False

    if random.random() < 0.5:

        if cached_rooms and selected_item.timeslot_id:

            for room in random.sample(
                cached_rooms,
                min(20, len(cached_rooms)),
            ):

                if (
                    room.id,
                    selected_item.timeslot_id,
                ) not in occupied_rooms:

                    if room.id != selected_item.room_id:
                        changed = True
                    selected_item.room_id = room.id
                    break

    else:

        for ts in random.sample(
            cached_timeslots,
            min(30, len(cached_timeslots)),
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
                if ts.id != selected_item.timeslot_id:
                    changed = True
                selected_item.timeslot_id = ts.id
                break

    return schedule, changed


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
    option_cache=None,
):
    """Returns (best_schedule, best_score) -- callers no longer need to
    re-run calculate_fitness on the result."""

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    current = clone_schedule(schedule)
    best = clone_schedule(schedule)

    best_score = calculate_fitness(
        best,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=cache,
    )

    tabu_list = []

    for i in range(TABU_ITERATIONS):

        candidate_indices = random.sample(
            range(len(current)),
            min(20, len(current)),
        )

        idx = random.choice(candidate_indices)

        neighbor = clone_schedule(current)
        selected_item = neighbor[idx]

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

        if random.random() < 0.5:

            if idx in option_cache:
                valid_times = option_cache[idx][1]
            else:
                valid_times = get_valid_timeslots(sections[idx], timeslots)

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

        else:

            if idx in option_cache:
                valid_rooms = option_cache[idx][0]
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

        if move in tabu_list:
            continue

        repair_schedule(neighbor, sections, rooms, timeslots, option_cache=option_cache)

        score = calculate_fitness(
            neighbor,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        if score > best_score:

            best_score = score
            best = clone_schedule(neighbor)
            current = neighbor

        tabu_list.append(move)

        if len(tabu_list) > TABU_TENURE:
            tabu_list.pop(0)

    return best, best_score

# =========================
# HYBRID GA + TABU SEARCH
# =========================
def genetic_schedule(sections, timeslots, rooms, cache=None, option_cache=None):

    if cache is None:
        cache = build_timeslot_guideline_cache(
            sections,
            timeslots,
        )

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    population = create_population(
        POPULATION_SIZE,
        sections,
        rooms,
        timeslots,
        option_cache=option_cache,
    )

    best_generation_score = 0
    stagnation = 0
    NO_IMPROVEMENT_LIMIT = 5

    # Carried forward so the final "best" pick at the end of this function
    # doesn't need to re-run calculate_fitness over the whole population.
    last_scored = None

    for i in range(GENERATIONS):

        # Score the current population exactly once per generation and
        # reuse it for both the elite cut and every tournament selection
        # (previously tournament_selection re-scored 3 competitors on
        # every single call).
        scored_population = [
            (
                s,
                calculate_fitness(
                    s,
                    rooms,
                    sections=sections,
                    timeslots=timeslots,
                    valid_timeslot_cache=cache,
                ),
            )
            for s in population
        ]
        scored_population.sort(key=lambda x: x[1], reverse=True)

        half = len(scored_population) // 2
        selected_scored = scored_population[:half]
        selected = [s for s, _ in selected_scored]
        fitness_lookup = {id(s): f for s, f in scored_population}

        elites = [clone_schedule(s) for s, _ in selected_scored[:ELITE_SIZE]]

        new_population = elites

        while len(new_population) < POPULATION_SIZE:

            p1 = tournament_selection(selected, fitness_lookup)
            p2 = tournament_selection(selected, fitness_lookup)

            child = crossover(p1, p2, option_cache=option_cache)

            # Crossover is conflict-aware but repair once more as a
            # cheap safety net for any leftover unscheduled items.
            repair_schedule(child, sections, rooms, timeslots, option_cache=option_cache)

            child, mutated = mutation(
                child,
                rooms,
                timeslots,
                option_cache=option_cache,
            )

            # Only re-repair if mutation actually changed something --
            # it fires ~8% of the time, so this skips the scan+sort in
            # repair_schedule for the other ~92% of children.
            if mutated:
                repair_schedule(child, sections, rooms, timeslots, option_cache=option_cache)

            new_population.append(child)

        scored_new = [
            (
                s,
                calculate_fitness(
                    s,
                    rooms,
                    sections=sections,
                    timeslots=timeslots,
                    valid_timeslot_cache=cache,
                ),
            )
            for s in new_population
        ]
        scored_new.sort(key=lambda x: x[1], reverse=True)
        new_population = [s for s, _ in scored_new]

        current_best = scored_new[0][1]

        if current_best > best_generation_score:
            best_generation_score = current_best
            stagnation = 0
        else:
            stagnation += 1

        apply_tabu = stagnation >= NO_IMPROVEMENT_LIMIT or i == GENERATIONS - 1

        if apply_tabu:

            top_k = max(
                1,
                int(len(new_population) * TS_APPLY_RATE)
            )

            for j in range(top_k):

                improved_schedule, improved_score = tabu_search(
                    new_population[j],
                    rooms,
                    timeslots,
                    sections,
                    cache,
                    option_cache=option_cache,
                )

                new_population[j] = improved_schedule
                scored_new[j] = (improved_schedule, improved_score)

            # tabu search may have changed the ranking among the top_k
            scored_new.sort(key=lambda x: x[1], reverse=True)
            new_population = [s for s, _ in scored_new]

        population = new_population
        last_scored = scored_new

        if stagnation >= NO_IMPROVEMENT_LIMIT:
            break

    best, _ = last_scored[0]

    # Final safety net: guarantee the schedule we hand back is conflict-free
    # (and give any still-unscheduled sections one last placement attempt).
    repair_schedule(best, sections, rooms, timeslots, option_cache=option_cache)

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

    # sections/rooms/timeslots don't change across runs, so build this
    # once for the whole batch instead of once per run.
    option_cache = build_option_cache(sections, rooms, timeslots)

    for i in range(num_runs):

        best_schedule = genetic_schedule(
            sections,
            timeslots,
            rooms,
            cache,
            option_cache=option_cache,
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