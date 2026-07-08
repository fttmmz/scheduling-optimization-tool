import random
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

# --- PARAMETERS ---
POPULATION_SIZE = 8
GENERATIONS = 15

CROSSOVER_RATE = 0.75
MUTATION_RATE = 0.25

TABU_ITERATIONS = 45
TABU_TENURE = 12

_ROOMS_BY_ID_KEY = "_rooms_by_id"
_TIMESLOTS_BY_ID_KEY = "_timeslots_by_id"


# --- SCHEDULE OBJECT CLONING ---
def clone_item(item):
    """Manually duplicates individual schedule item fields to bypass deepcopy overhead."""
    return ScheduleItem(
        course_id=item.course_id, course_name=item.course_name,
        course_type=item.course_type, course_dept=item.course_dept,
        capacity=item.capacity, instructor_id=item.instructor_id,
        room_id=item.room_id, timeslot_id=item.timeslot_id, section=item.section,
    )


def clone_schedule(schedule):
    """Loops and creates a quick duplicate copy of an entire schedule list."""
    return [clone_item(item) for item in schedule]


# --- CONSTRAINT OPTION CACHING ---
def build_option_cache(sections, rooms, timeslots):
    """Pre-maps valid rooms and slots once at launch so loops don't re-evaluate rules."""
    cache = {
        _ROOMS_BY_ID_KEY: {r.id: r for r in rooms},
        _TIMESLOTS_BY_ID_KEY: {t.id: t for t in timeslots},
    }
    for idx, section in enumerate(sections):
        cache[idx] = (get_viable_rooms(section, rooms), get_valid_timeslots(section, timeslots))
    return cache


# --- INITIAL POPULATION GENERATION ---
def choose_random_assignment(section, rooms, timeslots, occupied_instructors, occupied_rooms):
    """Helper that checks constraint tracking to find a clear spot during setup."""
    if not rooms or not timeslots:
        return None, None

    for ts in timeslots:
        if section.instructor_id is not None and (section.instructor_id, ts.id) in occupied_instructors:
            continue

        for room in rooms:
            if (room.id, ts.id) in occupied_rooms:
                continue

            if passes_hard_constraints(
                    section, room, ts,
                    occupied_instructors=occupied_instructors,
                    occupied_rooms=occupied_rooms,
            ):
                return room, ts

    if rooms and timeslots:
        return random.choice(rooms), random.choice(timeslots)

    return None, None


def create_population(size, sections, rooms, timeslots):
    """Generates initial schedules using smart candidate lists to start with higher fitness."""
    population = []
    section_candidates = [
        (section, get_viable_rooms(section, rooms), get_valid_timeslots(section, timeslots))
        for section in sections
    ]

    for i in range(size):
        schedule = []
        occupied_instructors = set()
        occupied_rooms = set()

        for section, viable_rooms, valid_timeslots in section_candidates:
            room, timeslot = choose_random_assignment(
                section, viable_rooms, valid_timeslots, occupied_instructors, occupied_rooms
            )
            item = ScheduleItem(
                course_id=section.course.id, course_name=section.course.name,
                course_type=section.course.type, course_dept=section.course.dept,
                capacity=section.capacity, instructor_id=section.instructor_id,
                room_id=room.id if room else None, timeslot_id=timeslot.id if timeslot else None,
                section=str(section.no),
            )

            schedule.append(item)
            if room is not None and timeslot is not None:
                occupied_rooms.add((room.id, timeslot.id))
            if section.instructor_id is not None and timeslot is not None:
                occupied_instructors.add((section.instructor_id, timeslot.id))

        population.append(schedule)

    return population


# --- GENETIC ALGORITHM SELECTION ---
def selection(population, rooms, sections, timeslots, valid_timeslot_cache=None):
    """Evaluates the generation based on fitness and filters out the weaker half."""
    population.sort(
        key=lambda s: calculate_fitness(
            s, rooms, sections=sections, timeslots=timeslots, valid_timeslot_cache=valid_timeslot_cache
        ),
        reverse=True,
    )
    return population[:len(population) // 2]


# --- GENETIC ALGORITHM CROSSOVER ---
def crossover(parent1, parent2):
    """Combines sections from two parents using our fast manual cloning routine."""
    if len(parent1) == 0:
        return []

    if random.random() > CROSSOVER_RATE:
        return clone_schedule(parent1)

    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return clone_schedule(child)


# --- GENETIC ALGORITHM MUTATION ---
def mutation(schedule, rooms, timeslots, sections=None):
    """Alters structural properties of an individual chromosome safely."""
    if len(schedule) == 0:
        return schedule

    if random.random() > MUTATION_RATE:
        return schedule

    selected_item = random.choice(schedule)
    occupied_instructors = set()
    occupied_rooms = set()

    for item in schedule:
        if item != selected_item:
            if item.instructor_id is not None and item.timeslot_id is not None:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))
            if item.room_id is not None and item.timeslot_id is not None:
                occupied_rooms.add((item.room_id, item.timeslot_id))

    if random.random() < 0.5:
        viable_rooms = get_viable_rooms_for_schedule_item(selected_item, rooms)
        if viable_rooms and selected_item.timeslot_id is not None:
            for room in random.sample(viable_rooms, len(viable_rooms)):
                if (room.id, selected_item.timeslot_id) not in occupied_rooms:
                    selected_item.room_id = room.id
                    break
    else:
        for ts in random.sample(timeslots, len(timeslots)):
            instructor_free = (selected_item.instructor_id, ts.id) not in occupied_instructors
            room_free = (selected_item.room_id, ts.id) not in occupied_rooms

            if instructor_free and room_free:
                selected_item.timeslot_id = ts.id
                break

    return schedule


# --- MEMETIC TABU REPAIR SEARCH ---
def tabu_local_search(schedule, rooms, timeslots, sections, valid_timeslot_cache, option_cache):
    """Main local optimization engine utilizing your clean cache structures directly."""
    best_schedule = clone_schedule(schedule)
    best_fitness = calculate_fitness(best_schedule, rooms, sections, timeslots, valid_timeslot_cache)

    current_schedule = clone_schedule(schedule)

    tabu_set = set()
    tabu_history = []

    # Standardized loop index counter variable 'i'
    for i in range(TABU_ITERATIONS):
        occupied_rooms = set()
        occupied_instructors = set()
        troubled_indices = []

        # Map active room and instructor collisions
        for idx, item in enumerate(current_schedule):
            if item.room_id is None or item.timeslot_id is None:
                troubled_indices.append(idx)
            else:
                room_ts = (item.room_id, item.timeslot_id)
                ins_ts = (item.instructor_id, item.timeslot_id) if item.instructor_id else None

                if room_ts in occupied_rooms or (ins_ts and ins_ts in occupied_instructors):
                    troubled_indices.append(idx)

                occupied_rooms.add(room_ts)
                if ins_ts:
                    occupied_instructors.add(ins_ts)

        if not troubled_indices:
            troubled_indices = random.sample(range(len(current_schedule)), min(20, len(current_schedule)))
        else:
            troubled_indices = random.sample(troubled_indices, min(35, len(troubled_indices)))

        best_move = None
        best_move_fitness = -float('inf')

        # Test localized structural neighborhood moves
        for idx in troubled_indices:
            item = current_schedule[idx]
            viable_rooms, valid_timeslots = option_cache[idx]
            if not viable_rooms or not valid_timeslots:
                continue

            # Pure object loops straight from your original design constraints
            for ts in random.sample(valid_timeslots, min(6, len(valid_timeslots))):
                if item.instructor_id and (item.instructor_id, ts.id) in occupied_instructors:
                    continue

                for rm in random.sample(viable_rooms, min(6, len(viable_rooms))):
                    if (rm.id, ts.id) in occupied_rooms:
                        continue

                    old_rm, old_ts = item.room_id, item.timeslot_id
                    item.room_id, item.timeslot_id = rm.id, ts.id

                    move_fitness = calculate_fitness(current_schedule, rooms, sections, timeslots, valid_timeslot_cache)
                    move_key = (idx, rm.id, ts.id)
                    is_tabu = move_key in tabu_set

                    # Apply aspiration rule
                    if move_fitness > best_move_fitness:
                        if not is_tabu or move_fitness > best_fitness:
                            best_move_fitness = move_fitness
                            best_move = (idx, rm.id, ts.id)

                    item.room_id, item.timeslot_id = old_rm, old_ts

        # Update short term tabu memory lists
        if best_move:
            idx, rm_id, ts_id = best_move
            current_schedule[idx].room_id = rm_id
            current_schedule[idx].timeslot_id = ts_id

            move_key = (idx, rm_id, ts_id)
            tabu_set.add(move_key)
            tabu_history.append(move_key)

            if len(tabu_history) > TABU_TENURE:
                oldest_move = tabu_history.pop(0)
                tabu_set.discard(oldest_move)

            if best_move_fitness > best_fitness:
                best_fitness = best_move_fitness
                best_schedule = clone_schedule(current_schedule)
        else:
            break

    return best_schedule


# --- MAIN EVOLUTION ENGINE ---
def genetic_schedule(sections, timeslots, rooms, valid_timeslot_cache=None, option_cache=None):
    """Coordinates seeding, generation sorting, and memetic Tabu search injections."""
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    population = create_population(
        size=POPULATION_SIZE, sections=sections, rooms=rooms, timeslots=timeslots
    )

    # Standardized loop index counter variable 'i'
    for i in range(GENERATIONS):
        selected = selection(
            population, rooms, sections, timeslots, valid_timeslot_cache=valid_timeslot_cache
        )

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutation(child, rooms, timeslots, sections)
            new_population.append(child)

        population = new_population

        # Injection: Run Tabu optimization pass over top solutions
        population.sort(
            key=lambda s: calculate_fitness(s, rooms, sections, timeslots, valid_timeslot_cache),
            reverse=True
        )

        print(f" -> Gen {i + 1}/{GENERATIONS} | Pre-Tabu Best: {calculate_fitness(population[0], rooms, sections, timeslots, valid_timeslot_cache):.4f}")

        population[0] = tabu_local_search(population[0], rooms, timeslots, sections, valid_timeslot_cache, option_cache)
        population[1] = tabu_local_search(population[1], rooms, timeslots, sections, valid_timeslot_cache, option_cache)

        print(f"    Gen {i + 1}/{GENERATIONS} | Post-Tabu Best: {calculate_fitness(population[0], rooms, sections, timeslots, valid_timeslot_cache):.4f}")

    best = max(
        population,
        key=lambda s: calculate_fitness(s, rooms, sections, timeslots, valid_timeslot_cache),
    )
    return best


# --- BENCHMARK TESTING ROUTINE ---
def genetic_runs(sections, timeslots, rooms, num_runs=5):
    """Processes loops over multiple execution runs to verify statistical convergence."""
    fitness_scores = []
    best_overall_schedule = None
    best_overall_fitness = -1

    print(f"\n=== Hybrid GA + Tabu Search Optimization Pipeline ===")
    print(f"Tracking {len(sections)} sections, {len(rooms)} rooms, and {len(timeslots)} timeslots.")

    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    option_cache = build_option_cache(sections, rooms, timeslots)

    # Standardized loop index counter variable 'i'
    for i in range(num_runs):
        print(f"\n--- Starting Evaluation Run {i + 1}/{num_runs} ---")
        best_schedule = genetic_schedule(
            sections, timeslots, rooms,
            valid_timeslot_cache=valid_timeslot_cache, option_cache=option_cache
        )
        score = calculate_fitness(
            best_schedule, rooms, sections=sections, timeslots=timeslots, valid_timeslot_cache=valid_timeslot_cache
        )
        fitness_scores.append(score)

        scheduled = sum(1 for item in best_schedule if item.room_id is not None and item.timeslot_id is not None)
        print(f"Run {i + 1} complete. Assigned: {scheduled}/{len(sections)} sections. Score: {score:.4f}")

        if score > best_overall_fitness:
            best_overall_fitness = score
            best_overall_schedule = best_schedule

    best_score = max(fitness_scores)
    worst_score = min(fitness_scores)
    avg_score = sum(fitness_scores) / len(fitness_scores)
    std_dev = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0

    print(f"\n=== Final Performance Metrics ===")
    print(f"Absolute Best Fitness: {best_score:.4f}")
    print(f"Absolute Worst Fitness: {worst_score:.4f}")
    print(f"Calculated Mean Fitness: {avg_score:.4f}")
    print(f"System Deviation Spread: {std_dev:.4f}")

    return best_overall_schedule