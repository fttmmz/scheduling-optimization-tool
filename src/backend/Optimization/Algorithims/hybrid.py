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


# --- HELPER: SAFE UNWRAP ---
def _safe_unwrap(items):
    """Recursively or linearly flattens tuples like (Object, score) into just Object."""
    return [i[0] if isinstance(i, tuple) and len(i) > 0 else i for i in items]


# --- SCHEDULE OBJECT CLONING ---
def clone_item(item):
    return ScheduleItem(
        course_id=item.course_id, course_name=item.course_name,
        course_type=item.course_type, course_dept=item.course_dept,
        capacity=item.capacity, instructor_id=item.instructor_id,
        room_id=item.room_id, timeslot_id=item.timeslot_id, section=item.section,
    )


def clone_schedule(schedule):
    return [clone_item(item) for item in schedule]


# --- OPTION CACHE ---
def build_option_cache(sections, rooms, timeslots):
    rooms = _safe_unwrap(rooms)
    timeslots = _safe_unwrap(timeslots)

    cache = {
        _ROOMS_BY_ID_KEY: {r.id: r for r in rooms},
        _TIMESLOTS_BY_ID_KEY: {t.id: t for t in timeslots},
    }
    for idx, section in enumerate(sections):
        v_rooms = _safe_unwrap(get_viable_rooms(section, rooms))
        v_slots = _safe_unwrap(get_valid_timeslots(section, timeslots))
        cache[idx] = (v_rooms, v_slots)
    return cache


# --- INITIAL POPULATION GENERATION ---
def create_population(size, sections, rooms, timeslots):
    population = []

    # Pre-clean candidates
    section_candidates = []
    for section in sections:
        v_rooms = _safe_unwrap(get_viable_rooms(section, rooms))
        v_slots = _safe_unwrap(get_valid_timeslots(section, timeslots))
        section_candidates.append((section, v_rooms, v_slots))

    for i in range(size):
        schedule = []
        occupied_instructors = set()
        occupied_rooms = set()

        for section, v_rooms, v_slots in section_candidates:
            # Simple greedy placement
            found_room, found_ts = None, None
            # Filter and sample
            for ts in v_slots:
                if section.instructor_id and (section.instructor_id, ts.id) in occupied_instructors:
                    continue
                for rm in v_rooms:
                    if (rm.id, ts.id) in occupied_rooms:
                        continue
                    if passes_hard_constraints(section, rm, ts, occupied_instructors, occupied_rooms):
                        found_room, found_ts = rm, ts
                        break
                if found_room: break

            # Fallback to random if stuck
            if not found_room and v_rooms and v_slots:
                found_room, found_ts = random.choice(v_rooms), random.choice(v_slots)

            item = ScheduleItem(
                course_id=section.course.id, course_name=section.course.name,
                course_type=section.course.type, course_dept=section.course.dept,
                capacity=section.capacity, instructor_id=section.instructor_id,
                room_id=found_room.id if found_room else None,
                timeslot_id=found_ts.id if found_ts else None,
                section=str(section.no),
            )
            schedule.append(item)
            if found_room and found_ts:
                occupied_rooms.add((found_room.id, found_ts.id))
            if section.instructor_id and found_ts:
                occupied_instructors.add((section.instructor_id, found_ts.id))

        population.append(schedule)
    return population


# --- GENETIC ALGORITHM SELECTION ---
def selection(population, rooms, sections, timeslots, cache):
    population.sort(key=lambda s: calculate_fitness(s, rooms, sections, timeslots, cache), reverse=True)
    return population[:len(population) // 2]


# --- GENETIC ALGORITHM CROSSOVER ---
def crossover(p1, p2):
    if not p1: return []
    if random.random() > CROSSOVER_RATE: return clone_schedule(p1)
    pt = random.randint(1, len(p1) - 1)
    return clone_schedule(p1[:pt] + p2[pt:])

# --- GENETIC ALGORITHM MUTATION ---
def mutation(schedule, rooms, timeslots):
    if not schedule or random.random() > MUTATION_RATE: return schedule
    item = random.choice(schedule)
    # Use global/or cached rooms/timeslots
    rooms = _safe_unwrap(rooms)
    timeslots = _safe_unwrap(timeslots)
    item.room_id = random.choice(rooms).id
    item.timeslot_id = random.choice(timeslots).id
    return schedule


# --- LOCAL TABU SEARCH ---
def tabu_local_search(schedule, rooms, timeslots, sections, cache, option_cache):
    current = clone_schedule(schedule)
    best = clone_schedule(current)
    best_fit = calculate_fitness(best, rooms, sections, timeslots, cache)

    for i in range(TABU_ITERATIONS):
        idx = random.randrange(len(current))
        item = current[idx]

        # SAFE UNWRAP
        viable_rooms, valid_timeslots = option_cache[idx]
        viable_rooms = _safe_unwrap(viable_rooms)
        valid_timeslots = _safe_unwrap(valid_timeslots)

        if not viable_rooms or not valid_timeslots: continue

        # Attempt swap
        old_rm, old_ts = item.room_id, item.timeslot_id
        item.room_id = random.choice(viable_rooms).id
        item.timeslot_id = random.choice(valid_timeslots).id

        fit = calculate_fitness(current, rooms, sections, timeslots, cache)
        if fit > best_fit:
            best_fit = fit
            best = clone_schedule(current)
        else:
            item.room_id, item.timeslot_id = old_rm, old_ts

    return best


# --- MAIN ENGINE ---
def genetic_schedule(sections, timeslots, rooms, cache, option_cache):
    population = create_population(POPULATION_SIZE, sections, rooms, timeslots)

    for gen in range(GENERATIONS):
        selected = selection(population, rooms, sections, timeslots, cache)
        population = [crossover(random.choice(selected), random.choice(selected)) for _ in range(POPULATION_SIZE)]
        population = [mutation(s, rooms, timeslots) for s in population]

        # Apply Tabu
        population[0] = tabu_local_search(population[0], rooms, timeslots, sections, cache, option_cache)

    return max(population, key=lambda s: calculate_fitness(s, rooms, sections, timeslots, cache))


# --- BENCHMARK ---
def genetic_runs(sections, timeslots, rooms, num_runs=5):
    cache = build_timeslot_guideline_cache(sections, timeslots)
    option_cache = build_option_cache(sections, rooms, timeslots)

    results = []
    for i in range(num_runs):
        best = genetic_schedule(sections, timeslots, rooms, cache, option_cache)
        results.append(calculate_fitness(best, rooms, sections, timeslots, cache))

    print(f"Mean Fitness: {statistics.mean(results):.4f}")
    return results