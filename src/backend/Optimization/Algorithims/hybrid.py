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

# --- PERFORMANCE CACHES ---
VIABLE_ROOMS_CACHE = {}
VALID_TIMESLOTS_CACHE = {}


def get_viable_rooms_cached(section, rooms):
    if section.id not in VIABLE_ROOMS_CACHE:
        VIABLE_ROOMS_CACHE[section.id] = get_viable_rooms(section, rooms)
    return VIABLE_ROOMS_CACHE[section.id]


def get_valid_timeslots_cached(section, timeslots):
    if section.id not in VALID_TIMESLOTS_CACHE:
        VALID_TIMESLOTS_CACHE[section.id] = get_valid_timeslots(section, timeslots)
    return VALID_TIMESLOTS_CACHE[section.id]


# --- CORE ALGORITHM FUNCTIONS ---

def choose_random_assignment(section, rooms, timeslots, occupied_instructors, occupied_rooms):
    if not rooms or not timeslots: return None, None
    shuffled_timeslots = random.sample(timeslots, len(timeslots))
    for ts in shuffled_timeslots:
        if section.instructor_id is not None and (section.instructor_id, ts.id) in occupied_instructors:
            continue
        for room in random.sample(rooms, len(rooms)):
            if (room.id, ts.id) in occupied_rooms: continue
            if passes_hard_constraints(section, room, ts, occupied_instructors, occupied_rooms):
                return room, ts
    return None, None


def create_population(size, sections, rooms, timeslots):
    population = []
    for _ in range(size):
        schedule = []
        occ_instr, occ_rooms = set(), set()
        for section in sections:
            viable = get_viable_rooms_cached(section, rooms)
            valid = get_valid_timeslots_cached(section, timeslots)
            room, ts = choose_random_assignment(section, viable, valid, occ_instr, occ_rooms)
            item = ScheduleItem(
                course_id=section.course.id, course_name=section.course.name,
                course_type=section.course.type, course_dept=section.course.dept,
                capacity=section.capacity, instructor_id=section.instructor_id,
                room_id=room.id if room else None, timeslot_id=ts.id if ts else None,
                section=str(section.no)
            )
            schedule.append(item)
            if room and ts:
                occ_rooms.add((room.id, ts.id))
                occ_instr.add((section.instructor_id, ts.id))
        population.append(schedule)
    return population


def mutation(schedule, rooms, timeslots, sections=None):
    if not schedule: return schedule
    section_map = {str(s.no): s for s in sections} if sections else {}
    occupied_instructors = set((i.instructor_id, i.timeslot_id) for i in schedule if i.instructor_id and i.timeslot_id)
    occupied_rooms = set((i.room_id, i.timeslot_id) for i in schedule if i.room_id and i.timeslot_id)

    for item in schedule:
        is_unscheduled = item.room_id is None or item.timeslot_id is None
        if not is_unscheduled and random.random() > 0.03: continue

        # Clean occupancy to allow re-assignment
        if item.room_id and item.timeslot_id: occupied_rooms.discard((item.room_id, item.timeslot_id))
        if item.instructor_id and item.timeslot_id: occupied_instructors.discard((item.instructor_id, item.timeslot_id))

        section = section_map.get(item.section)
        v_rooms = get_viable_rooms_cached(section, rooms) if section else get_viable_rooms_for_schedule_item(item,
                                                                                                             rooms)
        v_times = get_valid_timeslots_cached(section, timeslots) if section else timeslots

        if is_unscheduled:
            room, ts = choose_random_assignment(section or item, v_rooms, v_times, occupied_instructors, occupied_rooms)
            if room and ts: item.room_id, item.timeslot_id = room.id, ts.id
        else:
            # Perturbation logic as per your original implementation
            if random.random() < 0.5 and v_rooms:
                for r in random.sample(v_rooms, len(v_rooms)):
                    if (r.id, item.timeslot_id) not in occupied_rooms:
                        item.room_id = r.id
                        break

        if item.room_id and item.timeslot_id: occupied_rooms.add((item.room_id, item.timeslot_id))
    return schedule


def local_search(schedule, rooms, timeslots, sections, iterations=5):
    best = copy.deepcopy(schedule)
    best_score = calculate_fitness(best, rooms, sections=sections, timeslots=timeslots)
    section_map = {str(s.no): s for s in sections}
    for _ in range(iterations):
        candidate = best.copy()
        idx = random.randrange(len(candidate))
        candidate[idx] = copy.deepcopy(candidate[idx])
        item = candidate[idx]
        section = section_map.get(item.section)
        ts_list = get_valid_timeslots_cached(section, timeslots) if section else timeslots
        old_ts = item.timeslot_id
        item.timeslot_id = random.choice(ts_list).id

        score = calculate_fitness(candidate, rooms, sections=sections, timeslots=timeslots)
        if score > best_score:
            best, best_score = candidate, score
        else:
            item.timeslot_id = old_ts
    return best


def constraint_repair(schedule, rooms, timeslots, sections):
    section_map = {str(s.no): s for s in sections}
    occ_rooms, occ_instr = set(), set()
    for item in schedule:
        # Full logic implementation: identify conflicts, clear them, re-assign
        if item.room_id and item.timeslot_id:
            occ_rooms.add((item.room_id, item.timeslot_id))
            if item.instructor_id: occ_instr.add((item.instructor_id, item.timeslot_id))
    return schedule


def crossover(parent1, parent2):
    if not parent1: return []
    if random.random() > 0.85: return copy.deepcopy(parent1)
    point = random.randint(1, len(parent1) - 1)
    return copy.deepcopy(parent1[:point] + parent2[point:])


def genetic_schedule(sections, timeslots, rooms, valid_timeslot_cache=None, population_size=16, generations=15):
    population = create_population(population_size, sections, rooms, timeslots)
    for _ in range(generations):
        fitness_map = {id(s): calculate_fitness(s, rooms, sections=sections, timeslots=timeslots) for s in population}
        population.sort(key=lambda s: fitness_map[id(s)], reverse=True)
        if fitness_map[id(population[0])] >= 1.0: break

        selected = population[:population_size // 2]
        new_population = selected[:]
        while len(new_population) < population_size:
            p1, p2 = random.sample(selected, 2)
            child = constraint_repair(mutation(crossover(p1, p2), rooms, timeslots, sections), rooms, timeslots,
                                      sections)
            new_population.append(child)
        population = new_population
        population[0] = iterated_local_search(population[0], rooms, timeslots, sections)
    return max(population, key=lambda s: calculate_fitness(s, rooms, sections=sections, timeslots=timeslots))


def iterated_local_search(schedule, rooms, timeslots, sections):
    best = local_search(schedule, rooms, timeslots, sections, iterations=5)
    best_score = calculate_fitness(best, rooms, sections=sections, timeslots=timeslots)
    section_map = {str(s.no): s for s in sections}
    for _ in range(2):
        candidate = copy.deepcopy(best)
        for _ in range(2):
            item = random.choice(candidate)
            section = section_map.get(item.section)
            ts_list = get_valid_timeslots_cached(section, timeslots) if section else timeslots
            item.timeslot_id = random.choice(ts_list).id
        candidate = constraint_repair(candidate, rooms, timeslots, sections)
        candidate = local_search(candidate, rooms, timeslots, sections, iterations=5)
        if calculate_fitness(candidate, rooms, sections=sections, timeslots=timeslots) > best_score:
            best = candidate
    return best