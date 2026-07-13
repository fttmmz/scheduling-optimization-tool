import random
import copy
import statistics
from backend.Optimization.constraints import (
    get_valid_timeslots, passes_hard_constraints,
    get_viable_rooms, get_viable_rooms_for_schedule_item
)
from backend.models.models import ScheduleItem
from backend.Optimization.evaluation import calculate_fitness, build_timeslot_guideline_cache


# ==========================================
# 1. ALNS REPAIR (The "Fixer" Engine)
# ==========================================
def alns_repair(schedule, sections, rooms, timeslots):
    """Destroys 10% of the schedule and repairs it greedily."""
    num_to_destroy = max(1, len(schedule) // 10)
    indices_to_clear = random.sample(range(len(schedule)), num_to_destroy)

    for idx in indices_to_clear:
        schedule[idx].room_id = None
        schedule[idx].timeslot_id = None

    for i in indices_to_clear:
        sec = sections[i]
        v_rooms = get_viable_rooms(sec, rooms)
        v_ts = get_valid_timeslots(sec, timeslots)
        random.shuffle(v_ts)

        for ts in v_ts:
            random.shuffle(v_rooms)
            for room in v_rooms:
                if passes_hard_constraints(sec, room, ts, schedule=schedule):
                    schedule[i].room_id = room.id
                    schedule[i].timeslot_id = ts.id
                    break
            if schedule[i].room_id: break
    return schedule


# ==========================================
# 2. GA OPERATORS (Your existing logic)
# ==========================================
def crossover(parent1, parent2):
    if random.random() > 0.85: return copy.deepcopy(parent1)
    point = random.randint(1, len(parent1) - 1)
    return copy.deepcopy(parent1[:point] + parent2[point:])


def mutation(schedule, rooms, timeslots, sections):
    if random.random() > 0.03: return schedule
    idx = random.randrange(len(schedule))
    schedule[idx].room_id = None
    schedule[idx].timeslot_id = None
    return schedule


# ==========================================
# 3. MAIN LOOP (Integrating everything)
# ==========================================
def genetic_schedule(sections, timeslots, rooms, valid_timeslot_cache=None):
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    # Initial population
    population = create_population(10, sections, rooms, timeslots)

    for gen in range(20):
        # Selection
        selected = selection(population, rooms, sections, timeslots, valid_timeslot_cache)
        new_population = []

        while len(new_population) < len(population):
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutation(child, rooms, timeslots, sections)

            # Integrate ALNS Repair here
            if random.random() < 0.3:
                child = alns_repair(child, sections, rooms, timeslots)

            new_population.append(child)
        population = new_population

    return max(population, key=lambda s: calculate_fitness(
        s, rooms, sections=sections, timeslots=timeslots, valid_timeslot_cache=valid_timeslot_cache))

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
    return population[:len(population)//2]
