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
from Optimization.evaluation import calculate_fitness


def choose_random_assignment(
    section,
    rooms,
    timeslots,
    occupied_instructors,
    occupied_rooms,
):
    # If no viable rooms or timeslots, return None for that section
    if not rooms or not timeslots:
        return None, None

    # Try to find a valid assignment that passes all constraints
    for ts in timeslots:
        if section.instructor_id is not None and (section.instructor_id, ts.id) in occupied_instructors:
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

    # Fallback: if no fully valid pair exists, return random valid pair from viable options
    # This ensures we at least try to assign from rooms/timeslots that match static constraints
    if rooms and timeslots:
        return random.choice(rooms), random.choice(timeslots)
    
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
def selection(population, rooms):
    # sort schedules from best to worst using fitness score
    population.sort(key=lambda s: calculate_fitness(s, rooms), reverse=True)

    # keep only top 50%
    return population[:len(population)//2]


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
# This randomly changes small parts of a schedule
MUTATION_RATE = 0.03
def mutation(schedule, rooms, timeslots, sections=None):

    # if schedule is empty, do nothing
    if len(schedule) == 0:
        return schedule

    # decide if mutation happens
    if random.random() > MUTATION_RATE:
        return schedule

    # randomly pick one item from schedule
    selected_item = random.choice(schedule)
    
    # check for conflicts before
    occupied_instructors = set()
    occupied_rooms = set()
    
    for item in schedule:
        if item != selected_item:  # Don't include the item we're about to mutate
            if item.instructor_id is not None and item.timeslot_id is not None:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))
            if item.room_id is not None and item.timeslot_id is not None:
                occupied_rooms.add((item.room_id, item.timeslot_id))

    # 50% chance to change room or timeslot
    if random.random() < 0.5:
        # Try to assign a new room that doesn't create conflicts
        viable_rooms = get_viable_rooms_for_schedule_item(selected_item, rooms)
        if viable_rooms and selected_item.timeslot_id is not None:
            # Find a room that doesn't conflict with the current timeslot
            for room in random.sample(viable_rooms, len(viable_rooms)):
                if (room.id, selected_item.timeslot_id) not in occupied_rooms:
                    selected_item.room_id = room.id
                    break
    else:
        # Try to assign a new timeslot that doesn't create conflicts
        # Check BOTH instructor AND room are free at new timeslot
        for ts in random.sample(timeslots, len(timeslots)):
            instructor_free = (selected_item.instructor_id, ts.id) not in occupied_instructors
            room_free       = (selected_item.room_id, ts.id) not in occupied_rooms
            
            if instructor_free and room_free:
                selected_item.timeslot_id = ts.id
                break

    return schedule


# Genetic Algorithm
# Main function that runs the whole process
def genetic_schedule(sections, timeslots, rooms):

    # step 1: create initial random population
    population = create_population(
        size = 10,
        sections = sections,
        rooms = rooms,
        timeslots = timeslots
    )

    generations = 20

    # step 2: repeat evolution process many times
    for i in range(generations):

        # select best schedules
        selected = selection(population, rooms)

        new_population = []

        # create new generation
        while len(new_population) < len(population):

            # pick 2 random parents
            p1, p2 = random.sample(selected, 2)

            # crossover (mix parents)
            child = crossover(p1, p2)

            # mutation (small random change)
            child = mutation(child, rooms, timeslots, sections)

            new_population.append(child)

        # replace old population with new one
        population = new_population

    # step 3: get best schedule from final population
    best = max(population, key=lambda s: calculate_fitness(s, rooms))

    return best


def genetic_runs(sections, timeslots, rooms, num_runs=30):
    fitness_scores = []
    best_overall_schedule = None
    best_overall_fitness = -1
    
    print(f"\n=== Genetic Algorithm: {num_runs} runs ===")
    print(f"Total sections: {len(sections)}, rooms: {len(rooms)}, timeslots: {len(timeslots)}")
    
    for run in range(num_runs):
        best_schedule = genetic_schedule(sections, timeslots, rooms)
        score = calculate_fitness(best_schedule, rooms)
        fitness_scores.append(score)
        
        # Count how many sections are actually scheduled
        scheduled = sum(1 for item in best_schedule if item.room_id is not None and item.timeslot_id is not None)
        
        
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