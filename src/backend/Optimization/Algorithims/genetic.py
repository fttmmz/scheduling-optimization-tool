import random
import copy
from backend.Optimization.constraints import (
    check_all,
    check_instructors,
    check_room,
    check_capacity,
    check_room_type,
    check_department,
    check_campus,
    get_valid_timeslots,
    passes_hard_constraints,
    is_room_type_match,
    is_department_match,
    is_capacity_ok,
    is_campus_match,
)
from backend.models.models import ScheduleItem


def get_viable_rooms(section, rooms):
    """Return rooms that satisfy the section's static constraints."""
    return [
        room
        for room in rooms
        if is_room_type_match(section, room)
        and is_department_match(section, room)
        and is_capacity_ok(section, room)
        and is_campus_match(section, room)
    ]


def choose_random_assignment(
    section,
    rooms,
    timeslots,
    occupied_instructors,
    occupied_rooms,
):
    if not rooms or not timeslots:
        return (
            random.choice(rooms) if rooms else None,
            random.choice(timeslots) if timeslots else None,
        )

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

    # Fallback: if no fully valid pair exists, still return a random choice
    return random.choice(rooms), random.choice(timeslots)


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


# Fitness
# This checks how good a schedule is
def fitness(schedule, rooms):
    data = {"rooms": rooms}

    # check if schedule follows all constraints
    if check_all(schedule, data):
        return 10

    score = 0
    if check_instructors(schedule, data):
        score += 2
    if check_room(schedule, data):
        score += 2
    if check_capacity(schedule, data):
        score += 2
    if check_room_type(schedule, data):
        score += 2
    if check_department(schedule, data):
        score += 1
    if check_campus(schedule, data):
        score += 1

    return score


# Selection
# This keeps the better half of the population
def selection(population, rooms):
    # sort schedules from best to worst using fitness score
    population.sort(key=lambda s: fitness(s, rooms), reverse=True)

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
def mutation(schedule, rooms, timeslots):

    # if schedule is empty, do nothing
    if len(schedule) == 0:
        return schedule

    # decide if mutation happens
    if random.random() > MUTATION_RATE:
        return schedule

    # randomly pick one item from schedule
    selected_item = random.choice(schedule)

    # 50% chance to change room or timeslot
    if random.random() < 0.5:
        selected_item.room_id = random.choice(rooms).id
    else:
        selected_item.timeslot_id = random.choice(timeslots).id

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
            child = mutation(child, rooms, timeslots)

            new_population.append(child)

        # replace old population with new one
        population = new_population

    # step 3: get best schedule from final population
    best = max(population, key=lambda s: fitness(s, rooms))

    return best