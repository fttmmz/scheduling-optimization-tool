import random
import copy
from constraints import check_all
from models import ScheduleItem


# Create Population
# This function creates many random schedules (solutions)
def create_population(size, sections, rooms, timeslots):
    population = []

    # repeat to create many schedules
    for i in range(size):
        schedule = []

        # go through each section and assign random room + timeslot
        for section in sections:
            item = ScheduleItem(
                course_id = section.course.id,
                course_name = section.course.name,
                course_type = section.course.type,
                course_dept = section.course.dept,
                capacity = section.capacity,
                instructor_id = section.instructor_id,

                # randomly choose a room for this section
                room_id = random.choice(rooms).id,

                # randomly choose a timeslot for this section
                timeslot_id = random.choice(timeslots).id,

                # convert section number to string for consistency
                section = str(section.no)
            )

            schedule.append(item)

        # add this full schedule to population
        population.append(schedule)

    return population


# Fitness
# This checks how good a schedule is
def fitness(schedule, rooms):
    data = {"rooms": rooms}

    # check if schedule follows all constraints
    if check_all(schedule, data):
        return 10   # good schedule
    else:
        return -5   # bad schedule


# Selection
# This keeps the better half of the population
def selection(population, rooms):
    # sort schedules from best to worst using fitness score
    population.sort(key=lambda s: fitness(s, rooms), reverse=True)

    # keep only top 50%
    return population[:len(population)//2]


# Crossover
# This mixes two parent schedules to create a new one
def crossover(parent1, parent2):

    # if schedule is empty, return empty
    if len(parent1) == 0:
        return []

    # pick a random split point
    point = random.randint(1, len(parent1) - 1)

    # take part from parent1 + rest from parent2
    child = parent1[:point] + parent2[point:]

    # deepcopy to avoid linking objects together
    return copy.deepcopy(child)


# Mutation
# This randomly changes small parts of a schedule
def mutate(schedule, rooms, timeslots):

    # if schedule is empty, do nothing
    if len(schedule) == 0:
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
            child = mutate(child, rooms, timeslots)

            new_population.append(child)

        # replace old population with new one
        population = new_population

    # step 3: get best schedule from final population
    best = max(population, key=lambda s: fitness(s, rooms))

    return best