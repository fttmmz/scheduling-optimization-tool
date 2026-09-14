import random
import copy
import statistics
from backend.Optimization.constraints import (
    get_valid_timeslots,
    passes_hard_constraints,
    get_viable_rooms,
    get_viable_rooms_for_schedule_item,
    instructor_occupancy_ids,
    finalize_schedule,
    needs_for,
    section_needs,
)
from backend.models.models import ScheduleItem
from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
)


def _free_timeslot(section, timeslots, occupied_instructors):
    """First timeslot where this section's instructor is not already busy."""
    instructor_ids = instructor_occupancy_ids(section)
    for ts in timeslots:
        if all((instructor_id, ts.id) not in occupied_instructors
               for instructor_id in instructor_ids):
            return ts
    return random.choice(timeslots) if timeslots else None


def choose_random_assignment(
    section,
    rooms,
    timeslots,
    occupied_instructors,
    occupied_rooms,
    needs=None,
):
    """Pick a (room, timeslot) for *section*, honouring what it actually needs.

    Previously this always demanded BOTH a room and a timeslot, so thesis and
    supervision sections were handed rooms they never use and studio/lab
    sections that need only an hour competed for scarce rooms against real
    lectures. *needs* says which of the two the section actually requires; the
    unneeded half is returned as None.
    """
    if needs is None:
        needs = section_needs(section)

    if not needs.room and not needs.time:
        return None, None

    if not needs.room:                      # time only
        return None, _free_timeslot(section, timeslots, occupied_instructors)

    if not needs.time:                      # room only
        free = [room for room in rooms if (room.id, None) not in occupied_rooms]
        return (random.choice(free) if free else
                (random.choice(rooms) if rooms else None)), None

    # room AND time
    if not rooms or not timeslots:
        return None, None

    section_instructors = instructor_occupancy_ids(section)
    for ts in timeslots:
        if any((instructor_id, ts.id) in occupied_instructors
               for instructor_id in section_instructors):
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

    # Fallback: no clash-free pair exists, so take a random one from the
    # (already soft-ranked) candidates rather than leaving the section out.
    return random.choice(rooms), random.choice(timeslots)


# Create Population
# This function creates many random schedules (solutions)
def create_population(size, sections, rooms, timeslots):
    population = []

    # Precompute what each section needs and its candidates. Sections that need
    # no room get an empty room list, so they never consume one.
    section_candidates = []
    for section in sections:
        needs = section_needs(section)
        section_candidates.append((
            section,
            needs,
            get_viable_rooms(section, rooms) if needs.room else [],
            get_valid_timeslots(section, timeslots) if needs.time else [],
        ))

    # repeat to create many schedules
    for i in range(size):
        schedule = []
        occupied_instructors = set()
        occupied_rooms = set()

        # go through each section and assign what it needs
        for section, needs, viable_rooms, valid_timeslots in section_candidates:
            room, timeslot = choose_random_assignment(
                section,
                viable_rooms,
                valid_timeslots,
                occupied_instructors,
                occupied_rooms,
                needs=needs,
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
                instructor_ids=section.instructor_ids,
                pattern_index=section.pattern_index,
                # level drives classification -- without it every item looks
                # like a plain room+time lecture once detached from its Section.
                level=getattr(section.course, "level", None),
                course_class=getattr(section.course, "course_class", None),
            )

            schedule.append(item)
            if room is not None and timeslot is not None:
                occupied_rooms.add((room.id, timeslot.id))
            if timeslot is not None:
                for instructor_id in instructor_occupancy_ids(section):
                    occupied_instructors.add((instructor_id, timeslot.id))

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

    # Mutating along an axis the section does not use would re-introduce the
    # very assignments classification exists to prevent (a room for a thesis, a
    # timeslot for a supervision placement), so skip those.
    item_needs = needs_for(
        selected_item.course_type, getattr(selected_item, "level", None)
    )
    if not item_needs.room and not item_needs.time:
        return schedule

    # check for conflicts before
    occupied_instructors = set()
    occupied_rooms = set()
    
    for item in schedule:
        if item != selected_item:  # Don't include the item we're about to mutate
            if item.timeslot_id is not None:
                for instructor_id in instructor_occupancy_ids(item):
                    occupied_instructors.add((instructor_id, item.timeslot_id))
            if item.room_id is not None and item.timeslot_id is not None:
                occupied_rooms.add((item.room_id, item.timeslot_id))

    # 50% chance to change room or timeslot -- but only along an axis this
    # section actually uses; if it uses just one, always mutate that one.
    mutate_room = random.random() < 0.5
    if not item_needs.room:
        mutate_room = False
    elif not item_needs.time:
        mutate_room = True

    if mutate_room:
        # Try to assign a new room that doesn't create conflicts
        viable_rooms = get_viable_rooms_for_schedule_item(selected_item, rooms)
        if viable_rooms:
            # A room-only section has no timeslot to clash on, so any viable
            # room will do; otherwise avoid rooms taken at our timeslot.
            for room in random.sample(viable_rooms, len(viable_rooms)):
                if selected_item.timeslot_id is None or \
                        (room.id, selected_item.timeslot_id) not in occupied_rooms:
                    selected_item.room_id = room.id
                    break
    else:
        # Try to assign a new timeslot that doesn't create conflicts
        # Check BOTH instructor AND room are free at new timeslot.
        # The None guards are explicit rather than relying on a None key simply
        # never having been inserted above: that made correctness here depend on
        # a detail of a different loop, and it silently stops holding the moment
        # anyone inserts unconditionally.
        selected_instructors = instructor_occupancy_ids(selected_item)
        for ts in random.sample(timeslots, len(timeslots)):
            instructor_free = all(
                (instructor_id, ts.id) not in occupied_instructors
                for instructor_id in selected_instructors
            )
            room_free = (
                selected_item.room_id is None
                or (selected_item.room_id, ts.id) not in occupied_rooms
            )

            if instructor_free and room_free:
                selected_item.timeslot_id = ts.id
                break

    return schedule


# Genetic Algorithm
# Main function that runs the whole process
def genetic_schedule(sections, timeslots, rooms, valid_timeslot_cache=None):

    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

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
        selected = selection(
            population,
            rooms,
            sections,
            timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )

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
    best = max(
        population,
        key=lambda s: calculate_fitness(
            s,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        ),
    )

    # Siblings (meeting blocks of one section) belong in one room. The
    # placement loops have no cross-block state, so unify afterwards; the
    # pass only ever moves a block into a room that is free at its hour.
    return finalize_schedule(best, sections, timeslots, valid_timeslot_cache,
                             rooms=rooms)


def genetic_runs(sections, timeslots, rooms, num_runs=30):
    fitness_scores = []
    best_overall_schedule = None
    best_overall_fitness = -1
    
    print(f"\n=== Genetic Algorithm: {num_runs} runs ===")
    print(f"Total sections: {len(sections)}, rooms: {len(rooms)}, timeslots: {len(timeslots)}")
    
    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    for run in range(num_runs):
        best_schedule = genetic_schedule(
            sections,
            timeslots,
            rooms,
            valid_timeslot_cache=valid_timeslot_cache,
        )
        score = calculate_fitness(
            best_schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )
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