import random
import copy
import statistics
from backend.Optimization.constraints import (
    OverlapSet,
    get_valid_timeslots,
    passes_hard_constraints,
    get_viable_rooms,
    get_viable_rooms_for_schedule_item,
    instructor_occupancy_ids,
    finalize_schedule,
    item_needs,
    section_needs,
    sibling_key,
    slots_clash,
    time_soft_penalty,
)
from backend.models.models import ScheduleItem
from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
    count_hard_conflicts,
    count_scheduled_sections,
)
from backend.Optimization.benchmark_seeds import seed_all, seed_for_run
from backend.Optimization.instructor_preferences import current_preference_config
from backend.Optimization.provisional_preference_penalty import InstructorPreferenceTracker


def _free_timeslot(section, timeslots, occupied_instructors):
    """First timeslot where this section's instructor is not already busy."""
    instructor_ids = instructor_occupancy_ids(section)
    for ts in timeslots:
        if all((instructor_id, ts.id) not in occupied_instructors
               for instructor_id in instructor_ids):
            return ts
    # Every slot clashes for a teacher: leave it unscheduled (2.0) rather than force a clash (3.0).
    return None


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

    if not needs.room:  # time only
        return None, _free_timeslot(section, timeslots, occupied_instructors)

    if not needs.time:  # room only
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

    # No clash-free pair exists: leave the section unscheduled (2.0) rather than force a clash (3.0),
    # as the other algorithms do. Forcing one put ~5 real clashes into every genetic timetable.
    return None, None


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
        occupied_instructors = OverlapSet()
        occupied_rooms = OverlapSet()

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
            item = ScheduleItem.from_section(
                section,
                room_id=room.id if room else None,
                timeslot_id=timeslot.id if timeslot else None,
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


# Preference-aware ranking is used only for runs with active preferences.
def _preference_rank(schedule, rooms, sections, timeslots, valid_timeslot_cache):
    """Protect feasibility and completeness before comparing soft fitness."""
    return (
        -count_hard_conflicts(schedule, rooms),
        count_scheduled_sections(schedule, sections),
        calculate_fitness(
            schedule, rooms, sections=sections, timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        ),
    )


def preference_guided_mutation(schedule, sections, rooms, timeslots,
                               eligible_indices=None, candidate_limit=24):
    """Move an assigned class to a feasible time with a lower soft cost.

    Only teaching times are changed. Rooms and scheduled-section counts stay
    unchanged, and new placements must pass the existing hard checks.
    """
    config = current_preference_config()
    if not config or not config.active or not schedule or candidate_limit < 1:
        return schedule

    if eligible_indices is None:
        eligible_indices = [
            index for index, section in enumerate(sections)
            if any(str(inst) in config.by_instructor
                   for inst in instructor_occupancy_ids(section))
        ]
    if not eligible_indices:
        return schedule

    # Try a few instructors so an unscheduled class does not waste the move.
    chosen = None
    for index in random.sample(eligible_indices, min(8, len(eligible_indices))):
        if schedule[index].timeslot_id is not None and section_needs(sections[index]).time:
            chosen = index
            break
    if chosen is None:
        return schedule

    section = sections[chosen]
    item = schedule[chosen]
    slot_by_id = {slot.id: slot for slot in timeslots}
    old_slot = slot_by_id.get(item.timeslot_id)
    if old_slot is None:
        return schedule

    room_by_id = {room.id: room for room in rooms}
    room = room_by_id.get(item.room_id)
    if item.room_id is not None and room is None:
        return schedule

    occupied_rooms = OverlapSet()
    occupied_instructors = OverlapSet()
    siblings = set()
    key = sibling_key(section)
    tracker = InstructorPreferenceTracker.from_config(timeslots, config)
    for other_index, other in enumerate(schedule):
        other_slot = slot_by_id.get(other.timeslot_id)
        tracker.add(other, other_slot)
        if other_index == chosen:
            continue
        if other.timeslot_id is not None:
            for instructor_id in instructor_occupancy_ids(other):
                occupied_instructors.add((instructor_id, other.timeslot_id))
            if other.room_id is not None:
                occupied_rooms.add((other.room_id, other.timeslot_id))
            if sibling_key(sections[other_index]) == key:
                siblings.add(other.timeslot_id)

    valid_slots = get_valid_timeslots(section, timeslots)
    if not valid_slots:
        return schedule
    candidates = random.sample(valid_slots, min(candidate_limit, len(valid_slots)))
    best_slot = None
    best_change = 0.0
    old_time_cost = time_soft_penalty(section, old_slot)
    for new_slot in candidates:
        if new_slot.id == old_slot.id:
            continue
        if any(slots_clash(new_slot.id, sibling_slot) for sibling_slot in siblings):
            continue
        if not passes_hard_constraints(
                section, room, new_slot,
                occupied_instructors=occupied_instructors,
                occupied_rooms=occupied_rooms,
        ):
            continue
        change = (
                tracker.move_delta(item, old_slot, new_slot)
                + time_soft_penalty(section, new_slot) - old_time_cost
        )
        if change < best_change - 1e-9:
            best_change = change
            best_slot = new_slot

    if best_slot is not None:
        item.timeslot_id = best_slot.id
    return schedule


def refine_preferences_after_ga(schedule, sections, rooms, timeslots,
                                valid_timeslot_cache=None, passes=2,
                                candidate_limit=96):
    """ON-only local improvement on GA's *finalized* timetable.

    The existing GA often improves only a few assignments during evolution.
    This extra pass considers every scheduled teaching section with saved
    instructor preferences. It changes timeslots only, never rooms or whether
    a section is scheduled. A move must reduce the preference penalty *and*
    the combined preference + existing time penalty.

    The whole timetable is checked before returning; if anything worsens in
    feasibility, completeness, preference penalty, or fitness, restore the
    exact pre-refinement assignments. No work is done when preferences are OFF.
    """
    config = current_preference_config()
    if not (config and config.active and schedule and timeslots
            and passes > 0 and candidate_limit > 0):
        return schedule

    before_hard = count_hard_conflicts(schedule, rooms)
    # A conflicted input can have duplicate bookings that cannot safely be
    # removed from a set one at a time. Leave it to the existing GA instead.
    if before_hard:
        return schedule
    before_scheduled = count_scheduled_sections(schedule, sections)
    before_penalty = config.calculator(schedule, timeslots, config.by_instructor)
    before_fitness = calculate_fitness(
        schedule, rooms, sections=sections, timeslots=timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    original_times = [item.timeslot_id for item in schedule]

    slots_by_id = {ts.id: ts for ts in timeslots}
    rooms_by_id = {room.id: room for room in rooms}
    occupied_rooms = OverlapSet()
    occupied_instructors = OverlapSet()
    tracker = InstructorPreferenceTracker.from_config(timeslots, config)
    siblings = {}
    eligible = []

    for index, (section, item) in enumerate(zip(sections, schedule)):
        siblings.setdefault(sibling_key(section), []).append(index)
        if item.timeslot_id is None:
            continue
        if item.room_id is not None:
            occupied_rooms.add((item.room_id, item.timeslot_id))
        for instructor_id in instructor_occupancy_ids(item):
            occupied_instructors.add((instructor_id, item.timeslot_id))
        tracker.add(item, slots_by_id.get(item.timeslot_id))
        if (section_needs(section).time
                and any(str(inst) in config.by_instructor
                        for inst in instructor_occupancy_ids(section))):
            eligible.append(index)

    # Cache candidate sets per section. No changes to the existing GA's
    # population generation, crossover, mutation, or OFF selection.
    valid_candidates = {
        index: get_valid_timeslots(sections[index], timeslots)
        for index in eligible
    }
    for _ in range(passes):
        changed = False
        for index in eligible:
            section, item = sections[index], schedule[index]
            old_slot = slots_by_id.get(item.timeslot_id)
            if old_slot is None:
                continue
            room = rooms_by_id.get(item.room_id)
            if item.room_id is not None and room is None:
                continue
            instructor_ids = instructor_occupancy_ids(item)
            sibling_slots = [schedule[j].timeslot_id
                             for j in siblings[sibling_key(section)]
                             if j != index and schedule[j].timeslot_id is not None]
            old_time_cost = time_soft_penalty(section, old_slot)

            # The existing assignment must not block its own replacement.
            if item.room_id is not None:
                occupied_rooms.discard((item.room_id, item.timeslot_id))
            for instructor_id in instructor_ids:
                occupied_instructors.discard((instructor_id, item.timeslot_id))

            # Rank by actual incremental instructor preference cost. The
            # strongest candidates are checked for hard feasibility first.
            ranked = []
            for new_slot in valid_candidates[index]:
                if new_slot.id == old_slot.id:
                    continue
                pref_delta = tracker.move_delta(item, old_slot, new_slot)
                if pref_delta >= -1e-9:
                    continue
                combined_delta = (
                        pref_delta + time_soft_penalty(section, new_slot)
                        - old_time_cost
                )
                if combined_delta < -1e-9:
                    ranked.append((combined_delta, pref_delta, new_slot.id, new_slot))
            ranked.sort(key=lambda entry: (entry[0], entry[1], entry[2]))

            chosen = None
            for _, _, _, new_slot in ranked[:candidate_limit]:
                if any(slots_clash(new_slot.id, sibling_slot)
                       for sibling_slot in sibling_slots):
                    continue
                if passes_hard_constraints(
                        section, room, new_slot,
                        occupied_instructors=occupied_instructors,
                        occupied_rooms=occupied_rooms,
                ):
                    chosen = new_slot
                    break

            if chosen is not None:
                tracker.move(item, old_slot, chosen)
                item.timeslot_id = chosen.id
                changed = True
            if item.room_id is not None:
                occupied_rooms.add((item.room_id, item.timeslot_id))
            for instructor_id in instructor_ids:
                occupied_instructors.add((instructor_id, item.timeslot_id))
        if not changed:
            break

    # A final independent safety check also catches constraints not represented
    # in the incremental occupancy checks. Restore the original on any failure.
    after_penalty = config.calculator(schedule, timeslots, config.by_instructor)
    after_fitness = calculate_fitness(
        schedule, rooms, sections=sections, timeslots=timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    if (count_hard_conflicts(schedule, rooms) > before_hard
            or count_scheduled_sections(schedule, sections) != before_scheduled
            or after_penalty > before_penalty + 1e-8
            or after_fitness + 1e-8 < before_fitness):
        for item, old_id in zip(schedule, original_times):
            item.timeslot_id = old_id
    return schedule


# Selection
# This keeps the better half of the population
# and evaluates fitness using section/timeslot-aware penalties.
def selection(population, rooms, sections, timeslots, valid_timeslot_cache=None):
    # Keep the original fitness-only selection when preferences are disabled.
    config = current_preference_config()
    if config is not None and config.active:
        population.sort(
            key=lambda s: _preference_rank(
                s, rooms, sections, timeslots, valid_timeslot_cache,
            ),
            reverse=True,
        )
        return population[:len(population) // 2]

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
    return population[:len(population) // 2]


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
    selected_needs = item_needs(selected_item)
    if not selected_needs.room and not selected_needs.time:
        return schedule

    # check for conflicts before
    occupied_instructors = OverlapSet()
    occupied_rooms = OverlapSet()

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
    if not selected_needs.room:
        mutate_room = False
    elif not selected_needs.time:
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
        size=10,
        sections=sections,
        rooms=rooms,
        timeslots=timeslots
    )

    generations = 20
    config = current_preference_config()
    preferences_active = bool(config and config.active)
    eligible_indices = (
        [index for index, section in enumerate(sections)
         if any(str(inst) in config.by_instructor
                for inst in instructor_occupancy_ids(section))]
        if preferences_active else []
    )

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
            if preferences_active:
                child = preference_guided_mutation(
                    child, sections, rooms, timeslots,
                    eligible_indices=eligible_indices,
                )

            new_population.append(child)

        # replace old population with new one
        population = new_population

    # Prefer feasible, complete schedules when preferences are enabled.
    if preferences_active:
        best = max(
            population,
            key=lambda s: _preference_rank(
                s, rooms, sections, timeslots, valid_timeslot_cache,
            ),
        )
        finalized = finalize_schedule(best, sections, timeslots, valid_timeslot_cache,
                                      rooms=rooms)
        return refine_preferences_after_ga(
            finalized, sections, rooms, timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )

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


def genetic_runs(sections, timeslots, rooms, num_runs=30, seeds=None):
    fitness_scores = []
    best_overall_schedule = None
    best_overall_fitness = -1
    config = current_preference_config()
    preferences_active = bool(config and config.active)
    best_overall_rank = None

    print(f"\n=== Genetic Algorithm: {num_runs} runs ===")
    print(f"Total sections: {len(sections)}, rooms: {len(rooms)}, timeslots: {len(timeslots)}")

    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    for run in range(num_runs):
        # Shared benchmark seeds: run i uses the same seed for every algorithm.
        run_seed = seed_all(seed_for_run(run, seeds))

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
        scheduled = count_scheduled_sections(best_schedule, sections)

        print(f"Run {run + 1:2d} (seed {run_seed}): Fitness = {score:.4f} | Scheduled = {scheduled}")

        if preferences_active:
            rank = (
                -count_hard_conflicts(best_schedule, rooms),
                count_scheduled_sections(best_schedule, sections),
                score,
            )
            if best_overall_rank is None or rank > best_overall_rank:
                best_overall_rank = rank
                best_overall_fitness = score
                best_overall_schedule = best_schedule
        elif score > best_overall_fitness:
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