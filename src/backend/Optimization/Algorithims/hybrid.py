"""Hybrid Genetic Algorithm and Simulated Annealing for university timetabling.

The two optimization algorithms are:

1. Genetic Algorithm (GA), which explores multiple candidate timetables
   using selection, crossover, mutation, and elitism.
2. Simulated Annealing (SA), which locally improves the best GA timetable
   and can occasionally accept a worse move to escape a local optimum.

A constructive procedure creates the initial GA population. The final
schedule is returned directly after Simulated Annealing.

The scheduling engine calls genetic_schedule() and genetic_runs().
"""


# ====================================================================
# Imports
# ====================================================================

import heapq

import math

import random


from collections import (
    Counter,
    defaultdict,
)

from backend.models.models import ScheduleItem

from backend.Optimization.constraints import (
    NEEDS_NOTHING,
    NEEDS_ROOM_ONLY,
    classify_section,
    get_valid_timeslots,
    get_viable_rooms,
    is_department_fallback_required,
    passes_hard_constraints,
)

from backend.Optimization.evaluation import (
    build_timeslot_guideline_cache,
    calculate_fitness,
    count_room_conflicts,
    count_instructor_conflicts,
    count_campus_conflicts,
    count_room_type_conflicts,
    count_department_conflicts,
    count_capacity_conflicts,
    count_timeslot_guideline_conflicts,
)


# ====================================================================
# Parameters
# ====================================================================

CONSTRUCTION_PAIR_SAMPLE = 60

CONSTRUCTION_TOP_K = 8

STATIC_CACHE_LIMIT = 900000

USE_SOFT_FITNESS = True

_ROOMS_BY_ID_KEY = '_rooms_by_id'

_TIMESLOTS_BY_ID_KEY = '_timeslots_by_id'

_REQUIREMENT_KEY = '_requirement'


# ====================================================================
# Shared Scheduling Helpers
# ====================================================================

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

def _new_schedule(sections):
    return [ScheduleItem(course_id=section.course.id, course_name=section.course.name, course_type=section.course.type, course_dept=section.course.dept, capacity=section.capacity, instructor_id=section.instructor_id, room_id=None, timeslot_id=None, section=section.no) for section in sections]

def _safe_int(value, default=0):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default

def _section_capacity(section):
    return _safe_int(getattr(section, 'capacity', 0), default=0)

def _room_capacity(room):
    return _safe_int(getattr(room, 'capacity', 0), default=0)

def _item_requirement(item):
    course_type = getattr(item, 'course_type', None)

    if course_type in NEEDS_NOTHING:
        return 'NEEDS_NOTHING'

    if course_type in NEEDS_ROOM_ONLY:
        return 'NEEDS_ROOM_ONLY'

    return 'NEEDS_ROOM_AND_TIME'

def _is_scheduled(item):
    requirement = _item_requirement(item)

    if requirement == 'NEEDS_NOTHING':
        return True

    if requirement == 'NEEDS_ROOM_ONLY':
        return item.room_id is not None

    return item.room_id is not None and item.timeslot_id is not None

def count_unscheduled(schedule):
    return sum((1 for item in schedule if not _is_scheduled(item)))

def _scheduled_count(schedule):
    return sum((1 for item in schedule if _is_scheduled(item)))

def _unscheduled_indices(schedule, option_cache=None):
    indices = []

    for idx, item in enumerate(schedule):
        if _is_scheduled(item):
            continue

        if option_cache is not None:
            requirement = _requirement(idx, option_cache)

            if requirement == 'NEEDS_NOTHING':
                continue
        indices.append(idx)

    return indices

def build_option_cache(sections, rooms, timeslots):
    """Precompute the valid rooms, timeslots and requirement for each section."""
    cache = {_ROOMS_BY_ID_KEY: {room.id: room for room in rooms}, _TIMESLOTS_BY_ID_KEY: {timeslot.id: timeslot for timeslot in timeslots}}

    for idx, section in enumerate(sections):
        requirement = classify_section(section)
        viable_rooms = []
        valid_timeslots = []
        department_fallback = False

        if requirement != 'NEEDS_NOTHING':
            viable_rooms = list(get_viable_rooms(section, rooms))
            department_fallback = bool(is_department_fallback_required(section, rooms))

        if requirement == 'NEEDS_ROOM_AND_TIME':
            valid_timeslots = list(get_valid_timeslots(section, timeslots))
        room_ids = frozenset((room.id for room in viable_rooms))
        timeslot_ids = frozenset((ts.id for ts in valid_timeslots))
        cache[idx] = {_REQUIREMENT_KEY: requirement, 'rooms': viable_rooms, 'timeslots': valid_timeslots, 'room_ids': room_ids, 'timeslot_ids': timeslot_ids, 'domain_size': len(viable_rooms) if requirement == 'NEEDS_ROOM_ONLY' else len(viable_rooms) * len(valid_timeslots) if requirement == 'NEEDS_ROOM_AND_TIME' else 0, 'department_fallback': department_fallback}

    return cache

def _requirement(idx, option_cache):
    return option_cache[idx][_REQUIREMENT_KEY]

def _domain_size(idx, option_cache):
    return option_cache[idx].get('domain_size', 0)

def _has_static_domain(idx, option_cache):
    requirement = _requirement(idx, option_cache)

    if requirement == 'NEEDS_NOTHING':
        return True

    if not option_cache[idx]['rooms']:
        return False

    if requirement == 'NEEDS_ROOM_ONLY':
        return True

    return bool(option_cache[idx]['timeslots'])

def _build_occupancy(schedule):
    room_occupancy = defaultdict(set)
    instructor_occupancy = defaultdict(set)

    for idx, item in enumerate(schedule):
        if item.room_id is None or item.timeslot_id is None:
            continue
        room_key = (item.room_id, item.timeslot_id)
        room_occupancy[room_key].add(idx)

        if item.instructor_id is not None:
            instructor_key = (item.instructor_id, item.timeslot_id)
            instructor_occupancy[instructor_key].add(idx)

    return (room_occupancy, instructor_occupancy)

def _remove_assignment(schedule, idx, room_occupancy, instructor_occupancy):
    item = schedule[idx]
    room_id = item.room_id
    timeslot_id = item.timeslot_id
    instructor_id = item.instructor_id

    if room_id is not None and timeslot_id is not None:
        room_key = (room_id, timeslot_id)
        occupants = room_occupancy.get(room_key)

        if occupants is not None:
            occupants.discard(idx)

            if not occupants:
                room_occupancy.pop(room_key, None)

        if instructor_id is not None:
            instructor_key = (instructor_id, timeslot_id)
            instructor_users = instructor_occupancy.get(instructor_key)

            if instructor_users is not None:
                instructor_users.discard(idx)

                if not instructor_users:
                    instructor_occupancy.pop(instructor_key, None)
    item.room_id = None
    item.timeslot_id = None

def _add_assignment(
    schedule,
    idx,
    room,
    timeslot,
    room_occupancy,
    instructor_occupancy,
):
    item = schedule[idx]
    item.room_id = room.id if room is not None else None
    item.timeslot_id = timeslot.id if timeslot is not None else None

    if room is None or timeslot is None:
        return
    room_key = (room.id, timeslot.id)
    room_occupancy[room_key].add(idx)

    if item.instructor_id is not None:
        instructor_key = (item.instructor_id, timeslot.id)
        instructor_occupancy[instructor_key].add(idx)

def _placement_blockers(
    schedule,
    idx,
    room,
    timeslot,
    room_occupancy,
    instructor_occupancy,
):
    if room is None or timeslot is None:
        return set()
    blockers = set(room_occupancy.get((room.id, timeslot.id), set()))
    instructor_id = schedule[idx].instructor_id

    if instructor_id is not None:
        blockers.update(instructor_occupancy.get((instructor_id, timeslot.id), set()))
    blockers.discard(idx)

    return blockers

def _placement_is_free(
    schedule,
    idx,
    room,
    timeslot,
    room_occupancy,
    instructor_occupancy,
):
    return not _placement_blockers(schedule, idx, room, timeslot, room_occupancy, instructor_occupancy)

def _conflict_totals(room_occupancy, instructor_occupancy):
    room_conflicts = sum((max(0, len(indices) - 1) for indices in room_occupancy.values()))
    instructor_conflicts = sum((max(0, len(indices) - 1) for indices in instructor_occupancy.values()))

    return (room_conflicts, instructor_conflicts)

def _hard_metrics(schedule):
    room_occupancy, instructor_occupancy = _build_occupancy(schedule)
    room_conflicts, instructor_conflicts = _conflict_totals(room_occupancy, instructor_occupancy)

    return (count_unscheduled(schedule), room_conflicts, instructor_conflicts)

def _hard_key(schedule):
    unscheduled, room_conflicts, instructor_conflicts = _hard_metrics(schedule)

    return (unscheduled, room_conflicts + instructor_conflicts, room_conflicts, instructor_conflicts)

def _static_assignment_is_valid(
    idx,
    room,
    timeslot,
    sections,
    option_cache,
    static_memo,
):
    requirement = _requirement(idx, option_cache)

    if requirement == 'NEEDS_NOTHING':
        return True

    if room is None:
        return False

    if requirement == 'NEEDS_ROOM_ONLY':
        return True

    if timeslot is None:
        return False
    fallback_allowed = option_cache[idx].get('department_fallback', False)
    key = (idx, room.id, timeslot.id, fallback_allowed)
    cached = static_memo.get(key)

    if cached is not None:
        return cached
    valid = bool(passes_hard_constraints(sections[idx], room, timeslot, occupied_instructors=set(), occupied_rooms=set(), allow_department_fallback=fallback_allowed))

    if len(static_memo) < STATIC_CACHE_LIMIT:
        static_memo[key] = valid

    return valid

def _sample_room_timeslot_pairs(viable_rooms, valid_timeslots, sample_limit):
    if not viable_rooms or not valid_timeslots:
        return []
    total = len(viable_rooms) * len(valid_timeslots)

    if total <= sample_limit:
        pairs = [(room, timeslot) for timeslot in valid_timeslots for room in viable_rooms]
        random.shuffle(pairs)

        return pairs
    pairs = []
    seen = set()
    attempts = 0
    maximum_attempts = sample_limit * 8

    while len(pairs) < sample_limit and attempts < maximum_attempts:
        attempts += 1
        room = random.choice(viable_rooms)
        timeslot = random.choice(valid_timeslots)
        key = (room.id, timeslot.id)

        if key in seen:
            continue
        seen.add(key)
        pairs.append((room, timeslot))

    return pairs

def _build_scarcity_metadata(sections, option_cache):
    room_demand = Counter()
    timeslot_demand = Counter()
    instructor_load = Counter()

    for idx, section in enumerate(sections):
        requirement = _requirement(idx, option_cache)
        instructor_id = getattr(section, 'instructor_id', None)

        if instructor_id is not None:
            instructor_load[instructor_id] += 1

        if requirement == 'NEEDS_NOTHING':
            continue

        for room in option_cache[idx]['rooms']:
            room_demand[room.id] += 1

        if requirement == 'NEEDS_ROOM_AND_TIME':
            for timeslot in option_cache[idx]['timeslots']:
                timeslot_demand[timeslot.id] += 1

    return {'room_demand': room_demand, 'timeslot_demand': timeslot_demand, 'instructor_load': instructor_load}

def _capacity_slack_score(section, room):
    section_capacity = _section_capacity(section)
    room_capacity = _room_capacity(room)

    return max(0, room_capacity - section_capacity)

def _placement_score(
    schedule,
    idx,
    room,
    timeslot,
    remaining_indices,
    sections,
    option_cache,
    scarcity_metadata,
):
    if room is None:
        return (10 ** 12, 10 ** 12, 10 ** 12, random.random())
    fallback_penalty = int(option_cache[idx].get('department_fallback', False))
    room_scarcity = scarcity_metadata['room_demand'].get(room.id, 0)
    timeslot_scarcity = scarcity_metadata['timeslot_demand'].get(timeslot.id, 0) if timeslot is not None else 0
    capacity_slack = _capacity_slack_score(sections[idx], room)

    return (fallback_penalty, room_scarcity + timeslot_scarcity, capacity_slack, random.random())

def _generate_free_candidates(
    schedule,
    idx,
    sections,
    option_cache,
    room_occupancy,
    instructor_occupancy,
    static_memo,
    scarcity_metadata,
    remaining_indices,
    pair_sample_limit,
    top_k,
):
    requirement = _requirement(idx, option_cache)

    if requirement == 'NEEDS_NOTHING':
        return [((0, 0, 0, random.random()), None, None)]
    viable_rooms = option_cache[idx]['rooms']

    if not viable_rooms:
        return []

    if requirement == 'NEEDS_ROOM_ONLY':
        scored = (((int(option_cache[idx].get('department_fallback', False)), scarcity_metadata['room_demand'].get(room.id, 0), _capacity_slack_score(sections[idx], room), random.random()), room, None) for room in viable_rooms)

        return heapq.nsmallest(top_k, scored, key=lambda candidate: candidate[0])
    valid_timeslots = option_cache[idx]['timeslots']

    if not valid_timeslots:
        return []
    pairs = _sample_room_timeslot_pairs(
        viable_rooms,
        valid_timeslots,
        pair_sample_limit,
    )
    best = []
    legal_seen = 0

    for room, timeslot in pairs:
        if not _static_assignment_is_valid(idx, room, timeslot, sections, option_cache, static_memo):
            continue

        if not _placement_is_free(schedule, idx, room, timeslot, room_occupancy, instructor_occupancy):
            continue
        legal_seen += 1
        score = _placement_score(
            schedule,
            idx,
            room,
            timeslot,
            remaining_indices,
            sections,
            option_cache,
            scarcity_metadata,
        )
        entry = (score, room, timeslot)

        if len(best) < top_k:
            best.append(entry)

            if len(best) == top_k:
                best.sort(key=lambda candidate: candidate[0])
        elif score < best[-1][0]:
            best[-1] = entry
            best.sort(key=lambda candidate: candidate[0])

        if len(best) >= top_k and legal_seen >= top_k * 3:
            break
    best.sort(key=lambda candidate: candidate[0])

    return best

def _construction_priority_key(idx, sections, option_cache, scarcity_metadata):
    requirement = _requirement(idx, option_cache)

    if requirement == 'NEEDS_NOTHING':
        requirement_priority = 2
    elif requirement == 'NEEDS_ROOM_ONLY':
        requirement_priority = 1
    else:
        requirement_priority = 0
    domain_size = _domain_size(idx, option_cache)
    instructor_id = getattr(sections[idx], 'instructor_id', None)
    instructor_load = scarcity_metadata['instructor_load'].get(instructor_id, 0)
    capacity = _section_capacity(sections[idx])

    return (requirement_priority, domain_size, -instructor_load, -capacity, random.random())

def _construct_initial_schedule_once(
    sections,
    option_cache,
    scarcity_metadata,
    static_memo,
):
    """Build one conflict-free starting schedule using greedy MRV and LCV ideas."""
    schedule = _new_schedule(sections)
    room_occupancy, instructor_occupancy = _build_occupancy(schedule)
    timed_indices = []

    for idx in range(len(sections)):
        requirement = _requirement(idx, option_cache)

        if requirement == 'NEEDS_ROOM_ONLY':
            viable_rooms = option_cache[idx]['rooms']

            if viable_rooms:
                room = min(viable_rooms, key=lambda candidate: (int(option_cache[idx].get('department_fallback', False)), scarcity_metadata['room_demand'].get(candidate.id, 0), _capacity_slack_score(sections[idx], candidate), random.random()))
                schedule[idx].room_id = room.id
        elif requirement == 'NEEDS_ROOM_AND_TIME':
            timed_indices.append(idx)
    timed_indices.sort(key=lambda idx: _construction_priority_key(idx, sections, option_cache, scarcity_metadata))
    total = len(timed_indices)

    for position, idx in enumerate(timed_indices, 1):
        candidates = _generate_free_candidates(
            schedule,
            idx,
            sections,
            option_cache,
            room_occupancy,
            instructor_occupancy,
            static_memo,
            scarcity_metadata,
            (),
            pair_sample_limit=CONSTRUCTION_PAIR_SAMPLE,
            top_k=CONSTRUCTION_TOP_K,
        )

        if candidates:
            pool = candidates[:min(3, len(candidates))]
            selected = pool[0] if random.random() < 0.82 else random.choice(pool)
            _, room, timeslot = selected
            _add_assignment(
                schedule,
                idx,
                room,
                timeslot,
                room_occupancy,
                instructor_occupancy,
            )

        if position % 750 == 0 or position == total:
            pass  # Progress output removed.
    skipped = [idx for idx in timed_indices if not _is_scheduled(schedule[idx])]
    skipped.sort(key=lambda idx: (_domain_size(idx, option_cache), random.random()))

    for idx in skipped:
        candidates = _generate_free_candidates(
            schedule,
            idx,
            sections,
            option_cache,
            room_occupancy,
            instructor_occupancy,
            static_memo,
            scarcity_metadata,
            (),
            pair_sample_limit=max(120, CONSTRUCTION_PAIR_SAMPLE * 2),
            top_k=CONSTRUCTION_TOP_K,
        )

        if candidates:
            _, room, timeslot = candidates[0]
            _add_assignment(
                schedule,
                idx,
                room,
                timeslot,
                room_occupancy,
                instructor_occupancy,
            )

    return schedule

def _feasible_unscheduled_indices(schedule, option_cache):
    targets = []

    for idx in _unscheduled_indices(schedule, option_cache):
        if _has_static_domain(idx, option_cache):
            targets.append(idx)

    return targets

def _safe_soft_fitness(schedule, sections, rooms, timeslots, cache=None):
    if not USE_SOFT_FITNESS:
        return 0.0

    try:
        return float(calculate_fitness(schedule, rooms, sections=sections, timeslots=timeslots, valid_timeslot_cache=cache))
    except Exception as exc:

        return 0.0

def _complete_solution_key(schedule, sections, rooms, timeslots, cache=None):
    hard_key = _hard_key(schedule)
    soft_fitness = _safe_soft_fitness(schedule, sections, rooms, timeslots, cache)

    return (hard_key[0], hard_key[1], hard_key[2], hard_key[3], -soft_fitness)


# ====================================================================
# Parameters
# ====================================================================

GA_POPULATION_SIZE = 10

GA_GENERATIONS = 12

GA_ELITE_SIZE = 2

GA_TOURNAMENT_SIZE = 3

GA_CROSSOVER_RATE = 0.85

GA_MUTATION_RATE = 0.20

GA_MUTATION_ITEMS = 3

SA_ITERATIONS = 220

SA_INITIAL_TEMPERATURE = 2.5

SA_COOLING_RATE = 0.95

SA_MIN_TEMPERATURE = 0.05

SA_NEIGHBOR_MUTATIONS = 2

SA_INSERTION_PAIR_SAMPLE = 350

SA_MAX_ADDED_ROOM_CONFLICTS = 1

SA_MAX_ADDED_INSTRUCTOR_CONFLICTS = 0

SA_INSERTION_PROBABILITY = 0.85

SA_PROGRESS_STEP = 20


# ====================================================================
# Genetic Algorithm
# ====================================================================

def _ga_energy(schedule, sections, rooms, timeslots, cache):
    """Convert schedule quality into one numeric cost."""
    key = _complete_solution_key(
        schedule,
        sections,
        rooms,
        timeslots,
        cache,
    )

    return (
        key[0] * 1_000_000
        + key[1] * 100_000
        + key[4]
    )

def _ga_tournament_selection(
    population,
    sections,
    rooms,
    timeslots,
    cache,
):
    """Select one parent using tournament selection."""
    sample_size = min(GA_TOURNAMENT_SIZE, len(population))
    competitors = random.sample(population, sample_size)

    return min(
        competitors,
        key=lambda schedule: _complete_solution_key(
            schedule,
            sections,
            rooms,
            timeslots,
            cache,
        ),
    )

def _ga_repair_unscheduled(
    schedule,
    sections,
    option_cache,
    scarcity_metadata,
    static_memo,
):
    """Greedily place any sections left empty after crossover or mutation."""
    room_occupancy, instructor_occupancy = _build_occupancy(schedule)

    remaining = [
        idx
        for idx in _unscheduled_indices(schedule, option_cache)
        if _has_static_domain(idx, option_cache)
    ]

    remaining.sort(
        key=lambda idx: (
            _domain_size(idx, option_cache),
            -_section_capacity(sections[idx]),
            random.random(),
        )
    )

    for idx in remaining:
        candidates = _generate_free_candidates(
            schedule,
            idx,
            sections,
            option_cache,
            room_occupancy,
            instructor_occupancy,
            static_memo,
            scarcity_metadata,
            remaining,
            pair_sample_limit=120,
            top_k=10,
        )

        if not candidates:
            continue

        _, room, timeslot = candidates[0]

        _add_assignment(
            schedule,
            idx,
            room,
            timeslot,
            room_occupancy,
            instructor_occupancy,
        )

    return schedule

def _ga_crossover(
    parent_a,
    parent_b,
    sections,
    option_cache,
    scarcity_metadata,
    static_memo,
):
    """Create one child while preserving only conflict-free assignments."""
    child = _new_schedule(sections)
    room_occupancy, instructor_occupancy = _build_occupancy(child)

    indices = list(range(len(sections)))
    random.shuffle(indices)

    for idx in indices:
        requirement = _requirement(idx, option_cache)

        if requirement == "NEEDS_NOTHING":
            continue

        source_item = (
            parent_a[idx]
            if random.random() < 0.5
            else parent_b[idx]
        )

        if source_item.room_id is None:
            continue

        room = option_cache[_ROOMS_BY_ID_KEY].get(source_item.room_id)

        if requirement == "NEEDS_ROOM_ONLY":
            if room is not None:
                child[idx].room_id = room.id
            continue

        timeslot = option_cache[_TIMESLOTS_BY_ID_KEY].get(
            source_item.timeslot_id
        )

        if room is None or timeslot is None:
            continue

        if not _static_assignment_is_valid(
            idx,
            room,
            timeslot,
            sections,
            option_cache,
            static_memo,
        ):
            continue

        if not _placement_is_free(
            child,
            idx,
            room,
            timeslot,
            room_occupancy,
            instructor_occupancy,
        ):
            continue

        _add_assignment(
            child,
            idx,
            room,
            timeslot,
            room_occupancy,
            instructor_occupancy,
        )

    return _ga_repair_unscheduled(
        child,
        sections,
        option_cache,
        scarcity_metadata,
        static_memo,
    )

def _ga_mutate(
    schedule,
    sections,
    option_cache,
    scarcity_metadata,
    static_memo,
    mutation_items=None,
):
    """Relocate a few sections to create a new nearby timetable."""
    candidate = clone_schedule(schedule)
    room_occupancy, instructor_occupancy = _build_occupancy(candidate)

    timed_indices = [
        idx
        for idx, item in enumerate(candidate)
        if (
            _requirement(idx, option_cache) == "NEEDS_ROOM_AND_TIME"
            and _is_scheduled(item)
        )
    ]

    if not timed_indices:
        return candidate

    number_to_mutate = mutation_items or GA_MUTATION_ITEMS
    selected = random.sample(
        timed_indices,
        min(number_to_mutate, len(timed_indices)),
    )

    for idx in selected:
        old_room_id = candidate[idx].room_id
        old_timeslot_id = candidate[idx].timeslot_id

        _remove_assignment(
            candidate,
            idx,
            room_occupancy,
            instructor_occupancy,
        )

        choices = _generate_free_candidates(
            candidate,
            idx,
            sections,
            option_cache,
            room_occupancy,
            instructor_occupancy,
            static_memo,
            scarcity_metadata,
            set(),
            pair_sample_limit=100,
            top_k=10,
        )

        alternatives = []

        for choice in choices:
            _, room, timeslot = choice

            assignment = (
                None if room is None else room.id,
                None if timeslot is None else timeslot.id,
            )

            if assignment != (old_room_id, old_timeslot_id):
                alternatives.append(choice)

        if alternatives:
            _, room, timeslot = random.choice(
                alternatives[: min(4, len(alternatives))]
            )

            _add_assignment(
                candidate,
                idx,
                room,
                timeslot,
                room_occupancy,
                instructor_occupancy,
            )
        else:
            old_room = option_cache[_ROOMS_BY_ID_KEY].get(old_room_id)
            old_timeslot = option_cache[_TIMESLOTS_BY_ID_KEY].get(
                old_timeslot_id
            )

            _add_assignment(
                candidate,
                idx,
                old_room,
                old_timeslot,
                room_occupancy,
                instructor_occupancy,
            )

    return candidate

def _build_ga_population(
    sections,
    option_cache,
    scarcity_metadata,
    static_memo,
):
    """Create the initial GA population."""
    population = []

    for _ in range(GA_POPULATION_SIZE):
        schedule = _construct_initial_schedule_once(
            sections,
            option_cache,
            scarcity_metadata,
            static_memo,
        )
        population.append(schedule)

    return population

def _run_genetic_algorithm(
    sections,
    rooms,
    timeslots,
    option_cache,
    scarcity_metadata,
    static_memo,
    cache,
):
    """Run the Genetic Algorithm and return its best timetable."""
    population = _build_ga_population(
        sections,
        option_cache,
        scarcity_metadata,
        static_memo,
    )

    for _ in range(GA_GENERATIONS):
        population.sort(
            key=lambda schedule: _complete_solution_key(
                schedule,
                sections,
                rooms,
                timeslots,
                cache,
            )
        )

        next_population = [
            clone_schedule(schedule)
            for schedule in population[:GA_ELITE_SIZE]
        ]

        while len(next_population) < GA_POPULATION_SIZE:
            parent_a = _ga_tournament_selection(
                population,
                sections,
                rooms,
                timeslots,
                cache,
            )
            parent_b = _ga_tournament_selection(
                population,
                sections,
                rooms,
                timeslots,
                cache,
            )

            if random.random() < GA_CROSSOVER_RATE:
                child = _ga_crossover(
                    parent_a,
                    parent_b,
                    sections,
                    option_cache,
                    scarcity_metadata,
                    static_memo,
                )
            else:
                child = clone_schedule(parent_a)

            if random.random() < GA_MUTATION_RATE:
                child = _ga_mutate(
                    child,
                    sections,
                    option_cache,
                    scarcity_metadata,
                    static_memo,
                )

            next_population.append(child)

        population = next_population

    return min(
        population,
        key=lambda schedule: _complete_solution_key(
            schedule,
            sections,
            rooms,
            timeslots,
            cache,
        ),
    )


# ====================================================================
# Simulated Annealing
# ====================================================================

def _sa_insertion_neighbor(
    schedule,
    sections,
    option_cache,
    static_memo,
):
    """Insert one feasible unscheduled section with a small conflict allowance.

    The move may add at most one room conflict. Instructor conflicts are
    forbidden because two classes taught by the same instructor cannot occur
    at the same time.
    """
    targets = _feasible_unscheduled_indices(
        schedule,
        option_cache,
    )

    targets = [
        idx
        for idx in targets
        if _requirement(idx, option_cache) == "NEEDS_ROOM_AND_TIME"
    ]

    if not targets:
        return None

    targets.sort(
        key=lambda idx: (
            _domain_size(idx, option_cache),
            -_section_capacity(sections[idx]),
            random.random(),
        )
    )

    target_pool = targets[: min(30, len(targets))]
    target_idx = random.choice(target_pool)

    room_occupancy, instructor_occupancy = _build_occupancy(schedule)

    pairs = _sample_room_timeslot_pairs(
        option_cache[target_idx]["rooms"],
        option_cache[target_idx]["timeslots"],
        SA_INSERTION_PAIR_SAMPLE,
    )

    ranked = []

    for room, timeslot in pairs:
        if not _static_assignment_is_valid(
            target_idx,
            room,
            timeslot,
            sections,
            option_cache,
            static_memo,
        ):
            continue

        room_users = set(
            room_occupancy.get(
                (room.id, timeslot.id),
                set(),
            )
        )
        room_users.discard(target_idx)

        instructor_users = set()

        instructor_id = schedule[target_idx].instructor_id

        if instructor_id is not None:
            instructor_users = set(
                instructor_occupancy.get(
                    (instructor_id, timeslot.id),
                    set(),
                )
            )
            instructor_users.discard(target_idx)

        added_room_conflicts = len(room_users)
        added_instructor_conflicts = len(instructor_users)

        if added_room_conflicts > SA_MAX_ADDED_ROOM_CONFLICTS:
            continue

        if (
            added_instructor_conflicts
            > SA_MAX_ADDED_INSTRUCTOR_CONFLICTS
        ):
            continue

        ranked.append(
            (
                added_room_conflicts,
                _capacity_slack_score(
                    sections[target_idx],
                    room,
                ),
                random.random(),
                room,
                timeslot,
            )
        )

    if not ranked:
        return None

    ranked.sort(key=lambda value: value[:3])

    # Randomly choose among the best few assignments so SA does not always
    # explore the same neighboring timetable.
    selected_pool = ranked[: min(5, len(ranked))]
    _, _, _, room, timeslot = random.choice(selected_pool)

    candidate = clone_schedule(schedule)
    candidate[target_idx].room_id = room.id
    candidate[target_idx].timeslot_id = timeslot.id

    return candidate

def _create_sa_neighbor(
    schedule,
    sections,
    option_cache,
    scarcity_metadata,
    static_memo,
):
    """Create an SA neighbor using insertion or conflict-free relocation."""
    if random.random() < SA_INSERTION_PROBABILITY:
        candidate = _sa_insertion_neighbor(
            schedule,
            sections,
            option_cache,
            static_memo,
        )

        if candidate is not None:
            return candidate

    return _ga_mutate(
        schedule,
        sections,
        option_cache,
        scarcity_metadata,
        static_memo,
        mutation_items=SA_NEIGHBOR_MUTATIONS,
    )

def _run_simulated_annealing(
    initial_schedule,
    sections,
    rooms,
    timeslots,
    option_cache,
    scarcity_metadata,
    static_memo,
    cache,
):
    """Improve the best GA solution gradually using Simulated Annealing.

    Each SA iteration creates only one nearby timetable. An insertion move
    schedules at most one previously unscheduled section, so the improvement
    happens step by step rather than through one large repair operation.

    A move may introduce at most one room conflict, while instructor
    conflicts remain forbidden. Every candidate is evaluated immediately
    before the temperature is cooled.
    """
    current = clone_schedule(initial_schedule)
    best = clone_schedule(initial_schedule)

    current_energy = _ga_energy(
        current,
        sections,
        rooms,
        timeslots,
        cache,
    )
    best_energy = current_energy
    temperature = SA_INITIAL_TEMPERATURE

    initial_unscheduled = count_unscheduled(best)
    last_reported_unscheduled = initial_unscheduled

    for iteration in range(1, SA_ITERATIONS + 1):
        # One neighbor means at most one new section is inserted per iteration.
        candidate = _create_sa_neighbor(
            current,
            sections,
            option_cache,
            scarcity_metadata,
            static_memo,
        )

        # Re-evaluate immediately after this single move.
        candidate_energy = _ga_energy(
            candidate,
            sections,
            rooms,
            timeslots,
            cache,
        )

        difference = candidate_energy - current_energy

        if difference <= 0:
            accept = True
        elif temperature > SA_MIN_TEMPERATURE:
            exponent = -difference / max(
                temperature,
                SA_MIN_TEMPERATURE,
            )

            probability = (
                0.0
                if exponent < -700
                else math.exp(exponent)
            )

            accept = random.random() < probability
        else:
            accept = False

        if accept:
            current = candidate
            current_energy = candidate_energy

        if current_energy < best_energy:
            best = clone_schedule(current)
            best_energy = current_energy

            best_unscheduled = count_unscheduled(best)

            # Show occasional checkpoints instead of printing every iteration.
            if (
                last_reported_unscheduled - best_unscheduled
                >= SA_PROGRESS_STEP
            ):
                print(
                    f"SA progress at iteration {iteration}: "
                    f"unscheduled={best_unscheduled}, "
                    f"room_conflicts={count_room_conflicts(best)}, "
                    f"instructor_conflicts="
                    f"{count_instructor_conflicts(best)}"
                )
                last_reported_unscheduled = best_unscheduled

        # Cooling happens after every evaluated move.
        temperature = max(
            SA_MIN_TEMPERATURE,
            temperature * SA_COOLING_RATE,
        )

    return best


# ====================================================================
# Public Scheduler Entry Points
# ====================================================================

def genetic_schedule(
    sections,
    timeslots,
    rooms,
    cache=None,
    option_cache=None,
):
    """Run the Genetic Algorithm followed by Simulated Annealing."""
    if not sections:
        return []

    if cache is None:
        cache = build_timeslot_guideline_cache(
            sections,
            timeslots,
        )

    if option_cache is None:
        option_cache = build_option_cache(
            sections,
            rooms,
            timeslots,
        )

    static_memo = {}
    scarcity_metadata = _build_scarcity_metadata(
        sections,
        option_cache,
    )

    best = _run_genetic_algorithm(
        sections,
        rooms,
        timeslots,
        option_cache,
        scarcity_metadata,
        static_memo,
        cache,
    )

    print(
        f"After GA: unscheduled={count_unscheduled(best)}, "
        f"room_conflicts={count_room_conflicts(best)}, "
        f"instructor_conflicts={count_instructor_conflicts(best)}"
    )

    best = _run_simulated_annealing(
        best,
        sections,
        rooms,
        timeslots,
        option_cache,
        scarcity_metadata,
        static_memo,
        cache,
    )

    print(
        f"After SA: unscheduled={count_unscheduled(best)}, "
        f"room_conflicts={count_room_conflicts(best)}, "
        f"instructor_conflicts={count_instructor_conflicts(best)}"
    )

    return best

def genetic_runs(
    sections,
    timeslots,
    rooms,
    num_runs=1,
):
    """Run the GA-SA hybrid several times and keep the best result."""
    try:
        run_count = max(1, int(num_runs or 1))
    except (TypeError, ValueError):
        run_count = 1

    if not sections:
        return []

    cache = build_timeslot_guideline_cache(
        sections,
        timeslots,
    )
    option_cache = build_option_cache(
        sections,
        rooms,
        timeslots,
    )

    best_schedule = None
    best_key = None

    for _ in range(run_count):
        candidate = genetic_schedule(
            sections,
            timeslots,
            rooms,
            cache=cache,
            option_cache=option_cache,
        )

        candidate_key = _complete_solution_key(
            candidate,
            sections,
            rooms,
            timeslots,
            cache,
        )

        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_schedule = clone_schedule(candidate)

    scheduled = _scheduled_count(best_schedule)
    unscheduled = count_unscheduled(best_schedule)

    room_conflicts = count_room_conflicts(best_schedule)
    instructor_conflicts = count_instructor_conflicts(best_schedule)
    campus_conflicts = count_campus_conflicts(
        best_schedule,
        rooms,
    )
    room_type_conflicts = count_room_type_conflicts(
        best_schedule,
        rooms,
    )
    department_conflicts = count_department_conflicts(
        best_schedule,
        rooms,
    )
    capacity_conflicts = count_capacity_conflicts(
        best_schedule,
        rooms,
    )
    timeslot_conflicts = count_timeslot_guideline_conflicts(
        best_schedule,
        sections,
        timeslots,
        valid_timeslot_cache=cache,
    )

    total_conflicts = (
        room_conflicts
        + instructor_conflicts
        + campus_conflicts
        + room_type_conflicts
        + department_conflicts
        + capacity_conflicts
        + timeslot_conflicts
    )

    print("\n========== GA-SA FINAL RESULTS ==========")
    print(f"Scheduled sections: {scheduled}")
    print(f"Unscheduled sections: {unscheduled}")
    print(f"Room conflicts: {room_conflicts}")
    print(f"Instructor conflicts: {instructor_conflicts}")
    print(f"Campus conflicts: {campus_conflicts}")
    print(f"Room type conflicts: {room_type_conflicts}")
    print(f"Department conflicts: {department_conflicts}")
    print(f"Capacity conflicts: {capacity_conflicts}")
    print(f"Timeslot guideline conflicts: {timeslot_conflicts}")
    print(f"Total conflicts: {total_conflicts}")
    print("=========================================")

    return best_schedule