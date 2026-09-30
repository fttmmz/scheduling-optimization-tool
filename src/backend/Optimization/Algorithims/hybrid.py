"""Hybrid Genetic Algorithm + Tabu Search scheduler.

This implementation follows a simple two-stage hybrid design:
    1. Genetic Algorithm (GA) performs global exploration using a population,
       selection, crossover, mutation, and elitism.
    2. Tabu Search (TS) starts from the best GA schedule. It first repairs
       unscheduled items and hard conflicts, then improves soft constraints
       while preserving feasibility.

References
----------
- Al-Betar, M. A., Khader, A. T., & Zaman, M. (2016).
  "A Utilization-based Genetic Algorithm for Solving the University
  Timetabling Problem (UGA)."
  Alexandria Engineering Journal, 55(2), 1395-1409.
  https://doi.org/10.1016/j.aej.2016.02.017
  -> Genetic Algorithm applied to university timetabling using population,
     crossover, mutation, fitness evaluation, and scheduling constraints.

- Jat, S. N., & Yang, S. (2012).
  "Genetic Algorithm with Search Bank Strategies for University Course
  Timetabling Problem."
  Procedia Engineering, 38, 253-263.
  https://doi.org/10.1016/j.proeng.2012.06.033
  -> Supports combining Genetic Algorithm and Tabu Search for university
     course timetabling.

- Lu, Z., & Hao, J.-K. (2010).
  "Adaptive Tabu Search for Course Timetabling."
  European Journal of Operational Research, 200(1), 235-244.
  https://doi.org/10.1016/j.ejor.2008.12.007
  -> Tabu Search for reducing soft constraint violations while maintaining
     satisfaction of hard constraints.
"""

import copy
import random
import statistics
import time
from collections import Counter, defaultdict

from backend.models.models import ScheduleItem
from backend.Optimization.constraints import (
    NEEDS_NOTHING,
    NEEDS_ROOM_AND_TIME,
    NEEDS_ROOM_ONLY,
    NEEDS_TIME_ONLY,
    classify_section,
    finalize_schedule,
    get_valid_timeslots,
    get_viable_rooms,
    is_room_allowed,
    instructor_occupancy_ids,
    occupancy_key,
    room_soft_penalty,
    sibling_key,
    time_soft_penalty,
)
from backend.Optimization.benchmark_seeds import seed_all, seed_for_run
from backend.Optimization.evaluation import (
    build_timeslot_guideline_cache,
    calculate_fitness,
    count_hard_conflicts,
    count_instructor_conflicts,
    count_room_conflicts,
    count_scheduled_sections,
    count_sibling_conflicts,
    count_timeslot_guideline_conflicts,
    soft_violation_counts,
    total_soft_penalty,
    total_time_penalty,
)


# ============================================================
# PARAMETERS
# ============================================================

POPULATION_SIZE = 10
GENERATIONS = 12
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.05
ELITE_COUNT = 2

ROOM_LIMIT = 30
PAIR_SAMPLE = 80

TABU_ITERATIONS = 500
TABU_TENURE = 12
TABU_SAMPLE = 70

TS_SOFT_PASSES = 4
TS_SOFT_ITEMS = 900
TS_SOFT_PAIR_SAMPLE = 120
TS_SWAP_PASSES = 4
TS_SWAP_ITEMS = 1200
TS_SWAP_PARTNERS = 80


# ============================================================
# BASIC HELPERS
# ============================================================

def is_scheduled(item, requirement):
    if requirement == NEEDS_NOTHING:
        return True
    if requirement == NEEDS_ROOM_ONLY:
        return item.room_id is not None
    if requirement == NEEDS_TIME_ONLY:
        return item.timeslot_id is not None
    return item.room_id is not None and item.timeslot_id is not None


def make_item(section):
    """Create one empty ScheduleItem from a Section."""
    return ScheduleItem.from_section(section)


def build_cache(sections, rooms, timeslots):
    """Precompute what each section needs and its possible rooms/timeslots."""
    requirements = []
    candidates = []

    for section in sections:
        requirement = classify_section(section)
        requirements.append(requirement)

        if requirement in (NEEDS_ROOM_AND_TIME, NEEDS_ROOM_ONLY):
            room_options = list(get_viable_rooms(section, rooms, limit=ROOM_LIMIT))
        else:
            room_options = []

        if requirement in (NEEDS_ROOM_AND_TIME, NEEDS_TIME_ONLY):
            time_options = list(get_valid_timeslots(section, timeslots))
            time_options.sort(key=lambda ts: time_soft_penalty(section, ts))
        else:
            time_options = []

        candidates.append({
            "rooms": room_options,
            "timeslots": time_options,
        })

    return requirements, candidates


def sample_choices(requirement, rooms, timeslots, limit):
    """Return a small set of room/time choices instead of checking everything."""
    if requirement == NEEDS_NOTHING:
        return [(None, None)]

    if requirement == NEEDS_ROOM_ONLY:
        return [(room, None) for room in rooms[:limit]]

    if requirement == NEEDS_TIME_ONLY:
        chosen = timeslots if len(timeslots) <= limit else timeslots[:limit]
        return [(None, ts) for ts in chosen]

    if not rooms or not timeslots:
        return []

    total = len(rooms) * len(timeslots)
    if total <= limit:
        return [(room, ts) for room in rooms for ts in timeslots]

    choices = []
    seen = set()

    # Always try some of the best ranked rooms and times first.
    for room in rooms[:8]:
        for ts in timeslots[:8]:
            key = (room.id, ts.id)
            if key not in seen:
                seen.add(key)
                choices.append((room, ts))
                if len(choices) >= limit:
                    return choices

    while len(choices) < limit:
        room = random.choice(rooms[: min(len(rooms), ROOM_LIMIT)])
        ts = random.choice(timeslots)
        key = (room.id, ts.id)
        if key not in seen:
            seen.add(key)
            choices.append((room, ts))

    return choices


# ============================================================
# OCCUPANCY
# ============================================================

class Occupancy:
    """Stores which room, instructor and sibling is using each timeslot."""

    def __init__(self, schedule):
        self.schedule = schedule
        self.rooms = defaultdict(list)
        self.instructors = defaultdict(list)
        self.siblings = defaultdict(list)

        for i in range(len(schedule)):
            self.add(i)

    @staticmethod
    def _remove(mapping, key, index):
        values = mapping.get(key)
        if not values:
            return
        if index in values:
            values.remove(index)
        if not values:
            mapping.pop(key, None)

    def add(self, index):
        item = self.schedule[index]
        ts = item.timeslot_id
        if ts is None:
            return

        if item.room_id is not None:
            self.rooms[(item.room_id, ts)].append(index)

        for instructor_id in instructor_occupancy_ids(item):
            self.instructors[(instructor_id, ts)].append(index)

        self.siblings[(sibling_key(item), ts)].append(index)

    def remove(self, index):
        item = self.schedule[index]
        ts = item.timeslot_id
        if ts is None:
            return

        if item.room_id is not None:
            self._remove(self.rooms, (item.room_id, ts), index)

        for instructor_id in instructor_occupancy_ids(item):
            self._remove(self.instructors, (instructor_id, ts), index)

        self._remove(self.siblings, (sibling_key(item), ts), index)

    def occupant_count(self, indices):
        """Count room/instructor occupants using the same combined-class idea as evaluation.py."""
        groups = defaultdict(list)

        for index in indices:
            item = self.schedule[index]
            groups[occupancy_key(item.course_id, item.section)].append(item)

        count = 0
        for key, members in groups.items():
            if key[0] != "combined":
                count += 1
            else:
                per_course = Counter(item.course_id for item in members)
                count += max(per_course.values())

        return count

    def hard_cost(self, index, room_id, timeslot_id):
        """How many new hard conflicts would this placement create?"""
        if timeslot_id is None:
            return 0

        item = self.schedule[index]
        cost = 0

        if room_id is not None:
            current = self.rooms.get((room_id, timeslot_id), [])
            before = max(0, self.occupant_count(current) - 1)
            after = max(0, self.occupant_count(current + [index]) - 1)
            cost += after - before

        for instructor_id in instructor_occupancy_ids(item):
            current = self.instructors.get((instructor_id, timeslot_id), [])
            before = max(0, self.occupant_count(current) - 1)
            after = max(0, self.occupant_count(current + [index]) - 1)
            cost += after - before

        # Meeting blocks belonging to the same section cannot use the same timeslot.
        siblings = self.siblings.get((sibling_key(item), timeslot_id), [])
        cost += len(siblings)

        return cost

    def problem_indices(self, requirements):
        problems = []

        for i, item in enumerate(self.schedule):
            if not is_scheduled(item, requirements[i]):
                problems.append(i)
                continue

            ts = item.timeslot_id
            if ts is None:
                continue

            bad = False

            if item.room_id is not None:
                room_users = self.rooms.get((item.room_id, ts), [])
                if self.occupant_count(room_users) > 1:
                    bad = True

            if not bad:
                for instructor_id in instructor_occupancy_ids(item):
                    users = self.instructors.get((instructor_id, ts), [])
                    if self.occupant_count(users) > 1:
                        bad = True
                        break

            if not bad and len(self.siblings.get((sibling_key(item), ts), [])) > 1:
                bad = True

            if bad:
                problems.append(i)

        return problems


# ============================================================
# GENETIC ALGORITHM
# Holland (1975); Goldberg (1989)
# ============================================================

def create_population(size, sections, requirements, candidates):
    """Create the initial GA population."""
    population = []

    # Schedule the most constrained sections first.
    order = list(range(len(sections)))
    order.sort(
        key=lambda i: (
            max(1, len(candidates[i]["rooms"]))
            * max(1, len(candidates[i]["timeslots"])),
            -getattr(sections[i], "capacity", 0),
        )
    )

    for _ in range(size):
        schedule = [make_item(section) for section in sections]
        occ = Occupancy(schedule)

        for i in order:
            requirement = requirements[i]

            if requirement == NEEDS_NOTHING:
                continue

            choices = sample_choices(
                requirement,
                candidates[i]["rooms"],
                candidates[i]["timeslots"],
                PAIR_SAMPLE,
            )

            best = None
            for room, ts in choices:
                room_id = room.id if room is not None else None
                ts_id = ts.id if ts is not None else None

                hard = occ.hard_cost(i, room_id, ts_id)
                soft = 0
                if room is not None:
                    soft += room_soft_penalty(sections[i], room)
                if ts is not None:
                    soft += time_soft_penalty(sections[i], ts)

                value = (hard, soft, random.random(), room_id, ts_id)
                if best is None or value < best:
                    best = value

            if best is not None:
                _, _, _, room_id, ts_id = best
                schedule[i].room_id = room_id
                schedule[i].timeslot_id = ts_id
                occ.add(i)

        population.append(schedule)

    return population


# GA selection: keep better schedules more likely to reproduce.
def selection(population, sections, rooms, timeslots, cache):
    """Keep the best half. Hard feasibility is checked before fitness."""
    def key(schedule):
        unscheduled = sum(
            1
            for item, section in zip(schedule, sections)
            if not is_scheduled(item, classify_section(section))
        )
        hard = count_hard_conflicts(schedule)
        fitness = calculate_fitness(
            schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )
        return (unscheduled, hard, -fitness)

    population.sort(key=key)
    return population[: max(2, len(population) // 2)]


# GA crossover: combine assignments from two parent schedules.
def crossover(parent1, parent2):
    """Uniform crossover: each class comes from one of the two parents."""
    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1)

    child = []
    for a, b in zip(parent1, parent2):
        child.append(copy.deepcopy(a if random.random() < 0.5 else b))
    return child


# GA mutation: randomly change an assignment to maintain diversity.
def mutation(schedule, sections, requirements, candidates):
    """Randomly move one class to another good room/timeslot."""
    if random.random() > MUTATION_RATE:
        return schedule

    indices = [i for i, req in enumerate(requirements) if req != NEEDS_NOTHING]
    if not indices:
        return schedule

    i = random.choice(indices)
    occ = Occupancy(schedule)
    occ.remove(i)

    choices = sample_choices(
        requirements[i],
        candidates[i]["rooms"],
        candidates[i]["timeslots"],
        PAIR_SAMPLE // 2,
    )

    best = None
    for room, ts in choices:
        room_id = room.id if room is not None else None
        ts_id = ts.id if ts is not None else None
        hard = occ.hard_cost(i, room_id, ts_id)

        soft = 0
        if room is not None:
            soft += room_soft_penalty(sections[i], room)
        if ts is not None:
            soft += time_soft_penalty(sections[i], ts)

        value = (hard, soft, random.random(), room_id, ts_id)
        if best is None or value < best:
            best = value

    if best is not None:
        schedule[i].room_id = best[3]
        schedule[i].timeslot_id = best[4]

    return schedule


def genetic_schedule(sections, timeslots, rooms, requirements, candidates, cache):
    """Run the Genetic Algorithm."""
    population = create_population(
        POPULATION_SIZE,
        sections,
        requirements,
        candidates,
    )

    for generation in range(GENERATIONS):
        selected = selection(population, sections, rooms, timeslots, cache)

        # Keep a few of the best schedules unchanged.
        new_population = [copy.deepcopy(s) for s in selected[:ELITE_COUNT]]

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutation(child, sections, requirements, candidates)
            new_population.append(child)

        population = new_population

        if generation in (0, 3, 7, GENERATIONS - 1):
            best = selection(population, sections, rooms, timeslots, cache)[0]
            fitness = calculate_fitness(
                best,
                rooms,
                sections=sections,
                timeslots=timeslots,
                valid_timeslot_cache=cache,
            )
            unscheduled = sum(
                not is_scheduled(item, requirements[i])
                for i, item in enumerate(best)
            )
            print(
                f"[GA] generation={generation + 1}/{GENERATIONS} "
                f"unscheduled={unscheduled} "
                f"hard={count_hard_conflicts(best)} "
                f"fitness={fitness:.4f}"
            )

    return selection(population, sections, rooms, timeslots, cache)[0]


# ============================================================
# TABU SEARCH
# Glover (1989): neighborhood moves, tabu memory, tenure, aspiration.
# ============================================================

def tabu_search(schedule, sections, rooms, timeslots, requirements, candidates):
    """Tabu Search repairs hard conflicts, then improves soft constraints."""
    current = copy.deepcopy(schedule)
    best = copy.deepcopy(schedule)
    tabu = {}
    accepted = 0

    def simple_score(s):
        unscheduled = sum(
            not is_scheduled(item, requirements[i])
            for i, item in enumerate(s)
        )
        return (unscheduled, count_hard_conflicts(s))

    # --------------------------------------------------------
    # Phase 1: repair unscheduled sections and hard conflicts.
    # --------------------------------------------------------
    best_score = simple_score(best)

    for iteration in range(TABU_ITERATIONS):
        occ = Occupancy(current)
        problems = occ.problem_indices(requirements)

        if not problems:
            print(f"[TS] feasibility reached at iteration {iteration}")
            break

        random.shuffle(problems)
        problems = problems[:8]
        move = None

        for i in problems:
            old_room = current[i].room_id
            old_time = current[i].timeslot_id
            occ.remove(i)

            choices = sample_choices(
                requirements[i],
                candidates[i]["rooms"],
                candidates[i]["timeslots"],
                TABU_SAMPLE,
            )

            for room, ts in choices:
                room_id = room.id if room is not None else None
                ts_id = ts.id if ts is not None else None
                hard = occ.hard_cost(i, room_id, ts_id)

                soft = 0
                if room is not None:
                    soft += room_soft_penalty(sections[i], room)
                if ts is not None:
                    soft += time_soft_penalty(sections[i], ts)

                key = (i, room_id, ts_id)
                is_tabu = tabu.get(key, -1) > iteration

                # Aspiration criterion (Glover, 1989): a tabu move may still
                # be considered if it improves the best feasibility score
                # found so far. The full score is evaluated only for tabu
                # candidates, so normal candidate testing stays fast.
                aspiration = False
                if is_tabu:
                    saved_room = current[i].room_id
                    saved_time = current[i].timeslot_id
                    current[i].room_id = room_id
                    current[i].timeslot_id = ts_id
                    aspiration = simple_score(current) < best_score
                    current[i].room_id = saved_room
                    current[i].timeslot_id = saved_time

                candidate = (hard, soft, random.random(), i, room_id, ts_id)

                if (not is_tabu or aspiration) and (move is None or candidate < move):
                    move = candidate

            current[i].room_id = old_room
            current[i].timeslot_id = old_time
            occ.add(i)

        if move is None:
            break

        _, _, _, i, new_room, new_time = move
        old_key = (i, current[i].room_id, current[i].timeslot_id)
        current[i].room_id = new_room
        current[i].timeslot_id = new_time
        tabu[old_key] = iteration + TABU_TENURE + random.randint(0, 4)
        accepted += 1

        score = simple_score(current)
        if score < best_score:
            best = copy.deepcopy(current)
            best_score = score

    # Use the best feasible schedule found by the repair phase.
    current = copy.deepcopy(best)

    # --------------------------------------------------------
    # Phase 2: TS intensification for soft constraints.
    # Only feasible neighborhood moves are accepted, so the 0-hard-conflict
    # solution reached in Phase 1 is never sacrificed. This is the
    # local-improvement part of the hybrid timetabling search.
    # --------------------------------------------------------
    if best_score == (0, 0):
        room_by_id = {room.id: room for room in rooms}
        time_by_id = {ts.id: ts for ts in timeslots}
        soft_moves = 0
        swap_moves = 0

        # Try moving one class to a better free room/timeslot.
        for _ in range(TS_SOFT_PASSES):
            weighted = []

            for i, item in enumerate(current):
                if requirements[i] == NEEDS_NOTHING:
                    continue

                cost = 0
                if item.room_id in room_by_id:
                    cost += room_soft_penalty(sections[i], room_by_id[item.room_id])
                if item.timeslot_id in time_by_id:
                    cost += time_soft_penalty(sections[i], time_by_id[item.timeslot_id])

                if cost > 0:
                    weighted.append((cost, i))

            weighted.sort(reverse=True)
            occ = Occupancy(current)
            improved = 0

            for _, i in weighted[:TS_SOFT_ITEMS]:
                old_room = current[i].room_id
                old_time = current[i].timeslot_id

                old_soft = 0
                if old_room in room_by_id:
                    old_soft += room_soft_penalty(sections[i], room_by_id[old_room])
                if old_time in time_by_id:
                    old_soft += time_soft_penalty(sections[i], time_by_id[old_time])

                occ.remove(i)
                best_move = None

                choices = sample_choices(
                    requirements[i],
                    candidates[i]["rooms"],
                    candidates[i]["timeslots"],
                    TS_SOFT_PAIR_SAMPLE,
                )

                for room, ts in choices:
                    room_id = room.id if room is not None else None
                    ts_id = ts.id if ts is not None else None

                    # Soft phase never introduces a hard conflict.
                    if occ.hard_cost(i, room_id, ts_id) != 0:
                        continue

                    new_soft = 0
                    if room is not None:
                        new_soft += room_soft_penalty(sections[i], room)
                    if ts is not None:
                        new_soft += time_soft_penalty(sections[i], ts)

                    gain = old_soft - new_soft
                    if gain > 0:
                        value = (-gain, random.random(), room_id, ts_id)
                        if best_move is None or value < best_move:
                            best_move = value

                if best_move is not None:
                    current[i].room_id = best_move[2]
                    current[i].timeslot_id = best_move[3]
                    occ.add(i)
                    improved += 1
                    soft_moves += 1
                else:
                    current[i].room_id = old_room
                    current[i].timeslot_id = old_time
                    occ.add(i)

            if improved == 0:
                break

        # Try room swaps between classes at the same timeslot.
        # Times do not change, so instructor conflicts cannot be introduced.
        for _ in range(TS_SWAP_PASSES):
            by_time = defaultdict(list)
            weighted = []

            for i, item in enumerate(current):
                if requirements[i] != NEEDS_ROOM_AND_TIME:
                    continue
                if item.room_id is None or item.timeslot_id is None:
                    continue

                room = room_by_id.get(item.room_id)
                if room is None:
                    continue

                cost = room_soft_penalty(sections[i], room)
                by_time[item.timeslot_id].append(i)
                if cost > 0:
                    weighted.append((cost, i))

            weighted.sort(reverse=True)
            changed = 0

            for _, i in weighted[:TS_SWAP_ITEMS]:
                item_i = current[i]
                room_i = room_by_id.get(item_i.room_id)
                partners = by_time.get(item_i.timeslot_id, [])

                if room_i is None or len(partners) < 2:
                    continue

                old_i = room_soft_penalty(sections[i], room_i)
                partners = list(partners)
                if len(partners) > TS_SWAP_PARTNERS:
                    partners = random.sample(partners, TS_SWAP_PARTNERS)

                best_swap = None

                for j in partners:
                    if i == j:
                        continue

                    item_j = current[j]
                    if item_j.room_id is None or item_j.room_id == item_i.room_id:
                        continue

                    room_j = room_by_id.get(item_j.room_id)
                    if room_j is None:
                        continue
                    # Each room came from its own section's allowed list, not
                    # the other's: a lecture must not swap into a lab.
                    if not (is_room_allowed(sections[i], room_j)
                            and is_room_allowed(sections[j], room_i)):
                        continue

                    old_j = room_soft_penalty(sections[j], room_j)
                    new_i = room_soft_penalty(sections[i], room_j)
                    new_j = room_soft_penalty(sections[j], room_i)
                    gain = (old_i + old_j) - (new_i + new_j)

                    if gain > 0:
                        value = (-gain, random.random(), j)
                        if best_swap is None or value < best_swap:
                            best_swap = value

                if best_swap is not None:
                    j = best_swap[2]
                    current[i].room_id, current[j].room_id = (
                        current[j].room_id,
                        current[i].room_id,
                    )
                    changed += 1
                    swap_moves += 1

            if changed == 0:
                break

        # Safety check: keep the soft-improved result only if it is still feasible.
        if count_hard_conflicts(current) == 0:
            best = current

        print(
            f"[TS] soft_moves={soft_moves} room_swaps={swap_moves}"
        )

    final_score = simple_score(best)
    print(
        f"[TS] accepted={accepted} "
        f"best_unscheduled={final_score[0]} best_hard={final_score[1]}"
    )
    return best


# ============================================================
# MAIN GA + TS FUNCTIONS
# ============================================================

def hybrid_schedule(sections, timeslots, rooms, cache=None):
    """Run GA first, then let Tabu Search repair and improve the schedule."""
    if cache is None:
        cache = build_timeslot_guideline_cache(sections, timeslots)

    requirements, candidates = build_cache(sections, rooms, timeslots)

    # 1) Genetic Algorithm
    best = genetic_schedule(
        sections,
        timeslots,
        rooms,
        requirements,
        candidates,
        cache,
    )

    print(
        f"After GA: unscheduled="
        f"{sum(not is_scheduled(item, requirements[i]) for i, item in enumerate(best))}, "
        f"hard={count_hard_conflicts(best)}"
    )

    # 2) Tabu Search does BOTH repair and soft improvement.
    best = tabu_search(
        best,
        sections,
        rooms,
        timeslots,
        requirements,
        candidates,
    )

    print(
        f"After GA+TS: unscheduled="
        f"{sum(not is_scheduled(item, requirements[i]) for i, item in enumerate(best))}, "
        f"hard={count_hard_conflicts(best)}"
    )

    # Siblings (the meeting blocks of one section) must not share an hour and
    # must share a room. Tabu Search prices sibling TIME clashes in its
    # occupancy index, but nothing here unifies sibling ROOMS -- placement is
    # per-item with no cross-block state. finalize_schedule repairs both, and
    # only ever moves a block into a slot/room that is genuinely free, so it
    # can improve the schedule but never break it.
    return finalize_schedule(best, sections, timeslots, cache, rooms=rooms)


def genetic_runs(sections, timeslots, rooms, num_runs=1, seeds=None):
    try:
        run_count = max(1, int(num_runs or 1))
    except (TypeError, ValueError):
        run_count = 1

    cache = build_timeslot_guideline_cache(sections, timeslots)
    requirements = [classify_section(section) for section in sections]

    best_schedule = None
    best_key = None
    runtimes = []

    for run in range(run_count):
        # Shared benchmark seeds: run i uses the same seed for every algorithm.
        run_seed = seed_all(seed_for_run(run, seeds))

        print(f"\n[GA+TS] Run {run + 1}/{run_count} (seed {run_seed})")
        start = time.perf_counter()

        schedule = hybrid_schedule(sections, timeslots, rooms, cache)
        runtime = time.perf_counter() - start
        runtimes.append(runtime)

        unscheduled = sum(
            not is_scheduled(item, requirements[i])
            for i, item in enumerate(schedule)
        )
        hard = count_hard_conflicts(schedule)
        soft = (
            total_soft_penalty(schedule, rooms)
            + total_time_penalty(schedule, timeslots)
            + count_timeslot_guideline_conflicts(
                schedule,
                sections,
                timeslots,
                valid_timeslot_cache=cache,
            )
        )
        fitness = calculate_fitness(
            schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        key = (unscheduled, hard, soft, -fitness)

        print(
            f"[GA+TS] run={run + 1} seed={run_seed} runtime={runtime:.2f}s "
            f"unscheduled={unscheduled} hard={hard} "
            f"soft={soft:.1f} fitness={fitness:.4f}"
        )

        if best_key is None or key < best_key:
            best_key = key
            best_schedule = copy.deepcopy(schedule)

    if runtimes:
        print(
            f"[GA+TS] runtime mean={statistics.mean(runtimes):.2f}s "
            f"min={min(runtimes):.2f}s max={max(runtimes):.2f}s"
        )

    # Final result summary
    scheduled = count_scheduled_sections(best_schedule, sections)
    soft_counts = soft_violation_counts(best_schedule, rooms, timeslots)
    fitness = calculate_fitness(
        best_schedule,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=cache,
    )
    soft = (
        total_soft_penalty(best_schedule, rooms)
        + total_time_penalty(best_schedule, timeslots)
        + count_timeslot_guideline_conflicts(
            best_schedule,
            sections,
            timeslots,
            valid_timeslot_cache=cache,
        )
    )

    print("\n========== GA + TABU RESULT ==========")
    print(f"Scheduled: {scheduled}/{len(sections)}")
    print(f"Unscheduled: {best_key[0]}")
    print(f"Hard conflicts: {count_hard_conflicts(best_schedule)}")
    print(f"  Room: {count_room_conflicts(best_schedule)}")
    print(f"  Instructor: {count_instructor_conflicts(best_schedule)}")
    print(f"  Sibling: {count_sibling_conflicts(best_schedule)}")
    print(f"Soft weighted total: {soft:.2f}")
    print(f"Soft violation counts: {soft_counts}")
    print(f"Timeslot guideline violations: {count_timeslot_guideline_conflicts(best_schedule, sections, timeslots, valid_timeslot_cache=cache)}")
    print(f"Fitness: {fitness:.4f}")
    print("=============================================\n")

    return best_schedule
