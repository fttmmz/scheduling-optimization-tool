import math
import random
from collections import defaultdict

from backend.Optimization.constraints import (
    get_valid_timeslots,
    passes_hard_constraints,
    get_viable_rooms,
)
from backend.models.models import ScheduleItem
from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
)

# ============================================================
# PURE-PYTHON HYBRID: MRV GREEDY + LNS + MIN-CONFLICTS
# ============================================================
# The public function remains genetic_schedule(...) so the website does
# not need to change its import or call.

# Initial construction
GREEDY_VALUE_SAMPLES = 160
GREEDY_RANDOM_TOP = 5
CONSTRUCTION_RESTARTS = 3

# Large Neighborhood Search
LNS_ITERATIONS = 350
LNS_MIN_SIZE = 35
LNS_MAX_SIZE = 110
LNS_REBUILD_SAMPLES = 220
LNS_STAGNATION_EXPAND = 35

# Min-conflicts
MIN_CONFLICT_STEPS = 12000
MIN_CONFLICT_SAMPLES = 240
RANDOM_WALK_RATE = 0.06

# Ejection chains
EJECTION_ROUNDS = 4
EJECTION_DEPTH = 3
EJECTION_BRANCHES = 14

# Final soft polish
SOFT_POLISH_STEPS = 120
SOFT_POLISH_SAMPLES = 40

# Objective priorities. One unscheduled section must be more expensive than
# several ordinary clashes, so the search first completes the timetable.
UNSCHEDULED_WEIGHT = 10000
ROOM_CONFLICT_WEIGHT = 1000
INSTRUCTOR_CONFLICT_WEIGHT = 1000

# Static feasibility memo. It only stores combinations actually tested.
_STATIC_OK_CACHE_LIMIT = 600000


# ============================================================
# BASIC HELPERS
# ============================================================
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
    # Keep section.no in its original type. This helps evaluation.py match
    # ScheduleItem objects back to their original Section objects.
    return [
        ScheduleItem(
            course_id=section.course.id,
            course_name=section.course.name,
            course_type=section.course.type,
            course_dept=section.course.dept,
            capacity=section.capacity,
            instructor_id=section.instructor_id,
            room_id=None,
            timeslot_id=None,
            section=section.no,
        )
        for section in sections
    ]


def count_unscheduled(schedule):
    return sum(
        1
        for item in schedule
        if item.room_id is None or item.timeslot_id is None
    )


# ============================================================
# OPTIONS AND STATIC FEASIBILITY
# ============================================================
_ROOMS_BY_ID_KEY = "_rooms_by_id"
_TIMESLOTS_BY_ID_KEY = "_timeslots_by_id"


def build_option_cache(sections, rooms, timeslots):
    cache = {
        _ROOMS_BY_ID_KEY: {room.id: room for room in rooms},
        _TIMESLOTS_BY_ID_KEY: {ts.id: ts for ts in timeslots},
    }

    for idx, section in enumerate(sections):
        cache[idx] = (
            list(get_viable_rooms(section, rooms)),
            list(get_valid_timeslots(section, timeslots)),
        )

    return cache


def _domain_size(idx, option_cache):
    viable_rooms, valid_timeslots = option_cache[idx]
    return len(viable_rooms) * len(valid_timeslots)


def _static_ok(idx, room, ts, sections, memo):
    """
    Test room/type/capacity/campus/timeslot constraints without treating
    current room or instructor occupancy as a hard rejection.

    Occupancy conflicts are handled by the LNS objective, which is necessary
    because escaping mutual blocking sometimes requires a temporary clash.
    """
    key = (idx, room.id, ts.id)
    cached = memo.get(key)
    if cached is not None:
        return cached

    ok = passes_hard_constraints(
        sections[idx],
        room,
        ts,
        occupied_instructors=set(),
        occupied_rooms=set(),
    )

    if len(memo) < _STATIC_OK_CACHE_LIMIT:
        memo[key] = bool(ok)

    return bool(ok)


# ============================================================
# OCCUPANCY AND OBJECTIVE
# ============================================================
def _build_occupancy(schedule):
    room_counts = defaultdict(int)
    instructor_counts = defaultdict(int)

    for item in schedule:
        if item.room_id is None or item.timeslot_id is None:
            continue

        room_counts[(item.room_id, item.timeslot_id)] += 1
        if item.instructor_id is not None:
            instructor_counts[(item.instructor_id, item.timeslot_id)] += 1

    return room_counts, instructor_counts


def _add_assignment(item, room_id, timeslot_id, room_counts, instructor_counts):
    item.room_id = room_id
    item.timeslot_id = timeslot_id
    room_counts[(room_id, timeslot_id)] += 1
    if item.instructor_id is not None:
        instructor_counts[(item.instructor_id, timeslot_id)] += 1


def _remove_assignment(item, room_counts, instructor_counts):
    if item.room_id is None or item.timeslot_id is None:
        item.room_id = None
        item.timeslot_id = None
        return

    room_key = (item.room_id, item.timeslot_id)
    room_counts[room_key] -= 1
    if room_counts[room_key] <= 0:
        del room_counts[room_key]

    if item.instructor_id is not None:
        instructor_key = (item.instructor_id, item.timeslot_id)
        instructor_counts[instructor_key] -= 1
        if instructor_counts[instructor_key] <= 0:
            del instructor_counts[instructor_key]

    item.room_id = None
    item.timeslot_id = None


def _objective_from_counts(schedule, room_counts, instructor_counts):
    unscheduled = count_unscheduled(schedule)
    room_conflicts = sum(max(0, count - 1) for count in room_counts.values())
    instructor_conflicts = sum(
        max(0, count - 1) for count in instructor_counts.values()
    )

    return (
        unscheduled * UNSCHEDULED_WEIGHT
        + room_conflicts * ROOM_CONFLICT_WEIGHT
        + instructor_conflicts * INSTRUCTOR_CONFLICT_WEIGHT
    )


def _objective(schedule):
    room_counts, instructor_counts = _build_occupancy(schedule)
    return _objective_from_counts(schedule, room_counts, instructor_counts)


def _placement_cost(item, room_id, timeslot_id, room_counts, instructor_counts):
    room_conflicts = room_counts.get((room_id, timeslot_id), 0)
    instructor_conflicts = 0
    if item.instructor_id is not None:
        instructor_conflicts = instructor_counts.get(
            (item.instructor_id, timeslot_id), 0
        )

    return (
        room_conflicts * ROOM_CONFLICT_WEIGHT
        + instructor_conflicts * INSTRUCTOR_CONFLICT_WEIGHT
    )


def _item_is_problem(item, room_counts, instructor_counts):
    if item.room_id is None or item.timeslot_id is None:
        return True

    if room_counts.get((item.room_id, item.timeslot_id), 0) > 1:
        return True

    return (
        item.instructor_id is not None
        and instructor_counts.get((item.instructor_id, item.timeslot_id), 0) > 1
    )


def _problem_indices(schedule, room_counts=None, instructor_counts=None):
    if room_counts is None or instructor_counts is None:
        room_counts, instructor_counts = _build_occupancy(schedule)

    return [
        idx
        for idx, item in enumerate(schedule)
        if _item_is_problem(item, room_counts, instructor_counts)
    ]


# ============================================================
# CANDIDATE GENERATION
# ============================================================
def _sample_pairs(viable_rooms, valid_timeslots, limit):
    total = len(viable_rooms) * len(valid_timeslots)
    if total == 0:
        return []

    if total <= limit:
        pairs = [
            (room, ts)
            for ts in valid_timeslots
            for room in viable_rooms
        ]
        random.shuffle(pairs)
        return pairs

    pairs = []
    seen = set()
    attempts = 0
    max_attempts = limit * 5

    while len(pairs) < limit and attempts < max_attempts:
        attempts += 1
        room = random.choice(viable_rooms)
        ts = random.choice(valid_timeslots)
        key = (room.id, ts.id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((room, ts))

    return pairs


def _best_candidates(
    idx,
    schedule,
    sections,
    option_cache,
    room_counts,
    instructor_counts,
    static_memo,
    sample_limit,
    top_k=1,
):
    item = schedule[idx]
    viable_rooms, valid_timeslots = option_cache[idx]

    if not viable_rooms or not valid_timeslots:
        return []

    ranked = []
    for room, ts in _sample_pairs(viable_rooms, valid_timeslots, sample_limit):
        if not _static_ok(idx, room, ts, sections, static_memo):
            continue

        cost = _placement_cost(
            item,
            room.id,
            ts.id,
            room_counts,
            instructor_counts,
        )

        # A small random tie-breaker gives restarts and LNS rebuilds variety.
        ranked.append((cost, random.random(), room, ts))

    ranked.sort(key=lambda value: (value[0], value[1]))
    return [(cost, room, ts) for cost, _, room, ts in ranked[:top_k]]


# ============================================================
# INITIAL MRV CONSTRUCTION
# ============================================================
def _construction_order(sections, option_cache):
    instructor_load = defaultdict(int)
    for section in sections:
        if section.instructor_id is not None:
            instructor_load[section.instructor_id] += 1

    order = list(range(len(sections)))
    order.sort(
        key=lambda idx: (
            _domain_size(idx, option_cache),
            -instructor_load.get(sections[idx].instructor_id, 0),
            -getattr(sections[idx], "capacity", 0),
            random.random(),
        )
    )
    return order


def _construct_once(sections, option_cache, static_memo):
    schedule = _new_schedule(sections)
    room_counts = defaultdict(int)
    instructor_counts = defaultdict(int)

    for idx in _construction_order(sections, option_cache):
        candidates = _best_candidates(
            idx,
            schedule,
            sections,
            option_cache,
            room_counts,
            instructor_counts,
            static_memo,
            sample_limit=GREEDY_VALUE_SAMPLES,
            top_k=GREEDY_RANDOM_TOP,
        )

        if not candidates:
            continue

        # Prefer a zero-conflict assignment. If none exists, leave the section
        # unscheduled; the LNS stage will consider coordinated moves later.
        zero_cost = [candidate for candidate in candidates if candidate[0] == 0]
        if not zero_cost:
            continue

        _, room, ts = random.choice(zero_cost)
        _add_assignment(
            schedule[idx], room.id, ts.id, room_counts, instructor_counts
        )

    return schedule


def _initial_construction(sections, option_cache, static_memo):
    best = None
    best_cost = math.inf

    for _ in range(CONSTRUCTION_RESTARTS):
        candidate = _construct_once(sections, option_cache, static_memo)
        candidate_cost = _objective(candidate)
        if candidate_cost < best_cost:
            best = candidate
            best_cost = candidate_cost

    return best


# ============================================================
# LARGE NEIGHBORHOOD SEARCH
# ============================================================
def _indices_using_room_time(schedule, room_id, timeslot_id):
    return [
        idx
        for idx, item in enumerate(schedule)
        if item.room_id == room_id and item.timeslot_id == timeslot_id
    ]


def _indices_using_instructor_time(schedule, instructor_id, timeslot_id):
    if instructor_id is None:
        return []
    return [
        idx
        for idx, item in enumerate(schedule)
        if item.instructor_id == instructor_id
        and item.timeslot_id == timeslot_id
    ]


def _select_neighborhood(schedule, option_cache, target_size):
    room_counts, instructor_counts = _build_occupancy(schedule)
    problems = _problem_indices(schedule, room_counts, instructor_counts)

    if not problems:
        return []

    seed = random.choice(problems)
    chosen = {seed}
    frontier = [seed]

    while frontier and len(chosen) < target_size:
        idx = frontier.pop()
        item = schedule[idx]

        related = []
        if item.room_id is not None and item.timeslot_id is not None:
            related.extend(
                _indices_using_room_time(
                    schedule, item.room_id, item.timeslot_id
                )
            )
            related.extend(
                _indices_using_instructor_time(
                    schedule, item.instructor_id, item.timeslot_id
                )
            )

        # Include sections of the same instructor because moving one of them
        # often frees the exact timeslot needed by another.
        if item.instructor_id is not None:
            related.extend(
                idx2
                for idx2, other in enumerate(schedule)
                if other.instructor_id == item.instructor_id
            )

        random.shuffle(related)
        for idx2 in related:
            if len(chosen) >= target_size:
                break
            if idx2 not in chosen:
                chosen.add(idx2)
                frontier.append(idx2)

    # Fill the rest with difficult/problem sections first, then random nearby
    # sections. This makes the neighborhood large enough to break blocking.
    remaining_problems = [idx for idx in problems if idx not in chosen]
    remaining_problems.sort(key=lambda idx: _domain_size(idx, option_cache))

    for idx in remaining_problems:
        if len(chosen) >= target_size:
            break
        chosen.add(idx)

    if len(chosen) < target_size:
        pool = [idx for idx in range(len(schedule)) if idx not in chosen]
        random.shuffle(pool)
        chosen.update(pool[: target_size - len(chosen)])

    return list(chosen)


def _rebuild_neighborhood(
    schedule,
    neighborhood,
    sections,
    option_cache,
    static_memo,
):
    room_counts, instructor_counts = _build_occupancy(schedule)

    for idx in neighborhood:
        _remove_assignment(schedule[idx], room_counts, instructor_counts)

    neighborhood.sort(
        key=lambda idx: (
            _domain_size(idx, option_cache),
            -getattr(sections[idx], "capacity", 0),
            random.random(),
        )
    )

    for idx in neighborhood:
        candidates = _best_candidates(
            idx,
            schedule,
            sections,
            option_cache,
            room_counts,
            instructor_counts,
            static_memo,
            sample_limit=LNS_REBUILD_SAMPLES,
            top_k=GREEDY_RANDOM_TOP,
        )
        if not candidates:
            continue

        best_cost = candidates[0][0]
        near_best = [
            candidate
            for candidate in candidates
            if candidate[0] <= best_cost + ROOM_CONFLICT_WEIGHT
        ]
        _, room, ts = random.choice(near_best)
        _add_assignment(
            schedule[idx], room.id, ts.id, room_counts, instructor_counts
        )

    return schedule


def _large_neighborhood_search(schedule, sections, option_cache, static_memo):
    best = clone_schedule(schedule)
    best_cost = _objective(best)
    current = clone_schedule(best)
    current_cost = best_cost
    stagnation = 0

    for iteration in range(LNS_ITERATIONS):
        expansion = min(
            LNS_MAX_SIZE - LNS_MIN_SIZE,
            (stagnation // LNS_STAGNATION_EXPAND) * 15,
        )
        target_size = random.randint(
            LNS_MIN_SIZE,
            min(LNS_MAX_SIZE, LNS_MIN_SIZE + 25 + expansion),
        )

        neighborhood = _select_neighborhood(
            current, option_cache, target_size
        )
        if not neighborhood:
            break

        candidate = clone_schedule(current)
        _rebuild_neighborhood(
            candidate,
            neighborhood,
            sections,
            option_cache,
            static_memo,
        )
        candidate_cost = _objective(candidate)

        temperature = max(1.0, 5000.0 * (1.0 - iteration / LNS_ITERATIONS))
        accept_worse = (
            candidate_cost > current_cost
            and random.random()
            < math.exp(-(candidate_cost - current_cost) / temperature)
        )

        if candidate_cost <= current_cost or accept_worse:
            current = candidate
            current_cost = candidate_cost

        if candidate_cost < best_cost:
            best = clone_schedule(candidate)
            best_cost = candidate_cost
            stagnation = 0
        else:
            stagnation += 1

        if best_cost == 0:
            break

    return best


# ============================================================
# MIN-CONFLICTS REPAIR
# ============================================================
def _min_conflicts(schedule, sections, option_cache, static_memo):
    best = clone_schedule(schedule)
    best_cost = _objective(best)

    current = clone_schedule(schedule)
    room_counts, instructor_counts = _build_occupancy(current)

    for _ in range(MIN_CONFLICT_STEPS):
        problems = _problem_indices(current, room_counts, instructor_counts)
        if not problems:
            return current

        idx = random.choice(problems)
        item = current[idx]

        old_room_id = item.room_id
        old_timeslot_id = item.timeslot_id
        _remove_assignment(item, room_counts, instructor_counts)

        candidates = _best_candidates(
            idx,
            current,
            sections,
            option_cache,
            room_counts,
            instructor_counts,
            static_memo,
            sample_limit=MIN_CONFLICT_SAMPLES,
            top_k=12,
        )

        if not candidates:
            # Restore its old assignment if it had one.
            if old_room_id is not None and old_timeslot_id is not None:
                _add_assignment(
                    item,
                    old_room_id,
                    old_timeslot_id,
                    room_counts,
                    instructor_counts,
                )
            continue

        if random.random() < RANDOM_WALK_RATE:
            _, room, ts = random.choice(candidates)
        else:
            best_local_cost = candidates[0][0]
            best_local = [
                candidate
                for candidate in candidates
                if candidate[0] == best_local_cost
            ]
            _, room, ts = random.choice(best_local)

        _add_assignment(
            item, room.id, ts.id, room_counts, instructor_counts
        )

        current_cost = _objective_from_counts(
            current, room_counts, instructor_counts
        )
        if current_cost < best_cost:
            best = clone_schedule(current)
            best_cost = current_cost
            if best_cost == 0:
                break

    return best


# ============================================================
# DEPTH-LIMITED EJECTION CHAINS
# ============================================================
def _blocking_indices(schedule, idx, room_id, timeslot_id):
    item = schedule[idx]
    blockers = []

    for other_idx, other in enumerate(schedule):
        if other_idx == idx:
            continue
        if other.room_id == room_id and other.timeslot_id == timeslot_id:
            blockers.append(other_idx)
            continue
        if (
            item.instructor_id is not None
            and other.instructor_id == item.instructor_id
            and other.timeslot_id == timeslot_id
        ):
            blockers.append(other_idx)

    return blockers


def _try_ejection(
    schedule,
    idx,
    sections,
    option_cache,
    static_memo,
    depth,
    visited,
):
    if depth < 0 or idx in visited:
        return False

    visited = set(visited)
    visited.add(idx)

    room_counts, instructor_counts = _build_occupancy(schedule)
    item = schedule[idx]
    old_room_id = item.room_id
    old_timeslot_id = item.timeslot_id
    _remove_assignment(item, room_counts, instructor_counts)

    candidates = _best_candidates(
        idx,
        schedule,
        sections,
        option_cache,
        room_counts,
        instructor_counts,
        static_memo,
        sample_limit=MIN_CONFLICT_SAMPLES,
        top_k=EJECTION_BRANCHES,
    )

    # Restore before testing tentative recursive moves.
    if old_room_id is not None and old_timeslot_id is not None:
        _add_assignment(
            item,
            old_room_id,
            old_timeslot_id,
            room_counts,
            instructor_counts,
        )

    for cost, room, ts in candidates:
        if cost == 0:
            room_counts, instructor_counts = _build_occupancy(schedule)
            _remove_assignment(item, room_counts, instructor_counts)
            _add_assignment(
                item, room.id, ts.id, room_counts, instructor_counts
            )
            return True

        if depth == 0:
            continue

        blockers = _blocking_indices(
            schedule, idx, room.id, ts.id
        )
        if not blockers or len(blockers) > 2:
            continue

        snapshot = clone_schedule(schedule)
        moved_all = True

        # Temporarily remove the target so blockers can be relocated without
        # the target's old assignment occupying space.
        schedule[idx].room_id = None
        schedule[idx].timeslot_id = None

        for blocker_idx in blockers:
            if not _try_ejection(
                schedule,
                blocker_idx,
                sections,
                option_cache,
                static_memo,
                depth - 1,
                visited,
            ):
                moved_all = False
                break

        if moved_all:
            room_counts, instructor_counts = _build_occupancy(schedule)
            if _placement_cost(
                schedule[idx],
                room.id,
                ts.id,
                room_counts,
                instructor_counts,
            ) == 0:
                _add_assignment(
                    schedule[idx],
                    room.id,
                    ts.id,
                    room_counts,
                    instructor_counts,
                )
                return True

        schedule[:] = clone_schedule(snapshot)

    return False


def _ejection_chain_repair(schedule, sections, option_cache, static_memo):
    best = clone_schedule(schedule)
    best_cost = _objective(best)

    for _ in range(EJECTION_ROUNDS):
        room_counts, instructor_counts = _build_occupancy(best)
        problems = _problem_indices(best, room_counts, instructor_counts)
        problems.sort(key=lambda idx: _domain_size(idx, option_cache))

        improved = False
        for idx in problems:
            candidate = clone_schedule(best)
            if _try_ejection(
                candidate,
                idx,
                sections,
                option_cache,
                static_memo,
                EJECTION_DEPTH,
                visited=set(),
            ):
                candidate_cost = _objective(candidate)
                if candidate_cost < best_cost:
                    best = candidate
                    best_cost = candidate_cost
                    improved = True

        if not improved or best_cost == 0:
            break

    return best


# ============================================================
# FINAL SOFT-CONSTRAINT POLISH
# ============================================================
def _soft_polish(schedule, sections, rooms, timeslots, cache, option_cache, static_memo):
    best = clone_schedule(schedule)
    hard_cost = _objective(best)

    best_fitness = calculate_fitness(
        best,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=cache,
    )

    for _ in range(SOFT_POLISH_STEPS):
        idx = random.randrange(len(best))
        candidate = clone_schedule(best)
        room_counts, instructor_counts = _build_occupancy(candidate)
        _remove_assignment(candidate[idx], room_counts, instructor_counts)

        choices = _best_candidates(
            idx,
            candidate,
            sections,
            option_cache,
            room_counts,
            instructor_counts,
            static_memo,
            sample_limit=SOFT_POLISH_SAMPLES,
            top_k=8,
        )
        zero_conflict_choices = [choice for choice in choices if choice[0] == 0]
        if not zero_conflict_choices:
            continue

        _, room, ts = random.choice(zero_conflict_choices)
        _add_assignment(
            candidate[idx], room.id, ts.id, room_counts, instructor_counts
        )

        if _objective_from_counts(candidate, room_counts, instructor_counts) != hard_cost:
            continue

        candidate_fitness = calculate_fitness(
            candidate,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        if candidate_fitness > best_fitness:
            best = candidate
            best_fitness = candidate_fitness

    return best


# ============================================================
# PUBLIC ENTRY POINT -- NAME KEPT FOR WEBSITE COMPATIBILITY
# ============================================================
def genetic_schedule(sections, timeslots, rooms, cache=None, option_cache=None):
    """
    Website-compatible entry point.

    Despite the retained historical name, this function now runs:
        1. MRV greedy construction with randomized restarts
        2. Large Neighborhood Search
        3. Min-conflicts repair
        4. Depth-limited ejection chains
        5. Soft-constraint hill climbing

    It uses only the Python standard library and the project's existing
    constraint/evaluation functions.
    """
    print("\n*** RUNNING MRV + LNS + MIN-CONFLICTS HYBRID ***")

    if cache is None:
        cache = build_timeslot_guideline_cache(sections, timeslots)

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    static_memo = {}

    best = _initial_construction(sections, option_cache, static_memo)
    best = _large_neighborhood_search(
        best, sections, option_cache, static_memo
    )
    best = _min_conflicts(
        best, sections, option_cache, static_memo
    )
    best = _ejection_chain_repair(
        best, sections, option_cache, static_memo
    )

    # A second LNS/min-conflicts pass benefits from slots opened by ejections.
    best = _large_neighborhood_search(
        best, sections, option_cache, static_memo
    )
    best = _min_conflicts(
        best, sections, option_cache, static_memo
    )

    best = _soft_polish(
        best,
        sections,
        rooms,
        timeslots,
        cache,
        option_cache,
        static_memo,
    )

    _print_diagnostics(best, option_cache)
    return best