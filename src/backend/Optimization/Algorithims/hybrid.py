import math
import random
from collections import defaultdict

from backend.Optimization.constraints import (
    NEEDS_NOTHING,
    NEEDS_ROOM_ONLY,
    classify_section,
    get_valid_timeslots,
    get_viable_rooms,
    passes_hard_constraints,
)
from backend.models.models import ScheduleItem
from backend.Optimization.evaluation import (
    build_timeslot_guideline_cache,
    calculate_fitness,
)

# ============================================================
# PURE-PYTHON HYBRID
# Classification-aware MRV construction + LNS + min-conflicts
# ============================================================

CONSTRUCTION_RESTARTS = 4
CONSTRUCTION_PAIR_SAMPLES = 220

LNS_ITERATIONS = 100
LNS_MIN_SIZE = 30
LNS_MAX_SIZE = 100
LNS_PAIR_SAMPLES = 280

MIN_CONFLICT_STEPS = 7000
MIN_CONFLICT_PAIR_SAMPLES = 320
RANDOM_WALK_RATE = 0.06

# Ejection-chain repair: move blocking sections to make space for
# unscheduled sections instead of leaving them unassigned.
EJECTION_REPAIR_ROUNDS = 1
EJECTION_MAX_DEPTH = 1
EJECTION_MAX_BLOCKERS = 1
EJECTION_PAIR_SAMPLES = 250
EJECTION_TOP_K = 20


# Direct unscheduled rescue: first use conflict-free openings, then try a
# single-blocker relocation. This is cheaper than deep recursive ejection.
DIRECT_RESCUE_ROUNDS = 3
DIRECT_RESCUE_PAIR_SAMPLES = 900
DIRECT_RESCUE_BLOCKER_SAMPLES = 360

SOFT_POLISH_STEPS = 60
SOFT_POLISH_PAIR_SAMPLES = 60

UNSCHEDULED_WEIGHT = 10000
ROOM_CONFLICT_WEIGHT = 1000
INSTRUCTOR_CONFLICT_WEIGHT = 1000

STATIC_CACHE_LIMIT = 700000

_ROOMS_BY_ID_KEY = "_rooms_by_id"
_TIMESLOTS_BY_ID_KEY = "_timeslots_by_id"
_REQUIREMENT_KEY = "_requirement"


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
    return [clone_item(schedule_item) for schedule_item in schedule]


def _new_schedule(sections):
    # Preserve section.no exactly because evaluation.py uses it to match
    # ScheduleItem objects back to Section objects.
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


def _item_requirement(item):
    if item.course_type in NEEDS_NOTHING:
        return "NEEDS_NOTHING"
    if item.course_type in NEEDS_ROOM_ONLY:
        return "NEEDS_ROOM_ONLY"
    return "NEEDS_ROOM_AND_TIME"


def _is_scheduled(item):
    requirement = _item_requirement(item)
    if requirement == "NEEDS_NOTHING":
        return True
    if requirement == "NEEDS_ROOM_ONLY":
        return item.room_id is not None
    return item.room_id is not None and item.timeslot_id is not None


def count_unscheduled(schedule):
    return sum(1 for schedule_item in schedule if not _is_scheduled(schedule_item))


# ============================================================
# OPTION CACHE
# ============================================================
def build_option_cache(sections, rooms, timeslots):
    cache = {
        _ROOMS_BY_ID_KEY: {room.id: room for room in rooms},
        _TIMESLOTS_BY_ID_KEY: {timeslot.id: timeslot for timeslot in timeslots},
    }

    for idx, section in enumerate(sections):
        requirement = classify_section(section)

        if requirement == "NEEDS_NOTHING":
            viable_rooms = []
            valid_timeslots = []
        elif requirement == "NEEDS_ROOM_ONLY":
            viable_rooms = list(get_viable_rooms(section, rooms))
            valid_timeslots = []
        else:
            viable_rooms = list(get_viable_rooms(section, rooms))
            valid_timeslots = list(get_valid_timeslots(section, timeslots))

        cache[idx] = {
            _REQUIREMENT_KEY: requirement,
            "rooms": viable_rooms,
            "timeslots": valid_timeslots,
        }

    return cache


def _requirement(idx, option_cache):
    return option_cache[idx][_REQUIREMENT_KEY]


def _domain_size(idx, option_cache):
    requirement = _requirement(idx, option_cache)
    if requirement == "NEEDS_NOTHING":
        return 0
    if requirement == "NEEDS_ROOM_ONLY":
        return len(option_cache[idx]["rooms"])
    return len(option_cache[idx]["rooms"]) * len(option_cache[idx]["timeslots"])


# ============================================================
# OCCUPANCY AND OBJECTIVE
# ============================================================
def _build_occupancy(schedule):
    room_counts = defaultdict(int)
    instructor_counts = defaultdict(int)

    for schedule_item in schedule:
        # Room-only activities have no timeslot and therefore do not create
        # a room-time or instructor-time collision.
        if schedule_item.room_id is None or schedule_item.timeslot_id is None:
            continue

        room_counts[(schedule_item.room_id, schedule_item.timeslot_id)] += 1
        if schedule_item.instructor_id is not None:
            instructor_counts[(schedule_item.instructor_id, schedule_item.timeslot_id)] += 1

    return room_counts, instructor_counts


def _remove_assignment(schedule_item, room_counts, instructor_counts):
    if schedule_item.room_id is not None and schedule_item.timeslot_id is not None:
        room_key = (schedule_item.room_id, schedule_item.timeslot_id)
        room_counts[room_key] -= 1
        if room_counts[room_key] <= 0:
            del room_counts[room_key]

        if schedule_item.instructor_id is not None:
            instructor_key = (schedule_item.instructor_id, schedule_item.timeslot_id)
            instructor_counts[instructor_key] -= 1
            if instructor_counts[instructor_key] <= 0:
                del instructor_counts[instructor_key]

    schedule_item.room_id = None
    schedule_item.timeslot_id = None


def _add_assignment(schedule_item, room_id, timeslot_id, room_counts, instructor_counts):
    schedule_item.room_id = room_id
    schedule_item.timeslot_id = timeslot_id

    if room_id is None or timeslot_id is None:
        return

    room_counts[(room_id, timeslot_id)] += 1
    if schedule_item.instructor_id is not None:
        instructor_counts[(schedule_item.instructor_id, timeslot_id)] += 1


def _conflict_totals(room_counts, instructor_counts):
    room_conflicts = sum(max(0, count - 1) for count in room_counts.values())
    instructor_conflicts = sum(
        max(0, count - 1) for count in instructor_counts.values()
    )
    return room_conflicts, instructor_conflicts


def _objective_from_counts(schedule, room_counts, instructor_counts):
    room_conflicts, instructor_conflicts = _conflict_totals(
        room_counts, instructor_counts
    )
    return (
        count_unscheduled(schedule) * UNSCHEDULED_WEIGHT
        + room_conflicts * ROOM_CONFLICT_WEIGHT
        + instructor_conflicts * INSTRUCTOR_CONFLICT_WEIGHT
    )


def _objective(schedule):
    room_counts, instructor_counts = _build_occupancy(schedule)
    return _objective_from_counts(schedule, room_counts, instructor_counts)


def _placement_cost(schedule_item, room_id, timeslot_id, room_counts, instructor_counts):
    if timeslot_id is None:
        return 0

    cost = room_counts.get((room_id, timeslot_id), 0) * ROOM_CONFLICT_WEIGHT
    if schedule_item.instructor_id is not None:
        cost += (
            instructor_counts.get(
                (schedule_item.instructor_id, timeslot_id), 0
            )
            * INSTRUCTOR_CONFLICT_WEIGHT
        )
    return cost


def _is_problem(schedule_item, room_counts, instructor_counts):
    if not _is_scheduled(schedule_item):
        return True

    if schedule_item.timeslot_id is None:
        return False

    if room_counts.get((schedule_item.room_id, schedule_item.timeslot_id), 0) > 1:
        return True

    return (
        schedule_item.instructor_id is not None
        and instructor_counts.get(
            (schedule_item.instructor_id, schedule_item.timeslot_id), 0
        )
        > 1
    )


def _problem_indices(schedule, room_counts=None, instructor_counts=None):
    if room_counts is None or instructor_counts is None:
        room_counts, instructor_counts = _build_occupancy(schedule)

    return [
        idx
        for idx, schedule_item in enumerate(schedule)
        if _is_problem(schedule_item, room_counts, instructor_counts)
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
            (room, timeslot)
            for timeslot in valid_timeslots
            for room in viable_rooms
        ]
        random.shuffle(pairs)
        return pairs

    pairs = []
    seen = set()
    attempts = 0
    max_attempts = limit * 6

    while len(pairs) < limit and attempts < max_attempts:
        attempts += 1
        room = random.choice(viable_rooms)
        timeslot = random.choice(valid_timeslots)
        key = (room.id, timeslot.id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((room, timeslot))

    return pairs


def _static_ok(idx, room, timeslot, sections, static_memo):
    key = (idx, room.id, timeslot.id)
    if key in static_memo:
        return static_memo[key]

    ok = passes_hard_constraints(
        sections[idx],
        room,
        timeslot,
        occupied_instructors=set(),
        occupied_rooms=set(),
    )

    if len(static_memo) < STATIC_CACHE_LIMIT:
        static_memo[key] = bool(ok)

    return bool(ok)


def _best_candidates(
    idx,
    schedule,
    sections,
    option_cache,
    room_counts,
    instructor_counts,
    static_memo,
    sample_limit,
    top_k=8,
):
    requirement = _requirement(idx, option_cache)
    schedule_item = schedule[idx]

    if requirement == "NEEDS_NOTHING":
        return [(0, None, None)]

    viable_rooms = option_cache[idx]["rooms"]
    if not viable_rooms:
        return []

    if requirement == "NEEDS_ROOM_ONLY":
        rooms_to_try = list(viable_rooms)
        random.shuffle(rooms_to_try)
        return [(0, room, None) for room in rooms_to_try[:top_k]]

    valid_timeslots = option_cache[idx]["timeslots"]
    if not valid_timeslots:
        return []

    ranked = []
    for room, timeslot in _sample_pairs(
        viable_rooms, valid_timeslots, sample_limit
    ):
        if not _static_ok(idx, room, timeslot, sections, static_memo):
            continue

        cost = _placement_cost(
            schedule_item,
            room.id,
            timeslot.id,
            room_counts,
            instructor_counts,
        )
        ranked.append((cost, random.random(), room, timeslot))

    ranked.sort(key=lambda value: (value[0], value[1]))
    return [
        (cost, room, timeslot)
        for cost, _, room, timeslot in ranked[:top_k]
    ]


# ============================================================
# INITIAL MRV CONSTRUCTION
# ============================================================
def _construction_order(sections, option_cache):
    instructor_load = defaultdict(int)
    for section in sections:
        if section.instructor_id is not None:
            instructor_load[section.instructor_id] += 1

    indices = [
        idx
        for idx in range(len(sections))
        if _requirement(idx, option_cache) != "NEEDS_NOTHING"
    ]

    indices.sort(
        key=lambda idx: (
            _domain_size(idx, option_cache),
            -instructor_load.get(sections[idx].instructor_id, 0),
            -getattr(sections[idx], "capacity", 0),
            random.random(),
        )
    )
    return indices


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
            CONSTRUCTION_PAIR_SAMPLES,
            top_k=10,
        )
        if not candidates:
            continue

        zero_cost = [candidate for candidate in candidates if candidate[0] == 0]
        chosen = random.choice(zero_cost if zero_cost else candidates[:3])
        _, room, timeslot = chosen

        if room is None:
            continue

        _add_assignment(
            schedule[idx],
            room.id,
            timeslot.id if timeslot is not None else None,
            room_counts,
            instructor_counts,
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
def _select_neighborhood(schedule, option_cache, target_size):
    room_counts, instructor_counts = _build_occupancy(schedule)
    problems = _problem_indices(schedule, room_counts, instructor_counts)
    if not problems:
        return []

    seed = random.choice(problems)
    chosen = {seed}
    seed_item = schedule[seed]

    related = []
    for idx, schedule_item in enumerate(schedule):
        if idx == seed:
            continue

        same_room_time = (
            seed_item.room_id is not None
            and seed_item.timeslot_id is not None
            and schedule_item.room_id == seed_item.room_id
            and schedule_item.timeslot_id == seed_item.timeslot_id
        )
        same_instructor = (
            seed_item.instructor_id is not None
            and schedule_item.instructor_id == seed_item.instructor_id
        )
        same_timeslot = (
            seed_item.timeslot_id is not None
            and schedule_item.timeslot_id == seed_item.timeslot_id
        )

        if same_room_time or same_instructor or same_timeslot:
            related.append(idx)

    random.shuffle(related)
    for idx in related:
        if len(chosen) >= target_size:
            break
        chosen.add(idx)

    remaining_problems = [idx for idx in problems if idx not in chosen]
    remaining_problems.sort(key=lambda idx: _domain_size(idx, option_cache))
    for idx in remaining_problems:
        if len(chosen) >= target_size:
            break
        chosen.add(idx)

    if len(chosen) < target_size:
        pool = [
            idx
            for idx in range(len(schedule))
            if idx not in chosen
            and _requirement(idx, option_cache) != "NEEDS_NOTHING"
        ]
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
        requirement = _requirement(idx, option_cache)
        if requirement == "NEEDS_NOTHING":
            continue

        candidates = _best_candidates(
            idx,
            schedule,
            sections,
            option_cache,
            room_counts,
            instructor_counts,
            static_memo,
            LNS_PAIR_SAMPLES,
            top_k=12,
        )
        if not candidates:
            continue

        best_cost = candidates[0][0]
        near_best = [
            candidate
            for candidate in candidates
            if candidate[0] <= best_cost + ROOM_CONFLICT_WEIGHT
        ]
        _, room, timeslot = random.choice(near_best)

        if room is None:
            continue

        _add_assignment(
            schedule[idx],
            room.id,
            timeslot.id if timeslot is not None else None,
            room_counts,
            instructor_counts,
        )

    return schedule


def _large_neighborhood_search(schedule, sections, option_cache, static_memo):
    best = clone_schedule(schedule)
    best_cost = _objective(best)
    current = clone_schedule(best)
    current_cost = best_cost

    for iteration in range(LNS_ITERATIONS):
        target_size = random.randint(LNS_MIN_SIZE, LNS_MAX_SIZE)
        neighborhood = _select_neighborhood(current, option_cache, target_size)
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

        temperature = max(
            1.0,
            4000.0 * (1.0 - iteration / max(1, LNS_ITERATIONS)),
        )
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
        requirement = _requirement(idx, option_cache)
        if requirement == "NEEDS_NOTHING":
            continue

        schedule_item = current[idx]
        old_room_id = schedule_item.room_id
        old_timeslot_id = schedule_item.timeslot_id
        _remove_assignment(schedule_item, room_counts, instructor_counts)

        candidates = _best_candidates(
            idx,
            current,
            sections,
            option_cache,
            room_counts,
            instructor_counts,
            static_memo,
            MIN_CONFLICT_PAIR_SAMPLES,
            top_k=18,
        )

        if not candidates:
            if old_room_id is not None:
                _add_assignment(
                    schedule_item,
                    old_room_id,
                    old_timeslot_id,
                    room_counts,
                    instructor_counts,
                )
            continue

        if random.random() < RANDOM_WALK_RATE:
            _, room, timeslot = random.choice(candidates)
        else:
            best_local_cost = candidates[0][0]
            best_local = [
                candidate
                for candidate in candidates
                if candidate[0] == best_local_cost
            ]
            _, room, timeslot = random.choice(best_local)

        if room is not None:
            _add_assignment(
                schedule_item,
                room.id,
                timeslot.id if timeslot is not None else None,
                room_counts,
                instructor_counts,
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
# DIRECT UNSCHEDULED RESCUE
# ============================================================
def _zero_cost_candidates(
    idx,
    schedule,
    sections,
    option_cache,
    room_counts,
    instructor_counts,
    static_memo,
    sample_limit,
    top_k=30,
):
    return [
        candidate
        for candidate in _best_candidates(
            idx,
            schedule,
            sections,
            option_cache,
            room_counts,
            instructor_counts,
            static_memo,
            sample_limit,
            top_k=top_k,
        )
        if candidate[0] == 0
    ]


def _try_single_blocker_rescue(
    schedule,
    idx,
    sections,
    option_cache,
    static_memo,
):
    """Place idx by moving at most one blocking section to a free position."""
    base = clone_schedule(schedule)
    room_counts, instructor_counts = _build_occupancy(base)
    _remove_assignment(base[idx], room_counts, instructor_counts)

    candidates = _best_candidates(
        idx,
        base,
        sections,
        option_cache,
        room_counts,
        instructor_counts,
        static_memo,
        DIRECT_RESCUE_PAIR_SAMPLES,
        top_k=40,
    )

    ranked = []
    for cost, room, timeslot in candidates:
        if room is None:
            continue
        if timeslot is None:
            ranked.append((0, cost, random.random(), room, timeslot, []))
            continue
        blockers = _blocking_indices(base, idx, room.id, timeslot.id)
        if len(blockers) <= 1:
            ranked.append(
                (len(blockers), cost, random.random(), room, timeslot, blockers)
            )

    ranked.sort(key=lambda value: (value[0], value[1], value[2]))

    for blocker_count, _, _, room, timeslot, blockers in ranked:
        trial = clone_schedule(base)
        trial_room_counts, trial_instructor_counts = _build_occupancy(trial)

        if blocker_count == 0:
            _add_assignment(
                trial[idx],
                room.id,
                timeslot.id if timeslot is not None else None,
                trial_room_counts,
                trial_instructor_counts,
            )
            return trial

        blocker_idx = blockers[0]
        _remove_assignment(
            trial[blocker_idx], trial_room_counts, trial_instructor_counts
        )
        _add_assignment(
            trial[idx],
            room.id,
            timeslot.id,
            trial_room_counts,
            trial_instructor_counts,
        )

        blocker_choices = _zero_cost_candidates(
            blocker_idx,
            trial,
            sections,
            option_cache,
            trial_room_counts,
            trial_instructor_counts,
            static_memo,
            DIRECT_RESCUE_BLOCKER_SAMPLES,
            top_k=20,
        )
        if not blocker_choices:
            continue

        _, blocker_room, blocker_timeslot = random.choice(blocker_choices[:5])
        _add_assignment(
            trial[blocker_idx],
            blocker_room.id,
            blocker_timeslot.id if blocker_timeslot is not None else None,
            trial_room_counts,
            trial_instructor_counts,
        )
        return trial

    return None


def _direct_rescue_unscheduled(
    schedule,
    sections,
    option_cache,
    static_memo,
):
    """Prioritize reducing unscheduled sections without adding conflicts."""
    best = clone_schedule(schedule)

    for _ in range(DIRECT_RESCUE_ROUNDS):
        unscheduled = [
            idx
            for idx, item in enumerate(best)
            if not _is_scheduled(item)
            and _requirement(idx, option_cache) != "NEEDS_NOTHING"
        ]
        if not unscheduled:
            break

        unscheduled.sort(
            key=lambda idx: (
                _domain_size(idx, option_cache),
                -getattr(sections[idx], "capacity", 0),
            )
        )

        improved = False
        for idx in unscheduled:
            candidate = _try_single_blocker_rescue(
                best,
                idx,
                sections,
                option_cache,
                static_memo,
            )
            if candidate is None:
                continue

            # Accept only when the unscheduled count decreases and the total
            # number of hard conflicts does not increase.
            old_room_counts, old_instructor_counts = _build_occupancy(best)
            new_room_counts, new_instructor_counts = _build_occupancy(candidate)
            old_conflicts = sum(_conflict_totals(
                old_room_counts, old_instructor_counts
            ))
            new_conflicts = sum(_conflict_totals(
                new_room_counts, new_instructor_counts
            ))

            if (
                count_unscheduled(candidate) < count_unscheduled(best)
                and new_conflicts <= old_conflicts
            ):
                best = candidate
                improved = True

        if not improved:
            break

    return best


# ============================================================
# EJECTION-CHAIN REPAIR
# ============================================================
def _blocking_indices(schedule, moving_idx, room_id, timeslot_id):
    """Return sections that prevent moving_idx from using this placement."""
    moving_item = schedule[moving_idx]
    blockers = []

    for idx, item in enumerate(schedule):
        if idx == moving_idx or item.timeslot_id is None:
            continue

        room_block = (
            item.room_id == room_id
            and item.timeslot_id == timeslot_id
        )
        instructor_block = (
            moving_item.instructor_id is not None
            and item.instructor_id == moving_item.instructor_id
            and item.timeslot_id == timeslot_id
        )

        if room_block or instructor_block:
            blockers.append(idx)

    return blockers


def _try_ejection_chain(
    schedule,
    idx,
    sections,
    option_cache,
    static_memo,
    depth,
    visited,
):
    """
    Try to place one section. If its desired placement is occupied, move the
    blocking section(s) recursively. Returns a repaired schedule or None.
    """
    if idx in visited:
        return None

    requirement = _requirement(idx, option_cache)
    if requirement == "NEEDS_NOTHING":
        return clone_schedule(schedule)

    base = clone_schedule(schedule)
    room_counts, instructor_counts = _build_occupancy(base)
    _remove_assignment(base[idx], room_counts, instructor_counts)

    viable_rooms = option_cache[idx]["rooms"]
    if not viable_rooms:
        return None

    # Room-only activities do not use a timeslot, so any viable room works.
    if requirement == "NEEDS_ROOM_ONLY":
        chosen_room = random.choice(viable_rooms)
        _add_assignment(
            base[idx], chosen_room.id, None, room_counts, instructor_counts
        )
        return base

    candidates = _best_candidates(
        idx,
        base,
        sections,
        option_cache,
        room_counts,
        instructor_counts,
        static_memo,
        EJECTION_PAIR_SAMPLES,
        top_k=EJECTION_TOP_K,
    )
    if not candidates:
        return None

    ranked = []
    for cost, room, timeslot in candidates:
        blockers = _blocking_indices(
            base, idx, room.id, timeslot.id
        )
        ranked.append(
            (len(blockers), cost, random.random(), room, timeslot, blockers)
        )

    ranked.sort(key=lambda value: (value[0], value[1], value[2]))

    for blocker_count, _, _, room, timeslot, blockers in ranked:
        # Best case: the placement is already conflict-free.
        if blocker_count == 0:
            result = clone_schedule(base)
            result_room_counts, result_instructor_counts = _build_occupancy(result)
            _add_assignment(
                result[idx],
                room.id,
                timeslot.id,
                result_room_counts,
                result_instructor_counts,
            )
            return result

        if depth <= 0 or blocker_count > EJECTION_MAX_BLOCKERS:
            continue

        # Never eject a section that is already part of this repair chain.
        if any(blocker in visited for blocker in blockers):
            continue

        trial = clone_schedule(base)
        trial_room_counts, trial_instructor_counts = _build_occupancy(trial)

        # Temporarily remove the blockers and reserve the desired placement.
        for blocker in blockers:
            _remove_assignment(
                trial[blocker], trial_room_counts, trial_instructor_counts
            )

        _add_assignment(
            trial[idx],
            room.id,
            timeslot.id,
            trial_room_counts,
            trial_instructor_counts,
        )

        repaired = trial
        chain_ok = True
        next_visited = set(visited)
        next_visited.add(idx)

        # Move the most constrained blocker first.
        blockers = sorted(
            blockers,
            key=lambda blocker_idx: _domain_size(blocker_idx, option_cache),
        )

        for blocker in blockers:
            repaired = _try_ejection_chain(
                repaired,
                blocker,
                sections,
                option_cache,
                static_memo,
                depth - 1,
                next_visited,
            )
            if repaired is None:
                chain_ok = False
                break
            next_visited.add(blocker)

        if chain_ok:
            return repaired

    return None


def _ejection_chain_repair(
    schedule,
    sections,
    option_cache,
    static_memo,
):
    """Repair unscheduled sections using short recursive relocation chains."""
    best = clone_schedule(schedule)
    best_cost = _objective(best)

    for _ in range(EJECTION_REPAIR_ROUNDS):
        unscheduled = [
            idx
            for idx, item in enumerate(best)
            if not _is_scheduled(item)
            and _requirement(idx, option_cache) != "NEEDS_NOTHING"
        ]
        if not unscheduled:
            break

        # Hardest sections first gives flexible sections fewer chances to take
        # the scarce placements needed by constrained sections.
        unscheduled.sort(
            key=lambda idx: (
                _domain_size(idx, option_cache),
                -getattr(sections[idx], "capacity", 0),
            )
        )

        improved_this_round = False

        for idx in unscheduled:
            candidate = _try_ejection_chain(
                best,
                idx,
                sections,
                option_cache,
                static_memo,
                EJECTION_MAX_DEPTH,
                set(),
            )
            if candidate is None:
                continue

            candidate_cost = _objective(candidate)
            if candidate_cost < best_cost:
                best = candidate
                best_cost = candidate_cost
                improved_this_round = True

        if not improved_this_round:
            break

    return best


# ============================================================
# SOFT-CONSTRAINT POLISH
# ============================================================
def _soft_polish(
    schedule,
    sections,
    rooms,
    timeslots,
    cache,
    option_cache,
    static_memo,
):
    best = clone_schedule(schedule)
    hard_cost = _objective(best)
    best_fitness = calculate_fitness(
        best,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=cache,
    )

    movable_indices = [
        idx
        for idx in range(len(best))
        if _requirement(idx, option_cache) == "NEEDS_ROOM_AND_TIME"
    ]
    if not movable_indices:
        return best

    for _ in range(SOFT_POLISH_STEPS):
        idx = random.choice(movable_indices)
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
            SOFT_POLISH_PAIR_SAMPLES,
            top_k=10,
        )
        choices = [choice for choice in choices if choice[0] == 0]
        if not choices:
            continue

        _, room, timeslot = random.choice(choices)
        _add_assignment(
            candidate[idx],
            room.id,
            timeslot.id,
            room_counts,
            instructor_counts,
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
# PUBLIC FUNCTIONS EXPECTED BY THE WEBSITE
# ============================================================
def genetic_schedule(sections, timeslots, rooms, cache=None, option_cache=None):
    """
    Retains the historical function name used by the project, but runs a
    classification-aware MRV + LNS + min-conflicts hybrid.
    """
    if cache is None:
        cache = build_timeslot_guideline_cache(sections, timeslots)

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    static_memo = {}

    best = _initial_construction(sections, option_cache, static_memo)
    best = _direct_rescue_unscheduled(
        best, sections, option_cache, static_memo
    )
    best = _large_neighborhood_search(
        best, sections, option_cache, static_memo
    )
    best = _min_conflicts(
        best, sections, option_cache, static_memo
    )
    best = _direct_rescue_unscheduled(
        best, sections, option_cache, static_memo
    )
    best = _large_neighborhood_search(
        best, sections, option_cache, static_memo
    )
    best = _min_conflicts(
        best, sections, option_cache, static_memo
    )
    best = _ejection_chain_repair(
        best, sections, option_cache, static_memo
    )
    best = _direct_rescue_unscheduled(
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

    return best


def genetic_runs(sections, timeslots, rooms, num_runs=1):
    """
    Wrapper required by engine.py. Runs the hybrid one or more times and
    returns the best schedule. No other project file needs to change.
    """
    run_count = max(1, int(num_runs or 1))
    cache = build_timeslot_guideline_cache(sections, timeslots)
    option_cache = build_option_cache(sections, rooms, timeslots)

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

        candidate_objective = _objective(candidate)
        candidate_fitness = calculate_fitness(
            candidate,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        # Hard feasibility first; fitness breaks ties.
        candidate_key = (candidate_objective, -candidate_fitness)
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_schedule = clone_schedule(candidate)

    return best_schedule