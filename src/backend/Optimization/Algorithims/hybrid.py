import random
import sys

from backend.Optimization.constraints import (
    get_valid_timeslots,
    passes_hard_constraints,
    get_viable_rooms,
    get_viable_rooms_for_schedule_item,
    classify_section,
)

from backend.models.models import ScheduleItem

from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
)

# =========================
# PARAMETERS
# =========================
POPULATION_SIZE = 20
GENERATIONS = 25

CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.15  # gate for "nudge an already-scheduled item"; unscheduled
                       # items now bypass this gate entirely (see mutation()).

TABU_ITERATIONS = 12
TABU_TENURE = 7

TS_APPLY_RATE = 0.3  # top 30% of population gets tabu search

ELITE_SIZE = 2

# Quick-pass sample width. Kept moderate (not exhaustive) because this
# runs for EVERY item, EVERY child, EVERY generation -- it's the hot
# path. Items that fail this quick pass fall through to an exhaustive
# search (see _exhaustive_best_slot) rather than another sampled pass,
# so thoroughness for hard-to-place items no longer depends on this
# number being huge.
MAX_SEARCH_WIDTH = 30

# CHANGED: this used to hard-cap the final rescue pass at 50 leftover
# items -- meaning if more than 50 sections were unscheduled, the ONE
# exhaustive fallback pass silently never ran at all, and those
# sections stayed unscheduled forever regardless of how many
# generations you ran. Raised substantially, and the skip (if it ever
# happens) now prints a warning instead of failing silently.
RESCUE_MAX_ITEMS = 1000

# CSP construction controls
CSP_NODE_BUDGET = 180_000
CSP_BACKTRACK_LIMIT = 2_000
CSP_MRV_SAMPLE_SIZE = 80
CSP_VALUE_SAMPLE_SIZE = 120
CSP_RESTARTS = 1

# Final soft-constraint polish
HILL_CLIMB_ITERATIONS = 250
HILL_CLIMB_NEIGHBORS = 12


# =========================
# CLONE HELPERS
# =========================
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


def count_unscheduled(schedule):
    return sum(
        1 for item in schedule
        if item.room_id is None or item.timeslot_id is None
    )


# =========================
# OPTION CACHE
# =========================
_ROOMS_BY_ID_KEY = "_rooms_by_id"
_TIMESLOTS_BY_ID_KEY = "_timeslots_by_id"


def build_option_cache(sections, rooms, timeslots):
    cache = {
        _ROOMS_BY_ID_KEY: {r.id: r for r in rooms},
        _TIMESLOTS_BY_ID_KEY: {t.id: t for t in timeslots},
    }

    for idx, section in enumerate(sections):
        viable_rooms = get_viable_rooms(section, rooms)
        valid_timeslots = get_valid_timeslots(section, timeslots)
        cache[idx] = (viable_rooms, valid_timeslots)

    return cache


def _get_id_maps(option_cache, rooms, timeslots):
    if option_cache and _ROOMS_BY_ID_KEY in option_cache:
        rooms_by_id = option_cache[_ROOMS_BY_ID_KEY]
        timeslots_by_id = option_cache[_TIMESLOTS_BY_ID_KEY]
    else:
        rooms_by_id = {r.id: r for r in rooms}
        timeslots_by_id = {t.id: t for t in timeslots}
    return rooms_by_id, timeslots_by_id


def _constraint_score(idx, option_cache):
    if option_cache and idx in option_cache:
        viable_rooms, valid_timeslots = option_cache[idx]
        if viable_rooms and valid_timeslots:
            return len(viable_rooms) * len(valid_timeslots)
        return 0
    return 1_000_000


# =========================
# EXHAUSTIVE FALLBACK SEARCH
# =========================
def _exhaustive_best_slot(item, section, viable_rooms, valid_timeslots, occupied_instructors, occupied_rooms):
    """
    Full scan over every (room, timeslot) combo for `item`, looking for a
    genuinely conflict-free placement and falling back to the least-bad
    conflict count if none exists. Early-exits the instant a zero-conflict
    combo is found, so the full cost is only ever paid for items that are
    truly hard to place -- easy items get resolved by a cheap sampled pass
    before this ever runs. This is what the sampled fallbacks used to do
    with a bounded random sample (and could therefore miss a real free
    slot that existed outside the sample); this always finds one if it
    exists.
    """
    best_room, best_ts, best_conflicts = None, None, None

    for ts in valid_timeslots:
        instr_conflict = (
            item.instructor_id is not None
            and (item.instructor_id, ts.id) in occupied_instructors
        )
        for room in viable_rooms:
            if section is not None and not passes_hard_constraints(
                section, room, ts,
                occupied_instructors=occupied_instructors,
                occupied_rooms=occupied_rooms,
            ):
                continue

            conflicts = (1 if instr_conflict else 0) + (
                1 if (room.id, ts.id) in occupied_rooms else 0
            )

            if conflicts == 0:
                return room, ts, 0

            if best_conflicts is None or conflicts < best_conflicts:
                best_room, best_ts, best_conflicts = room, ts, conflicts

    return best_room, best_ts, best_conflicts


# =========================
# CSP CONSTRUCTION: MRV + FORWARD CHECKING
# =========================
def _new_empty_schedule(sections):
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


def _iter_feasible_values(
    idx,
    schedule,
    sections,
    option_cache,
    occupied_instructors,
    occupied_rooms,
    randomized=True,
    limit=None,
):
    """Yield valid (room, timeslot) values lazily to avoid huge domains."""
    item = schedule[idx]
    section = sections[idx]
    viable_rooms, valid_timeslots = option_cache[idx]

    timeslot_order = list(valid_timeslots)
    room_order = list(viable_rooms)
    if randomized:
        random.shuffle(timeslot_order)
        random.shuffle(room_order)

    yielded = 0
    for ts in timeslot_order:
        if (
            item.instructor_id is not None
            and (item.instructor_id, ts.id) in occupied_instructors
        ):
            continue

        for room in room_order:
            if (room.id, ts.id) in occupied_rooms:
                continue
            if not passes_hard_constraints(
                section,
                room,
                ts,
                occupied_instructors=occupied_instructors,
                occupied_rooms=occupied_rooms,
            ):
                continue

            yield room, ts
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _remaining_value_count(
    idx,
    schedule,
    sections,
    option_cache,
    occupied_instructors,
    occupied_rooms,
    stop_after=None,
):
    count = 0
    for _ in _iter_feasible_values(
        idx,
        schedule,
        sections,
        option_cache,
        occupied_instructors,
        occupied_rooms,
        randomized=False,
        limit=stop_after,
    ):
        count += 1
    return count


def _choose_mrv_index(
    unassigned,
    schedule,
    sections,
    option_cache,
    occupied_instructors,
    occupied_rooms,
):
    """Choose the section with the smallest current feasible domain."""
    if len(unassigned) <= CSP_MRV_SAMPLE_SIZE:
        candidates = list(unassigned)
    else:
        base = sorted(unassigned, key=lambda i: _constraint_score(i, option_cache))
        scarce = base[: CSP_MRV_SAMPLE_SIZE // 2]
        rest = base[CSP_MRV_SAMPLE_SIZE // 2 :]
        sampled = random.sample(
            rest,
            min(CSP_MRV_SAMPLE_SIZE - len(scarce), len(rest)),
        )
        candidates = scarce + sampled

    best_idx = None
    best_count = None
    for idx in candidates:
        cutoff = best_count if best_count is not None else None
        count = _remaining_value_count(
            idx,
            schedule,
            sections,
            option_cache,
            occupied_instructors,
            occupied_rooms,
            stop_after=cutoff,
        )
        if count == 0:
            return idx, 0
        if best_count is None or count < best_count:
            best_idx, best_count = idx, count

    return best_idx, best_count


def _forward_check(
    placed_idx,
    unassigned,
    schedule,
    sections,
    option_cache,
    occupied_instructors,
    occupied_rooms,
):
    """
    Check sections most affected by the latest assignment. Same-instructor
    sections are always checked; the currently scarcest sections are checked
    too. A zero-size domain causes immediate backtracking.
    """
    placed_instructor = schedule[placed_idx].instructor_id
    affected = []

    for idx in unassigned:
        if placed_instructor is not None and schedule[idx].instructor_id == placed_instructor:
            affected.append(idx)

    scarce = sorted(unassigned, key=lambda i: _constraint_score(i, option_cache))
    for idx in scarce[:CSP_MRV_SAMPLE_SIZE]:
        if idx not in affected:
            affected.append(idx)

    for idx in affected:
        if _remaining_value_count(
            idx,
            schedule,
            sections,
            option_cache,
            occupied_instructors,
            occupied_rooms,
            stop_after=1,
        ) == 0:
            return False
    return True


def csp_construct_schedule(
    sections,
    rooms,
    timeslots,
    option_cache=None,
    node_budget=CSP_NODE_BUDGET,
):
    """Build a conflict-free partial schedule using MRV and forward checking."""
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    sys.setrecursionlimit(max(10_000, len(sections) + 1_000))
    schedule = _new_empty_schedule(sections)

    # Only sections that truly need both a room and a timeslot belong in
    # the CSP search. Sections classified as NEEDS_NOTHING are already
    # complete with both values left as None. NEEDS_ROOM_ONLY sections
    # receive a viable room here and are also excluded from the timeslot
    # search. This keeps hybrid.py compatible with evaluation.py without
    # changing evaluation.py.
    unassigned = set()
    for idx, section in enumerate(sections):
        requirement = classify_section(section)

        if requirement == "NEEDS_NOTHING":
            continue

        if requirement == "NEEDS_ROOM_ONLY":
            viable_rooms, _ = option_cache[idx]
            if viable_rooms:
                schedule[idx].room_id = viable_rooms[0].id
            continue

        unassigned.add(idx)

    occupied_instructors = set()
    occupied_rooms = set()
    nodes = 0
    backtracks = 0
    best_snapshot = clone_schedule(schedule)
    best_assigned = 0

    def search():
        nonlocal nodes, backtracks, best_snapshot, best_assigned

        assigned = len(schedule) - len(unassigned)
        if assigned > best_assigned:
            best_assigned = assigned
            best_snapshot = clone_schedule(schedule)

        if not unassigned:
            return True
        if nodes >= node_budget or backtracks >= CSP_BACKTRACK_LIMIT:
            return False

        idx, domain_count = _choose_mrv_index(
            unassigned,
            schedule,
            sections,
            option_cache,
            occupied_instructors,
            occupied_rooms,
        )
        if idx is None or domain_count == 0:
            backtracks += 1
            return False

        values = list(_iter_feasible_values(
            idx,
            schedule,
            sections,
            option_cache,
            occupied_instructors,
            occupied_rooms,
            randomized=True,
            limit=CSP_VALUE_SAMPLE_SIZE,
        ))

        # Least-constraining-value approximation: prefer less-used timeslots.
        timeslot_load = {}
        for _, ts_id in occupied_rooms:
            timeslot_load[ts_id] = timeslot_load.get(ts_id, 0) + 1
        values.sort(key=lambda pair: timeslot_load.get(pair[1].id, 0))

        item = schedule[idx]
        unassigned.remove(idx)

        for room, ts in values:
            nodes += 1
            item.room_id = room.id
            item.timeslot_id = ts.id

            instr_key = None
            if item.instructor_id is not None:
                instr_key = (item.instructor_id, ts.id)
                occupied_instructors.add(instr_key)
            room_key = (room.id, ts.id)
            occupied_rooms.add(room_key)

            forward_ok = _forward_check(
                idx,
                unassigned,
                schedule,
                sections,
                option_cache,
                occupied_instructors,
                occupied_rooms,
            )

            if forward_ok and search():
                return True

            occupied_rooms.remove(room_key)
            if instr_key is not None:
                occupied_instructors.remove(instr_key)
            item.room_id = None
            item.timeslot_id = None

            if nodes >= node_budget or backtracks >= CSP_BACKTRACK_LIMIT:
                break

        unassigned.add(idx)
        backtracks += 1
        return False

    search()
    return best_snapshot


def hill_climb_polish(
    schedule,
    sections,
    rooms,
    timeslots,
    cache,
    option_cache=None,
):
    """Improve soft fitness while accepting only strictly better valid moves."""
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    current = clone_schedule(schedule)
    current_score = calculate_fitness(
        current,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=cache,
    )

    for _ in range(HILL_CLIMB_ITERATIONS):
        idx = random.randrange(len(current))
        item = current[idx]
        section = sections[idx]

        occupied_instructors = set()
        occupied_rooms = set()
        for j, other in enumerate(current):
            if j == idx or other.timeslot_id is None:
                continue
            if other.instructor_id is not None:
                occupied_instructors.add((other.instructor_id, other.timeslot_id))
            if other.room_id is not None:
                occupied_rooms.add((other.room_id, other.timeslot_id))

        viable_rooms, valid_timeslots = option_cache[idx]
        candidates = list(_iter_feasible_values(
            idx,
            current,
            sections,
            option_cache,
            occupied_instructors,
            occupied_rooms,
            randomized=True,
            limit=HILL_CLIMB_NEIGHBORS,
        ))

        old_room, old_ts = item.room_id, item.timeslot_id
        best_move = None
        best_score = current_score

        for room, ts in candidates:
            if room.id == old_room and ts.id == old_ts:
                continue
            item.room_id, item.timeslot_id = room.id, ts.id
            score = calculate_fitness(
                current,
                rooms,
                sections=sections,
                timeslots=timeslots,
                valid_timeslot_cache=cache,
            )
            if score > best_score:
                best_score = score
                best_move = (room.id, ts.id)

        if best_move is None:
            item.room_id, item.timeslot_id = old_room, old_ts
        else:
            item.room_id, item.timeslot_id = best_move
            current_score = best_score

    return current


# =========================
# CREATE POPULATION
# =========================
def create_population(size, sections, rooms, timeslots, option_cache=None):
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    population = []
    for _ in range(size):
        schedule = csp_construct_schedule(
            sections,
            rooms,
            timeslots,
            option_cache=option_cache,
        )
        population.append(schedule)
    return population


# =========================
# REPAIR
# =========================
def repair_schedule(schedule, sections, rooms, timeslots, option_cache=None):
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    rooms_by_id, timeslots_by_id = _get_id_maps(option_cache, rooms, timeslots)

    occupied_instructors = set()
    occupied_rooms = set()

    order = sorted(
        range(len(schedule)),
        key=lambda i: (_constraint_score(i, option_cache), random.random()),
    )

    for idx in order:

        item = schedule[idx]
        section = sections[idx] if idx < len(sections) else None

        is_unset = item.room_id is None or item.timeslot_id is None

        current_conflicts = 0
        hard_ok = True

        if not is_unset:
            if (
                    item.instructor_id is not None
                    and (item.instructor_id, item.timeslot_id) in occupied_instructors
            ):
                current_conflicts += 1
            if (item.room_id, item.timeslot_id) in occupied_rooms:
                current_conflicts += 1

            if section is not None:
                room_obj = rooms_by_id.get(item.room_id)
                ts_obj = timeslots_by_id.get(item.timeslot_id)
                if room_obj is not None and ts_obj is not None:
                    if not passes_hard_constraints(
                            section,
                            room_obj,
                            ts_obj,
                            occupied_instructors=occupied_instructors,
                            occupied_rooms=occupied_rooms,
                    ):
                        hard_ok = False

        if not is_unset and current_conflicts == 0 and hard_ok:
            if item.instructor_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )
            occupied_rooms.add((item.room_id, item.timeslot_id))
            continue

        if idx in option_cache:
            viable_rooms, valid_timeslots = option_cache[idx]
        elif section is not None:
            viable_rooms = get_viable_rooms(section, rooms)
            valid_timeslots = get_valid_timeslots(section, timeslots)
        else:
            viable_rooms = get_viable_rooms_for_schedule_item(item, rooms)
            valid_timeslots = timeslots

        if not viable_rooms or not valid_timeslots:
            if not is_unset:
                if item.instructor_id is not None:
                    occupied_instructors.add(
                        (item.instructor_id, item.timeslot_id)
                    )
                occupied_rooms.add((item.room_id, item.timeslot_id))
            continue

        search_timeslots = random.sample(valid_timeslots, min(MAX_SEARCH_WIDTH, len(valid_timeslots)))

        best_room, best_ts, best_conflicts = None, None, None

        for ts in search_timeslots:

            if (
                    item.instructor_id is not None
                    and (item.instructor_id, ts.id) in occupied_instructors
            ):
                continue

            search_rooms = random.sample(viable_rooms, min(MAX_SEARCH_WIDTH, len(viable_rooms)))

            for room in search_rooms:

                if (room.id, ts.id) in occupied_rooms:
                    continue

                if section is not None and not passes_hard_constraints(
                        section,
                        room,
                        ts,
                        occupied_instructors=occupied_instructors,
                        occupied_rooms=occupied_rooms,
                ):
                    continue

                best_room, best_ts, best_conflicts = room, ts, 0
                break

            if best_room is not None:
                break

        if best_room is None:
            # CHANGED: this used to be another *sampled* pass looking for
            # the least-bad conflict, which could still miss a genuinely
            # free slot sitting outside the random sample. Now it's a
            # full exhaustive scan -- only paid for items that already
            # failed the cheap sampled pass above, which should be the
            # minority, so the extra cost is proportional to how many
            # sections are actually hard to place.
            best_room, best_ts, best_conflicts = _exhaustive_best_slot(
                item, section, viable_rooms, valid_timeslots,
                occupied_instructors, occupied_rooms,
            )

        should_replace = (
                is_unset
                or not hard_ok
                or (best_conflicts is not None and best_conflicts < current_conflicts)
        )

        if best_room is not None and should_replace:
            item.room_id = best_room.id
            item.timeslot_id = best_ts.id

        if item.room_id is not None and item.timeslot_id is not None:
            if item.instructor_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )
            occupied_rooms.add((item.room_id, item.timeslot_id))

    return schedule


# =========================
# RESCUE PASS FOR LEFTOVER UNSCHEDULED ITEMS
# =========================
def force_place_remaining(schedule, sections, rooms, timeslots, option_cache=None):
    unscheduled = [
        i for i, item in enumerate(schedule)
        if item.room_id is None or item.timeslot_id is None
    ]

    if not unscheduled:
        return schedule

    if len(unscheduled) > RESCUE_MAX_ITEMS:
        # CHANGED: this used to skip the rescue pass completely and
        # silently, which is exactly the wrong behavior when there are
        # a lot of unscheduled items -- that's when this pass matters
        # most. Now it always runs, just with a visible warning so you
        # know it's doing a heavier pass.
        print(
            f"[force_place_remaining] {len(unscheduled)} unscheduled items "
            f"exceeds RESCUE_MAX_ITEMS ({RESCUE_MAX_ITEMS}) -- running anyway, "
            f"this may take a while. Raise RESCUE_MAX_ITEMS to silence this "
            f"once you've confirmed the runtime is acceptable."
        )

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    occupied_instructors = set()
    occupied_rooms = set()

    for item in schedule:
        if item.room_id is not None and item.timeslot_id is not None:
            if item.instructor_id is not None:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))
            occupied_rooms.add((item.room_id, item.timeslot_id))

    # most-constrained-first here too, same reasoning as repair_schedule:
    # scarce sections should get first crack at whatever's still free.
    unscheduled.sort(key=lambda i: _constraint_score(i, option_cache))

    for idx in unscheduled:

        item = schedule[idx]
        section = sections[idx] if idx < len(sections) else None

        if idx in option_cache:
            viable_rooms, valid_timeslots = option_cache[idx]
        elif section is not None:
            viable_rooms = get_viable_rooms(section, rooms)
            valid_timeslots = get_valid_timeslots(section, timeslots)
        else:
            viable_rooms = get_viable_rooms_for_schedule_item(item, rooms)
            valid_timeslots = timeslots

        if not viable_rooms or not valid_timeslots:
            continue

        best_room, best_ts, best_conflicts = _exhaustive_best_slot(
            item, section, viable_rooms, valid_timeslots,
            occupied_instructors, occupied_rooms,
        )

        if best_room is not None:
            item.room_id = best_room.id
            item.timeslot_id = best_ts.id
            if item.instructor_id is not None:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))
            occupied_rooms.add((item.room_id, item.timeslot_id))

    return schedule


# =========================
# CONFLICT RESOLUTION PASS
# =========================
def resolve_conflicts(schedule, sections, rooms, timeslots, option_cache=None):
    """
    force_place_remaining fixes items with NO room/timeslot at all. This
    fixes the other kind of leftover problem: items that DO have a room
    and timeslot, but it's a double-booked one (the "room conflicts" you
    were seeing). For each conflicted item, it rebuilds the occupancy
    picture excluding that item, then does an exhaustive search for a
    genuinely free slot -- only moving the item if it actually finds a
    strictly better (lower-conflict) placement.
    """
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    rooms_by_id, timeslots_by_id = _get_id_maps(option_cache, rooms, timeslots)

    room_counts = {}
    instr_counts = {}
    for item in schedule:
        if item.room_id is not None and item.timeslot_id is not None:
            key = (item.room_id, item.timeslot_id)
            room_counts[key] = room_counts.get(key, 0) + 1
        if item.instructor_id is not None and item.timeslot_id is not None:
            key = (item.instructor_id, item.timeslot_id)
            instr_counts[key] = instr_counts.get(key, 0) + 1

    conflicted_indices = []
    for idx, item in enumerate(schedule):
        if item.room_id is None or item.timeslot_id is None:
            continue
        room_conflict = room_counts.get((item.room_id, item.timeslot_id), 0) > 1
        instr_conflict = (
            item.instructor_id is not None
            and instr_counts.get((item.instructor_id, item.timeslot_id), 0) > 1
        )
        if room_conflict or instr_conflict:
            conflicted_indices.append(idx)

    if not conflicted_indices:
        return schedule

    # most-constrained-first, same reasoning as everywhere else: give
    # scarce sections first crack at whatever opens up
    conflicted_indices.sort(key=lambda i: _constraint_score(i, option_cache))

    for idx in conflicted_indices:
        item = schedule[idx]
        section = sections[idx] if idx < len(sections) else None

        # rebuild occupancy excluding this item specifically, so it
        # doesn't block itself while we search for somewhere better
        occupied_instructors = set()
        occupied_rooms = set()
        for j, other in enumerate(schedule):
            if j == idx:
                continue
            if other.instructor_id is not None and other.timeslot_id is not None:
                occupied_instructors.add((other.instructor_id, other.timeslot_id))
            if other.room_id is not None and other.timeslot_id is not None:
                occupied_rooms.add((other.room_id, other.timeslot_id))

        if idx in option_cache:
            viable_rooms, valid_timeslots = option_cache[idx]
        elif section is not None:
            viable_rooms = get_viable_rooms(section, rooms)
            valid_timeslots = get_valid_timeslots(section, timeslots)
        else:
            viable_rooms = get_viable_rooms_for_schedule_item(item, rooms)
            valid_timeslots = timeslots

        if not viable_rooms or not valid_timeslots:
            continue

        current_conflicts = 0
        if item.instructor_id is not None and (item.instructor_id, item.timeslot_id) in occupied_instructors:
            current_conflicts += 1
        if (item.room_id, item.timeslot_id) in occupied_rooms:
            current_conflicts += 1

        best_room, best_ts, best_conflicts = _exhaustive_best_slot(
            item, section, viable_rooms, valid_timeslots,
            occupied_instructors, occupied_rooms,
        )

        if best_room is not None and (best_conflicts is None or best_conflicts < current_conflicts):
            item.room_id = best_room.id
            item.timeslot_id = best_ts.id

    return schedule


# =========================
# EJECTION-CHAIN RESOLUTION (for mutual blocking)
# =========================
def _find_problem_indices(schedule):
    room_counts = {}
    instr_counts = {}
    for item in schedule:
        if item.room_id is not None and item.timeslot_id is not None:
            key = (item.room_id, item.timeslot_id)
            room_counts[key] = room_counts.get(key, 0) + 1
        if item.instructor_id is not None and item.timeslot_id is not None:
            key = (item.instructor_id, item.timeslot_id)
            instr_counts[key] = instr_counts.get(key, 0) + 1

    probs = []
    for idx, item in enumerate(schedule):
        if item.room_id is None or item.timeslot_id is None:
            probs.append(idx)
            continue
        room_conflict = room_counts.get((item.room_id, item.timeslot_id), 0) > 1
        instr_conflict = (
            item.instructor_id is not None
            and instr_counts.get((item.instructor_id, item.timeslot_id), 0) > 1
        )
        if room_conflict or instr_conflict:
            probs.append(idx)
    return probs


def resolve_hard_cases(schedule, sections, rooms, timeslots, option_cache=None, max_candidates_per_item=8):
    """
    resolve_conflicts and force_place_remaining only ever move ONE item at
    a time. That can't fix mutual blocking: section A's best slot is held
    by section B, but B can't be evicted by a single-item search because
    nowhere else looks better to B in isolation -- even though B actually
    has plenty of other fine options and moving it would free up exactly
    what A needs.

    This does a depth-1 ejection chain: for each still-broken item, look
    at who's occupying a slot it wants, tentatively evict them, and check
    whether THEY can be placed somewhere else with zero conflicts. If so,
    commit both moves. If not, undo and try the next candidate. Only ever
    commits a chain if the evicted item lands somewhere completely clean
    -- it never trades one problem for another.
    """
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    problem_indices = _find_problem_indices(schedule)
    if not problem_indices:
        return schedule, 0

    problem_indices.sort(key=lambda i: _constraint_score(i, option_cache))
    resolved_count = 0

    for p_idx in problem_indices:
        p_item = schedule[p_idx]
        p_section = sections[p_idx] if p_idx < len(sections) else None

        if p_idx in option_cache:
            p_rooms, p_timeslots = option_cache[p_idx]
        elif p_section is not None:
            p_rooms = get_viable_rooms(p_section, rooms)
            p_timeslots = get_valid_timeslots(p_section, timeslots)
        else:
            p_rooms = get_viable_rooms_for_schedule_item(p_item, rooms)
            p_timeslots = timeslots

        if not p_rooms or not p_timeslots:
            continue

        occupant_of = {}
        for k, item in enumerate(schedule):
            if k == p_idx:
                continue
            if item.room_id is not None and item.timeslot_id is not None:
                occupant_of[(item.room_id, item.timeslot_id)] = k

        candidates = []
        for room in p_rooms:
            for ts in p_timeslots:
                q_idx = occupant_of.get((room.id, ts.id))
                if q_idx is not None:
                    candidates.append((room, ts, q_idx))

        if not candidates:
            continue

        random.shuffle(candidates)
        candidates = candidates[: max_candidates_per_item * 3]

        tried = 0
        for room, ts, q_idx in candidates:
            if tried >= max_candidates_per_item:
                break
            tried += 1

            q_item = schedule[q_idx]
            q_section = sections[q_idx] if q_idx < len(sections) else None

            occ_instr = set()
            occ_room = set()
            for k, item in enumerate(schedule):
                if k in (p_idx, q_idx):
                    continue
                if item.instructor_id is not None and item.timeslot_id is not None:
                    occ_instr.add((item.instructor_id, item.timeslot_id))
                if item.room_id is not None and item.timeslot_id is not None:
                    occ_room.add((item.room_id, item.timeslot_id))

            p_instr_conflict = (
                p_item.instructor_id is not None
                and (p_item.instructor_id, ts.id) in occ_instr
            )
            if p_instr_conflict:
                continue
            if p_section is not None and not passes_hard_constraints(
                p_section, room, ts, occupied_instructors=occ_instr, occupied_rooms=occ_room
            ):
                continue

            occ_instr_for_q = set(occ_instr)
            occ_room_for_q = set(occ_room)
            if p_item.instructor_id is not None:
                occ_instr_for_q.add((p_item.instructor_id, ts.id))
            occ_room_for_q.add((room.id, ts.id))

            if q_idx in option_cache:
                q_rooms, q_timeslots = option_cache[q_idx]
            elif q_section is not None:
                q_rooms = get_viable_rooms(q_section, rooms)
                q_timeslots = get_valid_timeslots(q_section, timeslots)
            else:
                q_rooms = get_viable_rooms_for_schedule_item(q_item, rooms)
                q_timeslots = timeslots

            if not q_rooms or not q_timeslots:
                continue

            new_q_room, new_q_ts, new_q_conflicts = _exhaustive_best_slot(
                q_item, q_section, q_rooms, q_timeslots, occ_instr_for_q, occ_room_for_q
            )

            if new_q_room is not None and new_q_conflicts == 0:
                p_item.room_id, p_item.timeslot_id = room.id, ts.id
                q_item.room_id, q_item.timeslot_id = new_q_room.id, new_q_ts.id
                resolved_count += 1
                break

    return schedule, resolved_count


def diagnose_remaining_problems(schedule, sections, option_cache, top_n=15):
    """
    After all the repair passes, whatever's still broken is worth a
    quick look at WHY. If a stuck section has very few viable rooms
    and/or valid timeslots to begin with, no algorithm change can fix
    it -- that's a genuine supply/constraint problem (not enough of the
    right room type, or the instructor/section combination is simply
    over-constrained) and needs a data-level fix, not a code one.
    """
    problem_indices = _find_problem_indices(schedule)
    if not problem_indices:
        print("[diagnose] no remaining unscheduled/conflicted items.")
        return

    problem_indices.sort(key=lambda i: _constraint_score(i, option_cache))

    print(f"\n[diagnose] {len(problem_indices)} items still unscheduled/conflicted.")
    print("[diagnose] most constrained offenders (fewest room x timeslot options):")
    for idx in problem_indices[:top_n]:
        item = schedule[idx]
        section = sections[idx] if idx < len(sections) else None
        if idx in option_cache:
            viable_rooms, valid_timeslots = option_cache[idx]
            n_rooms, n_ts = len(viable_rooms), len(valid_timeslots)
        else:
            n_rooms, n_ts = "?", "?"
        status = "UNSCHEDULED" if (item.room_id is None or item.timeslot_id is None) else "CONFLICT"
        sec_label = getattr(section, "no", item.section) if section else item.section
        print(f"    section {sec_label}: {status}, viable_rooms={n_rooms}, valid_timeslots={n_ts}")


# =========================
# FITNESS SCORING
# =========================
def score_population(population, rooms, sections, timeslots, cache):
    return [
        (
            schedule,
            calculate_fitness(
                schedule,
                rooms,
                sections=sections,
                timeslots=timeslots,
                valid_timeslot_cache=cache,
            ),
        )
        for schedule in population
    ]


# =========================
# SELECTION
# =========================
def tournament_selection(population, fitness_lookup, tournament_size=3):
    competitors = random.sample(
        population,
        min(tournament_size, len(population)),
    )
    return max(competitors, key=lambda s: fitness_lookup[id(s)])


# =========================
# CROSSOVER
# =========================
def crossover(parent1, parent2, option_cache=None):
    if random.random() > CROSSOVER_RATE:
        return clone_schedule(parent1)

    child = [None] * len(parent1)

    occupied_instructors = set()
    occupied_rooms = set()

    order = sorted(
        range(len(parent1)),
        key=lambda i: (_constraint_score(i, option_cache), random.random()),
    )

    for idx in order:

        item1 = parent1[idx]
        item2 = parent2[idx]

        if random.random() < 0.5:
            preferred, other = item1, item2
        else:
            preferred, other = item2, item1

        chosen = None

        for candidate in (preferred, other):

            if candidate.room_id is None or candidate.timeslot_id is None:
                continue

            instructor_conflict = (
                    candidate.instructor_id is not None
                    and (candidate.instructor_id, candidate.timeslot_id)
                    in occupied_instructors
            )

            room_conflict = (
                                candidate.room_id,
                                candidate.timeslot_id,
                            ) in occupied_rooms

            if not instructor_conflict and not room_conflict:
                chosen = candidate
                break

        if chosen is not None:

            new_item = clone_item(chosen)
            child[idx] = new_item

            if new_item.instructor_id is not None:
                occupied_instructors.add(
                    (new_item.instructor_id, new_item.timeslot_id)
                )
            occupied_rooms.add((new_item.room_id, new_item.timeslot_id))

        else:
            new_item = clone_item(preferred)
            new_item.room_id = None
            new_item.timeslot_id = None
            child[idx] = new_item

    return child


# =========================
# MUTATION
# =========================
def mutation(schedule, rooms, timeslots, option_cache=None, sections=None):
    if len(schedule) == 0:
        return schedule, False

    indexed = list(enumerate(schedule))

    unscheduled = [
        (i, item) for i, item in indexed
        if item.room_id is None or item.timeslot_id is None
    ]

    # CHANGED: unscheduled items now always get an attempt, bypassing the
    # MUTATION_RATE gate entirely. Previously, an unscheduled item only got
    # touched by mutation ~15% of the time (once the gate passed AND it was
    # the one item randomly picked) -- meaning most generations, most
    # unscheduled items got zero help from mutation and relied solely on
    # repair_schedule's sampling to close the gap. Scheduled items still
    # respect the normal probability gate so we're not introducing excess
    # churn into an otherwise-fine schedule.
    if not unscheduled:
        if random.random() > MUTATION_RATE:
            return schedule, False
        idx, selected_item = random.choice(indexed)
    else:
        idx, selected_item = random.choice(unscheduled)

    section = sections[idx] if sections and idx < len(sections) else None
    rooms_by_id, timeslots_by_id = _get_id_maps(option_cache, rooms, timeslots)

    occupied_instructors = set()
    occupied_rooms = set()

    for i, item in indexed:

        if i == idx:
            continue

        if item.instructor_id and item.timeslot_id:
            occupied_instructors.add(
                (item.instructor_id, item.timeslot_id)
            )

        if item.room_id and item.timeslot_id:
            occupied_rooms.add(
                (item.room_id, item.timeslot_id)
            )

    if option_cache and idx in option_cache:
        cached_rooms, cached_timeslots = option_cache[idx]
    else:
        cached_rooms = get_viable_rooms_for_schedule_item(selected_item, rooms)
        cached_timeslots = timeslots

    if selected_item.room_id is None or selected_item.timeslot_id is None:

        if cached_rooms and cached_timeslots:

            search_rooms = random.sample(cached_rooms, min(MAX_SEARCH_WIDTH, len(cached_rooms)))
            search_timeslots = random.sample(cached_timeslots, min(MAX_SEARCH_WIDTH, len(cached_timeslots)))

            best_room, best_ts, best_conflicts = None, None, None

            for ts in search_timeslots:

                instr_conflict = (
                        selected_item.instructor_id is not None
                        and (selected_item.instructor_id, ts.id) in occupied_instructors
                )

                for room in search_rooms:

                    if section is not None and not passes_hard_constraints(
                            section,
                            room,
                            ts,
                            occupied_instructors=occupied_instructors,
                            occupied_rooms=occupied_rooms,
                    ):
                        continue

                    conflicts = (1 if instr_conflict else 0) + (
                        1 if (room.id, ts.id) in occupied_rooms else 0
                    )

                    if conflicts == 0:
                        best_room, best_ts, best_conflicts = room, ts, 0
                        break

                    if best_conflicts is None or conflicts < best_conflicts:
                        best_room, best_ts, best_conflicts = room, ts, conflicts

                if best_conflicts == 0:
                    break

            if best_room is None:
                # same reasoning as repair_schedule -- fall through to an
                # exhaustive search rather than giving up after one
                # sampled pass, since this branch specifically targets
                # unscheduled items.
                best_room, best_ts, best_conflicts = _exhaustive_best_slot(
                    selected_item, section, cached_rooms, cached_timeslots,
                    occupied_instructors, occupied_rooms,
                )

            if best_room is not None:
                selected_item.room_id = best_room.id
                selected_item.timeslot_id = best_ts.id
                return schedule, True

        return schedule, False

    changed = False

    if random.random() < 0.5:

        if cached_rooms and selected_item.timeslot_id:

            ts_obj = timeslots_by_id.get(selected_item.timeslot_id)

            for room in random.sample(
                    cached_rooms,
                    min(MAX_SEARCH_WIDTH, len(cached_rooms)),
            ):

                if (
                        room.id,
                        selected_item.timeslot_id,
                ) in occupied_rooms:
                    continue

                if (
                        section is not None
                        and ts_obj is not None
                        and not passes_hard_constraints(
                    section,
                    room,
                    ts_obj,
                    occupied_instructors=occupied_instructors,
                    occupied_rooms=occupied_rooms,
                )
                ):
                    continue

                if room.id != selected_item.room_id:
                    changed = True
                selected_item.room_id = room.id
                break

    else:

        room_obj = rooms_by_id.get(selected_item.room_id)

        for ts in random.sample(
                cached_timeslots,
                min(MAX_SEARCH_WIDTH, len(cached_timeslots)),
        ):

            instructor_free = (
                                  selected_item.instructor_id,
                                  ts.id,
                              ) not in occupied_instructors

            room_free = (
                            selected_item.room_id,
                            ts.id,
                        ) not in occupied_rooms

            if not (instructor_free and room_free):
                continue

            if (
                    section is not None
                    and room_obj is not None
                    and not passes_hard_constraints(
                section,
                room_obj,
                ts,
                occupied_instructors=occupied_instructors,
                occupied_rooms=occupied_rooms,
            )
            ):
                continue

            if ts.id != selected_item.timeslot_id:
                changed = True
            selected_item.timeslot_id = ts.id
            break

    return schedule, changed


# =========================
# TABU SEARCH
# =========================
def tabu_search(
        schedule,
        rooms,
        timeslots,
        sections,
        cache,
        option_cache=None,
):
    """
    CHANGED: the neighbor-generation step used to be a stub -- it picked
    a 50/50 "would change timeslot or room" branch but never actually
    changed anything, so every "neighbor" was identical to `current` and
    the whole tabu search did nothing but call repair_schedule on clones
    of the same schedule TABU_ITERATIONS times. Now it actually perturbs
    the chosen item's room or timeslot to a real candidate before
    repairing/scoring, so it can genuinely explore the neighborhood and
    escape local optima -- which is the whole point of running it when
    the GA plateaus.

    It also now prefers picking a currently-unscheduled item when one is
    available among the sampled indices, since the point of running this
    is specifically to close remaining gaps.
    """
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    rooms_by_id, timeslots_by_id = _get_id_maps(option_cache, rooms, timeslots)

    current = clone_schedule(schedule)
    best = clone_schedule(schedule)

    best_score = calculate_fitness(
        best,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=cache,
    )

    tabu_list = []

    for i in range(TABU_ITERATIONS):
        candidate_indices = random.sample(
            range(len(current)),
            min(MAX_SEARCH_WIDTH, len(current)),
        )

        unscheduled_candidates = [
            i for i in candidate_indices
            if current[i].room_id is None or current[i].timeslot_id is None
        ]
        idx = random.choice(unscheduled_candidates) if unscheduled_candidates else random.choice(candidate_indices)

        neighbor = clone_schedule(current)
        selected_item = neighbor[idx]
        section = sections[idx] if idx < len(sections) else None

        occupied_instructors = set()
        occupied_rooms = set()

        for j, item in enumerate(neighbor):
            if j == idx:
                continue
            if item.instructor_id is not None and item.timeslot_id is not None:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))
            if item.room_id is not None and item.timeslot_id is not None:
                occupied_rooms.add((item.room_id, item.timeslot_id))

        if idx in option_cache:
            viable_rooms, valid_timeslots = option_cache[idx]
        elif section is not None:
            viable_rooms = get_viable_rooms(section, rooms)
            valid_timeslots = get_valid_timeslots(section, timeslots)
        else:
            viable_rooms = get_viable_rooms_for_schedule_item(selected_item, rooms)
            valid_timeslots = timeslots

        old_room_id = selected_item.room_id
        old_ts_id = selected_item.timeslot_id

        change_timeslot = random.random() < 0.5

        if change_timeslot and valid_timeslots:
            move = ("ts", idx, old_ts_id)

            for ts in random.sample(valid_timeslots, min(MAX_SEARCH_WIDTH, len(valid_timeslots))):
                instr_conflict = (
                        selected_item.instructor_id is not None
                        and (selected_item.instructor_id, ts.id) in occupied_instructors
                )
                room_conflict = (
                        selected_item.room_id is not None
                        and (selected_item.room_id, ts.id) in occupied_rooms
                )
                if instr_conflict or room_conflict:
                    continue

                if section is not None and selected_item.room_id is not None:
                    room_obj = rooms_by_id.get(selected_item.room_id)
                    if room_obj is not None and not passes_hard_constraints(
                            section, room_obj, ts,
                            occupied_instructors=occupied_instructors,
                            occupied_rooms=occupied_rooms,
                    ):
                        continue

                selected_item.timeslot_id = ts.id
                break

        elif viable_rooms:
            move = ("room", idx, old_room_id)

            ts_obj = timeslots_by_id.get(selected_item.timeslot_id) if selected_item.timeslot_id else None

            for room in random.sample(viable_rooms, min(MAX_SEARCH_WIDTH, len(viable_rooms))):
                if selected_item.timeslot_id is not None and (room.id, selected_item.timeslot_id) in occupied_rooms:
                    continue

                if section is not None and ts_obj is not None and not passes_hard_constraints(
                        section, room, ts_obj,
                        occupied_instructors=occupied_instructors,
                        occupied_rooms=occupied_rooms,
                ):
                    continue

                selected_item.room_id = room.id
                break
        else:
            move = ("none", idx, None)

        # let repair_schedule clean up any fallout from the perturbation
        # (e.g. if it left the item partially unset, or bumped a hard
        # constraint elsewhere)
        repair_schedule(neighbor, sections, rooms, timeslots, option_cache=option_cache)
        score = calculate_fitness(neighbor, rooms, sections=sections, timeslots=timeslots, valid_timeslot_cache=cache)

        is_tabu = move in tabu_list
        is_aspirational = score > best_score

        if is_tabu and not is_aspirational:
            continue

        if score > best_score:
            best_score = score
            best = clone_schedule(neighbor)
            current = neighbor
        else:
            current = neighbor

        tabu_list.append(move)
        if len(tabu_list) > TABU_TENURE:
            tabu_list.pop(0)

    return best, best_score



# =========================
# LARGE-SCALE MIN-CONFLICTS REPAIR
# =========================
def min_conflicts_repair(
    schedule,
    sections,
    rooms,
    timeslots,
    option_cache=None,
    max_steps=20_000,
    candidate_limit=120,
    random_walk_rate=0.08,
):
    """
    Repair a large timetable using the classic min-conflicts strategy.

    Unlike the earlier cleanup functions, this method is allowed to move an
    item to a slot that is not immediately perfect when that move reduces the
    total number of collisions. This is important because many timetable
    conflicts are mutually blocking and cannot be solved by perfect-slot-only
    moves. Static hard constraints such as room type, capacity, campus, and
    valid timeslots are still enforced.
    """
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    room_counts = {}
    instructor_counts = {}

    def add_item(item):
        if item.room_id is not None and item.timeslot_id is not None:
            rk = (item.room_id, item.timeslot_id)
            room_counts[rk] = room_counts.get(rk, 0) + 1
            if item.instructor_id is not None:
                ik = (item.instructor_id, item.timeslot_id)
                instructor_counts[ik] = instructor_counts.get(ik, 0) + 1

    def remove_item(item):
        if item.room_id is not None and item.timeslot_id is not None:
            rk = (item.room_id, item.timeslot_id)
            room_counts[rk] -= 1
            if room_counts[rk] <= 0:
                del room_counts[rk]
            if item.instructor_id is not None:
                ik = (item.instructor_id, item.timeslot_id)
                instructor_counts[ik] -= 1
                if instructor_counts[ik] <= 0:
                    del instructor_counts[ik]

    def item_conflicts(item):
        if item.room_id is None or item.timeslot_id is None:
            return 3
        conflicts = max(0, room_counts.get((item.room_id, item.timeslot_id), 0) - 1)
        if item.instructor_id is not None:
            conflicts += max(
                0,
                instructor_counts.get((item.instructor_id, item.timeslot_id), 0) - 1,
            )
        return conflicts

    for item in schedule:
        add_item(item)

    best = clone_schedule(schedule)
    best_problem_count = len(_find_problem_indices(schedule))
    stagnant_steps = 0

    for step in range(max_steps):
        problem_indices = [
            idx for idx, item in enumerate(schedule)
            if item_conflicts(item) > 0
        ]

        if not problem_indices:
            return schedule

        # Prefer scarce sections, but retain randomness to escape cycles.
        sample = random.sample(
            problem_indices,
            min(50, len(problem_indices)),
        )
        idx = min(sample, key=lambda i: _constraint_score(i, option_cache))
        item = schedule[idx]
        section = sections[idx]

        viable_rooms, valid_timeslots = option_cache[idx]
        if not viable_rooms or not valid_timeslots:
            continue

        remove_item(item)
        old_room_id, old_timeslot_id = item.room_id, item.timeslot_id

        # Sample across the full Cartesian domain rather than taking only
        # the first rooms/timeslots. This keeps runtime bounded while giving
        # every valid area of the domain a chance to be explored.
        total_pairs = len(viable_rooms) * len(valid_timeslots)
        attempts = min(candidate_limit, total_pairs)
        candidates = []
        seen = set()

        if old_room_id is not None and old_timeslot_id is not None:
            seen.add((old_room_id, old_timeslot_id))

        while len(candidates) < attempts and len(seen) < total_pairs:
            room = random.choice(viable_rooms)
            ts = random.choice(valid_timeslots)
            key = (room.id, ts.id)
            if key in seen:
                continue
            seen.add(key)

            # Empty occupancy sets intentionally check only intrinsic hard
            # constraints. Occupancy collisions are measured by the
            # min-conflicts objective below instead of being rejected.
            if not passes_hard_constraints(
                section,
                room,
                ts,
                occupied_instructors=set(),
                occupied_rooms=set(),
            ):
                continue

            conflict_cost = room_counts.get((room.id, ts.id), 0)
            if item.instructor_id is not None:
                conflict_cost += instructor_counts.get(
                    (item.instructor_id, ts.id), 0
                )
            candidates.append((conflict_cost, room.id, ts.id))

        if not candidates:
            item.room_id, item.timeslot_id = old_room_id, old_timeslot_id
            add_item(item)
            continue

        candidates.sort(key=lambda x: x[0])
        minimum_cost = candidates[0][0]
        best_candidates = [c for c in candidates if c[0] == minimum_cost]

        if random.random() < random_walk_rate:
            _, new_room_id, new_timeslot_id = random.choice(candidates)
        else:
            _, new_room_id, new_timeslot_id = random.choice(best_candidates)

        item.room_id = new_room_id
        item.timeslot_id = new_timeslot_id
        add_item(item)

        if step % 100 == 0:
            current_problem_count = len(_find_problem_indices(schedule))
            if current_problem_count < best_problem_count:
                best_problem_count = current_problem_count
                best = clone_schedule(schedule)
                stagnant_steps = 0
            else:
                stagnant_steps += 100

            # Diversify after a long plateau by unsetting a few conflicted
            # items. The following iterations will place them again based on
            # the current global occupancy picture.
            if stagnant_steps >= 3_000 and problem_indices:
                for kick_idx in random.sample(
                    problem_indices, min(8, len(problem_indices))
                ):
                    kick_item = schedule[kick_idx]
                    remove_item(kick_item)
                    kick_item.room_id = None
                    kick_item.timeslot_id = None
                    add_item(kick_item)
                stagnant_steps = 0

    # Never return a result worse than the best state visited.
    if len(_find_problem_indices(schedule)) < best_problem_count:
        return schedule
    return best

# =========================
# MAIN GA + TABU LOOP
# =========================
def genetic_schedule(sections, timeslots, rooms, cache=None, option_cache=None):
    """
    Pure-Python CSP pipeline:
      1. MRV backtracking construction with forward checking.
      2. Existing exhaustive cleanup and ejection-chain repair.
      3. Soft-constraint hill-climbing polish.

    The public function name is unchanged, so the rest of the application
    can continue calling genetic_schedule().
    """
    if cache is None:
        cache = build_timeslot_guideline_cache(sections, timeslots)
    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    best = None
    best_score = None
    best_problem_count = None

    for _ in range(CSP_RESTARTS):
        candidate = csp_construct_schedule(
            sections,
            rooms,
            timeslots,
            option_cache=option_cache,
        )

        for _ in range(5):
            before = len(_find_problem_indices(candidate))
            force_place_remaining(
                candidate, sections, rooms, timeslots,
                option_cache=option_cache,
            )
            resolve_conflicts(
                candidate, sections, rooms, timeslots,
                option_cache=option_cache,
            )
            candidate, resolved = resolve_hard_cases(
                candidate, sections, rooms, timeslots,
                option_cache=option_cache,
            )
            after = len(_find_problem_indices(candidate))
            if after >= before and resolved == 0:
                break

        # Escape the local minimum left by perfect-slot-only cleanup.
        candidate = min_conflicts_repair(
            candidate,
            sections,
            rooms,
            timeslots,
            option_cache=option_cache,
        )

        # Run the deterministic cleanup again after min-conflicts has
        # rearranged the blocked areas of the timetable.
        for _ in range(3):
            force_place_remaining(
                candidate, sections, rooms, timeslots,
                option_cache=option_cache,
            )
            resolve_conflicts(
                candidate, sections, rooms, timeslots,
                option_cache=option_cache,
            )
            candidate, _ = resolve_hard_cases(
                candidate, sections, rooms, timeslots,
                option_cache=option_cache,
                max_candidates_per_item=20,
            )

        candidate = hill_climb_polish(
            candidate,
            sections,
            rooms,
            timeslots,
            cache,
            option_cache=option_cache,
        )

        problems = len(_find_problem_indices(candidate))
        score = calculate_fitness(
            candidate,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        if (
            best is None
            or problems < best_problem_count
            or (problems == best_problem_count and score > best_score)
        ):
            best = clone_schedule(candidate)
            best_problem_count = problems
            best_score = score

        if problems == 0:
            break

    diagnose_remaining_problems(best, sections, option_cache)
    return best