import random

from backend.Optimization.constraints import (
    get_valid_timeslots,
    passes_hard_constraints,
    get_viable_rooms,
    get_viable_rooms_for_schedule_item,
)

from backend.models.models import ScheduleItem

from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
)

# =========================
# PARAMETERS
# =========================
POPULATION_SIZE = 15
GENERATIONS = 15

CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.15  # bumped from 0.08 for more variety

TABU_ITERATIONS = 5
TABU_TENURE = 7

TS_APPLY_RATE = 0.2  # top 20% of population gets tabu search

ELITE_SIZE = 2

# with thousands of sections and hundreds of rooms/timeslots, checking
# every room/timeslot combo for a stuck section is way too slow. so we
# just check a random sample of this many before giving up and taking
# whatever's least bad. bigger = better placements but slower.
MAX_SEARCH_WIDTH = 20

# how many leftover unscheduled items the final rescue pass will fully
# brute force (see force_place_remaining). usually only a few items
# need this so it's fine to search everything for them.
RESCUE_MAX_ITEMS = 50


# =========================
# CLONE HELPERS
# =========================
# copy.deepcopy() was too slow for this since it does a bunch of
# reflection even for something simple like ScheduleItem. just making
# a new one manually with the same fields does the same job faster.
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
    # counts sections that still don't have a room + timeslot
    return sum(
        1 for item in schedule
        if item.room_id is None or item.timeslot_id is None
    )


# =========================
# OPTION CACHE
# =========================
# get_viable_rooms / get_valid_timeslots get called a ton (every
# repair, every mutation, etc), so just compute them once up front and
# reuse instead of recalculating every time. big speedup.
#
# also stores id -> object lookup maps for rooms/timeslots under two
# reserved string keys. section indices are always ints so these keys
# never collide, and it means any function with option_cache can look
# up a room/timeslot by id without needing a separate param passed in.
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
    # fallback to building fresh maps if option_cache didn't come from
    # build_option_cache, just to be safe
    if option_cache and _ROOMS_BY_ID_KEY in option_cache:
        rooms_by_id = option_cache[_ROOMS_BY_ID_KEY]
        timeslots_by_id = option_cache[_TIMESLOTS_BY_ID_KEY]
    else:
        rooms_by_id = {r.id: r for r in rooms}
        timeslots_by_id = {t.id: t for t in timeslots}
    return rooms_by_id, timeslots_by_id


def _constraint_score(idx, option_cache):
    # lower = fewer valid options = harder to place, so handle these first
    if option_cache and idx in option_cache:
        viable_rooms, valid_timeslots = option_cache[idx]
        if viable_rooms and valid_timeslots:
            return len(viable_rooms) * len(valid_timeslots)
        return 0

    return 1_000_000  # no info, just push it to the back


# =========================
# CREATE POPULATION
# =========================
def create_population(size, sections, rooms, timeslots, option_cache=None):
    # start every individual as a completely empty schedule and let
    # repair_schedule() do the placing - it already knows how to prefer
    # a free slot and fall back to least-bad conflict, so no need to
    # duplicate that logic here.

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    population = []

    for _ in range(size):
        schedule = [
            ScheduleItem(
                course_id=section.course.id,
                course_name=section.course.name,
                course_type=section.course.type,
                course_dept=section.course.dept,
                capacity=section.capacity,
                instructor_id=section.instructor_id,
                room_id=None,
                timeslot_id=None,
                section=str(section.no),
            )
            for section in sections
        ]

        repair_schedule(schedule, sections, rooms, timeslots, option_cache=option_cache)

        population.append(schedule)

    return population


# =========================
# REPAIR
# =========================
def repair_schedule(schedule, sections, rooms, timeslots, option_cache=None):
    """
    cleans up the schedule:
      - double bookings (same room/instructor at the same timeslot)
      - hard constraint violations that mutation/tabu might've caused
      - anything still unassigned

    conflicts are allowed instead of hard-blocked. reasoning: if there's
    really no free slot left, setting room/timeslot back to None doesn't
    help anyone - it still needs to be fixed by hand, except now it's
    not even visible on the schedule. so: prefer a free slot if one
    exists, otherwise fall back to whatever has the fewest conflicts.
    only actually leaves something unassigned if it has zero viable
    rooms or zero valid timeslots to begin with (real infeasibility).

    only moves an item if the new slot is actually better (fewer
    conflicts), or it had nothing yet, or its current slot breaks a
    hard constraint. otherwise it just leaves it alone.
    """

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    rooms_by_id, timeslots_by_id = _get_id_maps(option_cache, rooms, timeslots)

    occupied_instructors = set()
    occupied_rooms = set()

    # most constrained sections get to pick their slot first
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

            # the double-booking checks above don't catch everything
            # (room too small, instructor unavailable, etc), so double
            # check the current slot is actually still valid
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
            # already fine, keep it and mark the slot as taken
            if item.instructor_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )
            occupied_rooms.add((item.room_id, item.timeslot_id))
            continue

        # unset, conflicting, or breaking a hard constraint - try to fix it
        if idx in option_cache:
            viable_rooms, valid_timeslots = option_cache[idx]
        elif section is not None:
            viable_rooms = get_viable_rooms(section, rooms)
            valid_timeslots = get_valid_timeslots(section, timeslots)
        else:
            viable_rooms = get_viable_rooms_for_schedule_item(item, rooms)
            valid_timeslots = timeslots

        if not viable_rooms or not valid_timeslots:
            # genuinely nowhere for this to go, nothing we can do here
            if not is_unset:
                if item.instructor_id is not None:
                    occupied_instructors.add(
                        (item.instructor_id, item.timeslot_id)
                    )
                occupied_rooms.add((item.room_id, item.timeslot_id))
            continue

        # random.sample gives us a bounded shuffled subset in one shot,
        # which is what keeps this fast even with tons of rooms/timeslots
        search_timeslots = random.sample(valid_timeslots, min(MAX_SEARCH_WIDTH, len(valid_timeslots)))

        best_room, best_ts, best_conflicts = None, None, None

        # phase 1: quick pass, skip anything occupied with a plain set
        # lookup instead of calling passes_hard_constraints on it. this
        # covers the common case where the item just needs any free slot.
        #
        # note: we re-sample rooms for each timeslot here instead of
        # reusing one fixed sample for all of them. reusing one sample
        # meant that if those particular rooms happened to be busy at
        # every sampled timeslot, we'd miss a free room sitting
        # elsewhere and force a conflict for no reason. same cost as
        # before, just fixes which rooms actually get checked.
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

        # phase 2: only runs if nothing was free. more expensive since
        # it has to check occupied combos too, but only kicks in for
        # sections that actually need it
        if best_room is None:

            search_rooms = random.sample(viable_rooms, min(MAX_SEARCH_WIDTH, len(viable_rooms)))

            for ts in search_timeslots:

                for room in search_rooms:

                    if section is not None and not passes_hard_constraints(
                            section,
                            room,
                            ts,
                            occupied_instructors=occupied_instructors,
                            occupied_rooms=occupied_rooms,
                    ):
                        continue

                    conflicts = 0
                    if (
                            item.instructor_id is not None
                            and (item.instructor_id, ts.id) in occupied_instructors
                    ):
                        conflicts += 1
                    if (room.id, ts.id) in occupied_rooms:
                        conflicts += 1

                    if best_conflicts is None or conflicts < best_conflicts:
                        best_room, best_ts, best_conflicts = room, ts, conflicts

        # swap in the new slot if: item had nothing yet, current slot
        # breaks a hard constraint (has to move regardless), or the new
        # slot beats the current conflict count
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
    """
    repair_schedule only samples MAX_SEARCH_WIDTH rooms/timeslots each
    try, so a few sections can end up unscheduled just from bad luck in
    the sampling, not because they're actually impossible to place. by
    the time everything else has settled this leftover list is usually
    tiny, so it's cheap to just check every option instead of sampling.
    only runs if the leftover list isn't too big, so it can't blow up
    runtime on a genuinely bad schedule.
    """

    unscheduled = [
        i for i, item in enumerate(schedule)
        if item.room_id is None or item.timeslot_id is None
    ]

    if not unscheduled or len(unscheduled) > RESCUE_MAX_ITEMS:
        return schedule

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    occupied_instructors = set()
    occupied_rooms = set()

    for item in schedule:
        if item.room_id is not None and item.timeslot_id is not None:
            if item.instructor_id is not None:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))
            occupied_rooms.add((item.room_id, item.timeslot_id))

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

        best_room, best_ts, best_conflicts = None, None, None

        for ts in valid_timeslots:

            instr_conflict = (
                    item.instructor_id is not None
                    and (item.instructor_id, ts.id) in occupied_instructors
            )

            for room in viable_rooms:

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

        if best_room is not None:
            item.room_id = best_room.id
            item.timeslot_id = best_ts.id
            if item.instructor_id is not None:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))
            occupied_rooms.add((item.room_id, item.timeslot_id))

    return schedule


# =========================
# FITNESS SCORING
# =========================
def score_population(population, rooms, sections, timeslots, cache):
    # just a small helper so we're not retyping this everywhere
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
    # fitness_lookup is id(schedule) -> fitness, built once per
    # generation so we're not recalculating fitness every single call

    competitors = random.sample(
        population,
        min(tournament_size, len(population)),
    )

    return max(competitors, key=lambda s: fitness_lookup[id(s)])


# =========================
# CROSSOVER
# =========================
def crossover(parent1, parent2, option_cache=None):
    # builds the child section by section, most constrained first.
    # picks whichever parent's slot 50/50 for variety, but if that pick
    # conflicts with something already placed in the child, falls back
    # to the other parent. only stays unscheduled if both parents'
    # slots are taken.

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
            # neither parent had a free slot, leave it open for repair
            # instead of forcing a conflict
            new_item = clone_item(preferred)
            new_item.room_id = None
            new_item.timeslot_id = None
            child[idx] = new_item

    return child


# =========================
# MUTATION
# =========================
def mutation(schedule, rooms, timeslots, option_cache=None, sections=None):
    # returns (schedule, changed) so the caller knows whether it needs
    # to rerun repair. mutation only fires ~15% of the time, so no
    # point rescanning everything if nothing actually changed

    if len(schedule) == 0:
        return schedule, False

    if random.random() > MUTATION_RATE:
        return schedule, False

    indexed = list(enumerate(schedule))

    unscheduled = [
        (i, item) for i, item in indexed
        if item.room_id is None or item.timeslot_id is None
    ]

    idx, selected_item = (
        random.choice(unscheduled) if unscheduled else random.choice(indexed)
    )

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

    # if it's unscheduled, use a bigger sample since this is the one we
    # actually want fixed. same idea as repair_schedule: prefer a free
    # slot, fall back to least-conflict instead of leaving it unset.
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

            if best_room is not None:
                selected_item.room_id = best_room.id
                selected_item.timeslot_id = best_ts.id
                return schedule, True

        return schedule, False

    # already scheduled, just nudge it a little
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

        idx = random.choice(candidate_indices)
        neighbor = clone_schedule(current)
        selected_item = neighbor[idx]
        section = sections[idx] if idx < len(sections) else None

        occupied_instructors = set()
        occupied_rooms = set()

        for j, item in enumerate(neighbor):
            if j == idx: continue
            if item.instructor_id is not None and item.timeslot_id is not None:
                occupied_instructors.add((item.instructor_id, item.timeslot_id))
            if item.room_id is not None and item.timeslot_id is not None:
                occupied_rooms.add((item.room_id, item.timeslot_id))

        # 50/50 change the timeslot or the room
        if random.random() < 0.5:
            # ... [Keep your existing timeslot selection logic here] ...
            move = ("ts", idx, selected_item.timeslot_id)
        else:
            # ... [Keep your existing room selection logic here] ...
            move = ("room", idx, selected_item.room_id)

        # --- aspiration criterion ---
        # check the fitness to see if we actually want this move
        repair_schedule(neighbor, sections, rooms, timeslots, option_cache=option_cache)
        score = calculate_fitness(neighbor, rooms, sections=sections, timeslots=timeslots, valid_timeslot_cache=cache)

        is_tabu = move in tabu_list
        is_aspirational = score > best_score

        # only skip the move if it's tabu AND not good enough to override that
        if is_tabu and not is_aspirational:
            continue

        # otherwise the move is allowed, either because it's not tabu
        # or because it's good enough to break the tabu rule
        if score > best_score:
            best_score = score
            best = clone_schedule(neighbor)
            current = neighbor
        else:
            current = neighbor  # still accept it, just don't set it as new best

        tabu_list.append(move)
        if len(tabu_list) > TABU_TENURE:
            tabu_list.pop(0)

    return best, best_score


# =========================
# MAIN GA + TABU LOOP
# =========================
def genetic_schedule(sections, timeslots, rooms, cache=None, option_cache=None):
    if cache is None:
        cache = build_timeslot_guideline_cache(
            sections,
            timeslots,
        )

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    population = create_population(
        POPULATION_SIZE,
        sections,
        rooms,
        timeslots,
        option_cache=option_cache,
    )

    best_generation_score = 0
    stagnation = 0
    NO_IMPROVEMENT_LIMIT = 5

    # keep track of this so we don't have to redo fitness for the
    # whole population again at the end
    last_scored = None

    for i in range(GENERATIONS):

        # score everyone once and reuse for picking elites + tournaments
        scored_population = score_population(population, rooms, sections, timeslots, cache)
        scored_population.sort(key=lambda x: x[1], reverse=True)

        half = len(scored_population) // 2
        selected_scored = scored_population[:half]
        selected = [s for s, _ in selected_scored]
        fitness_lookup = {id(s): f for s, f in scored_population}

        elites = [clone_schedule(s) for s, _ in selected_scored[:ELITE_SIZE]]

        new_population = elites

        while len(new_population) < POPULATION_SIZE:

            p1 = tournament_selection(selected, fitness_lookup)
            p2 = tournament_selection(selected, fitness_lookup)

            child = crossover(p1, p2, option_cache=option_cache)

            # crossover only assigns a slot when it's conflict-free, so
            # the only thing left to fix is a section it couldn't place
            # at all - skip repair if that didn't even happen
            if count_unscheduled(child) > 0:
                repair_schedule(child, sections, rooms, timeslots, option_cache=option_cache)

            child, mutated = mutation(
                child,
                rooms,
                timeslots,
                option_cache=option_cache,
                sections=sections,
            )

            if mutated:
                repair_schedule(child, sections, rooms, timeslots, option_cache=option_cache)

            new_population.append(child)

        scored_new = score_population(new_population, rooms, sections, timeslots, cache)
        scored_new.sort(key=lambda x: x[1], reverse=True)
        new_population = [s for s, _ in scored_new]

        current_best = scored_new[0][1]

        if current_best > best_generation_score:
            best_generation_score = current_best
            stagnation = 0
        else:
            stagnation += 1

        # this is the "hybrid" part - GA alone tends to plateau, so once
        # we stall for a few generations (or it's the last one anyway)
        # polish the top few with tabu search instead of just hoping
        # crossover eventually finds something better
        apply_tabu = stagnation >= NO_IMPROVEMENT_LIMIT or i == GENERATIONS - 1

        if apply_tabu:

            top_k = max(
                1,
                int(len(new_population) * TS_APPLY_RATE)
            )

            for j in range(top_k):
                improved_schedule, improved_score = tabu_search(
                    new_population[j],
                    rooms,
                    timeslots,
                    sections,
                    cache,
                    option_cache=option_cache,
                )

                new_population[j] = improved_schedule
                scored_new[j] = (improved_schedule, improved_score)

            scored_new.sort(key=lambda x: x[1], reverse=True)
            new_population = [s for s, _ in scored_new]

        population = new_population
        last_scored = scored_new

        if stagnation >= NO_IMPROVEMENT_LIMIT:
            break

    best, _ = last_scored[0]

    # one last repair in case something's still unscheduled or has a
    # hard-constraint violation, then a targeted rescue pass for
    # whatever few items repair's sampling still missed
    repair_schedule(best, sections, rooms, timeslots, option_cache=option_cache)
    force_place_remaining(best, sections, rooms, timeslots, option_cache=option_cache)

    return best


# =========================
# RUN IT A BUNCH OF TIMES
# =========================
def genetic_runs(
        sections,
        timeslots,
        rooms,
        num_runs=30,
):
    fitness_scores = []
    unscheduled_counts = []

    best_overall_schedule = None
    best_overall_fitness = -1

    print(
        f"\n=== Hybrid GA + Tabu Search: {num_runs} runs ==="
    )

    cache = build_timeslot_guideline_cache(
        sections,
        timeslots,
    )

    # sections/rooms/timeslots don't change between runs, so just build
    # this once instead of redoing it every single run
    option_cache = build_option_cache(sections, rooms, timeslots)

    for i in range(num_runs):

        best_schedule = genetic_schedule(
            sections,
            timeslots,
            rooms,
            cache,
            option_cache=option_cache,
        )

        score = calculate_fitness(
            best_schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        fitness_scores.append(score)
        unscheduled_counts.append(count_unscheduled(best_schedule))

        if score > best_overall_fitness:
            best_overall_fitness = score
            best_overall_schedule = best_schedule

    print("\n=== Results ===")
    print(f"Best fitness: {max(fitness_scores):.4f}")
    print(f"Worst fitness: {min(fitness_scores):.4f}")
    print(f"Average fitness: {sum(fitness_scores) / len(fitness_scores):.4f}")
    print(f"Average unscheduled sections: {sum(unscheduled_counts) / len(unscheduled_counts):.1f}")

    return best_overall_schedule