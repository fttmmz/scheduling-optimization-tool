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
MUTATION_RATE = 0.15  # was 0.08, bumped it up a bit for more variety

TABU_ITERATIONS = 5
TABU_TENURE = 7

TS_APPLY_RATE = 0.2  # top 20% of population gets tabu search

ELITE_SIZE = 2

# with thousands of sections and hundreds of rooms/timeslots, checking
# EVERY possible room/timeslot combo for a section that's stuck (the
# "phase 2" fallback below) gets way too slow. so instead we only look
# at a random sample of this many rooms/timeslots before giving up and
# taking the least-bad option we found. bigger = better placements but
# slower, smaller = faster but more conflicts left over.
MAX_SEARCH_WIDTH = 20

# how many still-unscheduled items the final "rescue" pass is allowed
# to brute-force search exhaustively (see force_place_remaining below).
# this normally only ever needs to touch a handful of items, so it's
# safe to search all of their options instead of a random sample.
RESCUE_MAX_ITEMS = 50


# =========================
# CLONE HELPERS
# =========================
# using copy.deepcopy() everywhere was really slow, since it has to do a
# bunch of reflection stuff even for a simple object like ScheduleItem.
# just building a new ScheduleItem manually with the same fields is way
# faster and does the exact same thing for us here.
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
    # quick check for how many sections still don't have a room+timeslot
    return sum(
        1 for item in schedule
        if item.room_id is None or item.timeslot_id is None
    )


# =========================
# OPTION CACHE
# =========================
# instead of recomputing get_viable_rooms / get_valid_timeslots every
# single time something touches a section (which happens A LOT - every
# repair call, every mutation, etc), just compute it once at the start
# of a run and reuse it everywhere. huge speedup.
#
# also stashes id->object lookup maps for rooms/timeslots under a
# couple of reserved string keys. every section index is an int, so
# these string keys can never collide with a real section index -
# this lets every function that already receives option_cache look up
# a room/timeslot object by id in O(1) without needing a brand new
# parameter threaded through the whole call chain.
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
    # falls back to building fresh maps if option_cache wasn't built via
    # build_option_cache for some reason - keeps this defensive without
    # forcing every caller to guarantee the keys exist
    if option_cache and _ROOMS_BY_ID_KEY in option_cache:
        rooms_by_id = option_cache[_ROOMS_BY_ID_KEY]
        timeslots_by_id = option_cache[_TIMESLOTS_BY_ID_KEY]
    else:
        rooms_by_id = {r.id: r for r in rooms}
        timeslots_by_id = {t.id: t for t in timeslots}
    return rooms_by_id, timeslots_by_id


def _constraint_score(idx, option_cache):
    # smaller number = fewer options = section is harder to place
    # so we want to handle these first before they run out of room
    if option_cache and idx in option_cache:
        viable_rooms, valid_timeslots = option_cache[idx]
        if viable_rooms and valid_timeslots:
            return len(viable_rooms) * len(valid_timeslots)
        return 0

    return 1_000_000  # no info on this one, just shove it to the back


# =========================
# CREATE POPULATION
# =========================
def create_population(size, sections, rooms, timeslots, option_cache=None):
    # build each individual as a totally empty schedule (no room/timeslot
    # assigned to anything yet) and let repair_schedule() do the actual
    # placing. repair_schedule already knows how to prefer a free slot
    # and fall back to the least-bad conflict, so there's no need for a
    # second version of that same logic just for the initial population.

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
    goes through the schedule and tries to clean up:
      - double bookings (same room/instructor at the same timeslot)
      - hard-constraint violations (wrong room type, instructor
        unavailable, etc) that got introduced by mutation/tabu moves
      - anything that's still unassigned

    allows conflicts instead of hard blocking them. reasoning: if
    there's genuinely no free slot left, dropping the section
    (room_id/timeslot_id = None) doesn't actually help - someone still
    has to go fix it by hand either way, except now it's not even
    showing up on the schedule to be fixed. so: always prefer a totally
    free slot if one exists, but if it doesn't, fall back to whatever
    slot has the fewest conflicts instead of leaving it empty. only
    actually unassigns something if it truly has zero viable rooms or
    zero valid timeslots to begin with - that's a real infeasibility,
    not something we can patch around.

    only swaps an item to a new slot if the new slot is actually BETTER
    (fewer conflicts) than what it already has, or if it didn't have
    anything yet, or if what it currently has actually breaks a hard
    constraint. otherwise leaves it where it is.
    """

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    rooms_by_id, timeslots_by_id = _get_id_maps(option_cache, rooms, timeslots)

    occupied_instructors = set()
    occupied_rooms = set()

    # most constrained sections get first dibs on their (limited) slots
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

            # double-booking checks above don't catch every hard
            # constraint (room too small, instructor not available at
            # that time, etc) - mutation/tabu can produce a slot that's
            # "free" in the double-booking sense but still invalid, so
            # double check that here before trusting the current slot
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
            # totally fine where it is, keep it and mark the slot taken
            if item.instructor_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )
            occupied_rooms.add((item.room_id, item.timeslot_id))
            continue

        # either unset, sitting in a conflict, or breaking a hard
        # constraint - see if we can do better
        if idx in option_cache:
            viable_rooms, valid_timeslots = option_cache[idx]
        elif section is not None:
            viable_rooms = get_viable_rooms(section, rooms)
            valid_timeslots = get_valid_timeslots(section, timeslots)
        else:
            viable_rooms = get_viable_rooms_for_schedule_item(item, rooms)
            valid_timeslots = timeslots

        if not viable_rooms or not valid_timeslots:
            # section genuinely has nowhere it's allowed to go, no amount
            # of conflict tolerance fixes that
            if not is_unset:
                if item.instructor_id is not None:
                    occupied_instructors.add(
                        (item.instructor_id, item.timeslot_id)
                    )
                occupied_rooms.add((item.room_id, item.timeslot_id))
            continue

        # random.sample gives us a bounded, shuffled subset in one go -
        # this is what keeps repair fast even with hundreds of rooms and
        # timeslots (see MAX_SEARCH_WIDTH comment up top)
        search_timeslots = random.sample(valid_timeslots, min(MAX_SEARCH_WIDTH, len(valid_timeslots)))

        best_room, best_ts, best_conflicts = None, None, None

        # phase 1: cheap pass - skip anything occupied with a plain set
        # lookup, don't even bother calling passes_hard_constraints on
        # it. this is the common case (item just needs ANY free slot).
        #
        # NOTE: a fresh room sample is drawn for each candidate timeslot
        # here, instead of reusing one fixed room sample for every
        # timeslot (as before). reusing one fixed sample meant that if
        # that particular draw of rooms happened to be busy at every
        # sampled timeslot, a free room sitting elsewhere in the list
        # was never even considered, forcing a needless conflict even
        # when a clean placement existed. same cost as before (still
        # bounded to MAX_SEARCH_WIDTH rooms checked per timeslot) - just
        # fixes which rooms get tried.
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

        # phase 2: only runs if nothing was free. has to check occupied
        # combos too so it's more expensive, but it only runs for
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

        # accept the new slot if: item had nothing yet, current slot
        # breaks a hard constraint (has to move no matter what), or the
        # new slot is a strict improvement over the current conflict count
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
    repair_schedule only samples MAX_SEARCH_WIDTH rooms/timeslots per
    attempt, so a handful of hard-to-place sections can still end up
    unscheduled just from bad luck in the sampling, not because they're
    truly infeasible. by the time everything else has settled, that
    leftover set is normally tiny (a handful of items, not thousands),
    so it's cheap to search their full option lists exhaustively
    instead of a random sample. only runs if there aren't too many
    leftovers, so it can't blow up runtime on a genuinely bad schedule.
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
    # small helper just so we're not retyping this call everywhere
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
    # fitness_lookup is just a dict of id(schedule) -> fitness that we
    # build once per generation, so we're not recalculating fitness for
    # every competitor every single time this gets called

    competitors = random.sample(
        population,
        min(tournament_size, len(population)),
    )

    return max(competitors, key=lambda s: fitness_lookup[id(s)])


# =========================
# CROSSOVER
# =========================
def crossover(parent1, parent2, option_cache=None):
    # builds the child section by section, most constrained ones first.
    # picks whichever parent's version 50/50 for variety, but if that one
    # already conflicts with something we already put in the child, we
    # just fall back to the other parent instead. only leaves it
    # unscheduled if BOTH parents' slots are taken already.

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
            # neither parent had a free slot for this one, leave it open
            # for repair to deal with instead of forcing a conflict
            new_item = clone_item(preferred)
            new_item.room_id = None
            new_item.timeslot_id = None
            child[idx] = new_item

    return child


# =========================
# MUTATION
# =========================
def mutation(schedule, rooms, timeslots, option_cache=None, sections=None):
    # returns (schedule, changed) so the caller knows if it actually
    # needs to re-run repair afterward. mutation only fires ~15% of the
    # time anyway, no point re-scanning the whole schedule when nothing
    # actually changed

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

    # if it's unscheduled, check a bigger sample of options instead of a
    # tiny one - this is the one we actually care about fixing. same as
    # repair_schedule, prefer a free slot but fall back to least-conflict
    # instead of leaving it unscheduled.
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

    # already scheduled item, just nudge it a bit
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
    # returns (best_schedule, best_score) so we don't have to recalculate
    # fitness again right after calling this

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

            if j == idx:
                continue

            if item.instructor_id is not None and item.timeslot_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )

            if item.room_id is not None and item.timeslot_id is not None:
                occupied_rooms.add(
                    (item.room_id, item.timeslot_id)
                )

        # 50/50 change the timeslot or the room
        if random.random() < 0.5:

            if idx in option_cache:
                valid_times = option_cache[idx][1]
            else:
                valid_times = get_valid_timeslots(sections[idx], timeslots)

            room_obj = rooms_by_id.get(selected_item.room_id)

            for ts in random.sample(
                valid_times,
                min(MAX_SEARCH_WIDTH, len(valid_times)),
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

                selected_item.timeslot_id = ts.id
                break

            move = (
                "ts",
                idx,
                selected_item.timeslot_id,
            )

        else:

            if idx in option_cache:
                valid_rooms = option_cache[idx][0]
            else:
                valid_rooms = get_viable_rooms_for_schedule_item(
                    selected_item,
                    rooms,
                )

            ts_obj = timeslots_by_id.get(selected_item.timeslot_id)

            for room in random.sample(
                valid_rooms,
                min(MAX_SEARCH_WIDTH, len(valid_rooms)),
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

                selected_item.room_id = room.id
                break

            move = (
                "room",
                idx,
                selected_item.room_id,
            )

        if move in tabu_list:
            continue

        # changing one item could still mess something up elsewhere,
        # so double check the whole thing before trusting the score
        repair_schedule(neighbor, sections, rooms, timeslots, option_cache=option_cache)

        score = calculate_fitness(
            neighbor,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        )

        if score > best_score:

            best_score = score
            best = clone_schedule(neighbor)
            current = neighbor

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

    # keeping track of this so we don't have to recompute fitness for
    # the whole population again at the very end
    last_scored = None

    for i in range(GENERATIONS):

        # score everyone once and reuse it for both picking the elites
        # and for every tournament selection call below
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

            # crossover only ever assigns a slot when it's conflict-free,
            # so the only thing left to fix is a section it couldn't
            # place at all - skip the repair pass when that didn't happen
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

        # this is the part that actually makes this "hybrid" - GA alone
        # tends to plateau, so once we stall for a few generations (or
        # we're on the last one anyway) polish the best few individuals
        # with tabu search instead of just hoping crossover finds better
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

    # one last repair pass in case anything's still sitting unscheduled
    # or ended up with a hard-constraint violation, then a targeted
    # rescue pass for whatever handful of items repair's sampling still
    # couldn't place (see force_place_remaining docstring)
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

    # sections/rooms/timeslots stay the same across every run, so just
    # build this once instead of redoing it num_runs times
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
    print(f"Average fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")
    print(f"Average unscheduled sections: {sum(unscheduled_counts)/len(unscheduled_counts):.1f}")

    return best_overall_schedule