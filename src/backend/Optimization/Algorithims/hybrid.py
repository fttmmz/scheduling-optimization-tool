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

# how many times rescue_unscheduled() gets to run in a row before giving up
RESCUE_ROUNDS = 3


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


# =========================
# OPTION CACHE
# =========================
# instead of recomputing get_viable_rooms / get_valid_timeslots every
# single time something touches a section (which happens A LOT - every
# repair call, every mutation, etc), just compute it once at the start
# of a run and reuse it everywhere. huge speedup.
def build_option_cache(sections, rooms, timeslots):

    cache = {}

    for idx, section in enumerate(sections):
        viable_rooms = get_viable_rooms(section, rooms)
        valid_timeslots = get_valid_timeslots(section, timeslots)
        cache[idx] = (viable_rooms, valid_timeslots)

    return cache


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
# RANDOM ASSIGNMENT
# =========================
def choose_random_assignment(
    section,
    rooms,
    timeslots,
    occupied_instructors,
    occupied_rooms,
):
    # rooms/timeslots should already be shuffled by the caller so we get
    # different results each time this runs. we just take the first
    # combo that actually works and passes the hard constraints.
    # if nothing works we just return None, None and let repair deal
    # with it later - no random fallback, that was causing double
    # bookings before.

    if not rooms or not timeslots:
        return None, None

    for ts in timeslots:

        if (
            section.instructor_id is not None
            and (section.instructor_id, ts.id) in occupied_instructors
        ):
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

    return None, None


# =========================
# CREATE POPULATION
# =========================
def create_population(size, sections, rooms, timeslots, option_cache=None):

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    population = []

    section_candidates = [
        (idx, section, option_cache[idx][0], option_cache[idx][1])
        for idx, section in enumerate(sections)
    ]

    for i in range(size):

        schedule = [None] * len(sections)

        occupied_instructors = set()
        occupied_rooms = set()

        # place the most constrained sections first so they don't get
        # stuck with nothing left. random.random() tiebreak so each
        # individual in the population still comes out a bit different.
        order = sorted(
            section_candidates,
            key=lambda c: (
                len(c[2]) * len(c[3]) if c[2] and c[3] else 0,
                random.random(),
            ),
        )

        for idx, section, viable_rooms, valid_timeslots in order:

            shuffled_rooms = viable_rooms[:]
            shuffled_timeslots = valid_timeslots[:]
            random.shuffle(shuffled_rooms)
            random.shuffle(shuffled_timeslots)

            room, ts = choose_random_assignment(
                section,
                shuffled_rooms,
                shuffled_timeslots,
                occupied_instructors,
                occupied_rooms,
            )

            item = ScheduleItem(
                course_id=section.course.id,
                course_name=section.course.name,
                course_type=section.course.type,
                course_dept=section.course.dept,
                capacity=section.capacity,
                instructor_id=section.instructor_id,
                room_id=room.id if room else None,
                timeslot_id=ts.id if ts else None,
                section=str(section.no),
            )

            schedule[idx] = item

            if room and ts:
                occupied_rooms.add((room.id, ts.id))

            if section.instructor_id and ts:
                occupied_instructors.add(
                    (section.instructor_id, ts.id)
                )

        # this only runs once per individual so it's fine to do the
        # expensive deep repair here (regular repair + rescue rounds)
        deep_repair_schedule(schedule, sections, rooms, timeslots, option_cache=option_cache)

        population.append(schedule)

    return population


# =========================
# REPAIR
# =========================
def repair_schedule(schedule, sections, rooms, timeslots, option_cache=None):
    """
    goes through the schedule and fixes:
      - double bookings (same room/instructor at the same timeslot)
      - anything that's still unassigned

    note: this already tries EVERY viable room/timeslot for each item,
    not just a random handful. so just calling this again and again on
    the same schedule won't magically schedule more stuff - if a section
    couldn't find a slot, every free slot was already exhausted, and
    that won't change unless something else moves out of the way first.
    that's what rescue_unscheduled() below is for.
    """

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

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

        has_conflict = False

        if item.instructor_id is not None and item.timeslot_id is not None:
            if (item.instructor_id, item.timeslot_id) in occupied_instructors:
                has_conflict = True

        if item.room_id is not None and item.timeslot_id is not None:
            if (item.room_id, item.timeslot_id) in occupied_rooms:
                has_conflict = True

        needs_assignment = (
            item.room_id is None
            or item.timeslot_id is None
            or has_conflict
        )

        if not needs_assignment:
            if item.instructor_id is not None and item.timeslot_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )
            if item.room_id is not None and item.timeslot_id is not None:
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
            item.room_id = None
            item.timeslot_id = None
            continue

        shuffled_rooms = viable_rooms[:]
        shuffled_timeslots = valid_timeslots[:]
        random.shuffle(shuffled_rooms)
        random.shuffle(shuffled_timeslots)

        assigned = False

        for ts in shuffled_timeslots:

            if (
                item.instructor_id is not None
                and (item.instructor_id, ts.id) in occupied_instructors
            ):
                continue

            for room in shuffled_rooms:

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

                item.room_id = room.id
                item.timeslot_id = ts.id
                assigned = True
                break

            if assigned:
                break

        if assigned:
            if item.instructor_id is not None:
                occupied_instructors.add(
                    (item.instructor_id, item.timeslot_id)
                )
            occupied_rooms.add((item.room_id, item.timeslot_id))
        else:
            # couldn't find anywhere to put it, leave it unscheduled
            # instead of forcing a bad slot
            item.room_id = None
            item.timeslot_id = None

    return schedule


# =========================
# RESCUE PASS (this is the actual fix for the unscheduled count)
# =========================
def rescue_unscheduled(schedule, sections, rooms, timeslots, option_cache=None):
    """
    ok so the problem with repair_schedule is it only fills empty slots -
    it never moves something that's already scheduled, even if moving it
    would free up a slot that some other stuck section desperately needs.

    so this function looks at each unscheduled section, checks its
    possible room/timeslot combos, and if a combo is only blocked by ONE
    other item (that item could just move somewhere else), it bumps that
    item out of the way and puts the stuck section in.

    only goes one item deep (doesn't chase a whole chain of moves) to
    keep it from getting out of hand. returns how many sections it
    managed to rescue so the caller knows whether to keep trying.
    """

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    room_ts_owner = {}
    instr_ts_owner = {}
    occupied_rooms = set()
    occupied_instructors = set()

    for idx, item in enumerate(schedule):
        if item.room_id is not None and item.timeslot_id is not None:
            room_ts_owner[(item.room_id, item.timeslot_id)] = idx
            occupied_rooms.add((item.room_id, item.timeslot_id))
        if item.instructor_id is not None and item.timeslot_id is not None:
            instr_ts_owner[(item.instructor_id, item.timeslot_id)] = idx
            occupied_instructors.add((item.instructor_id, item.timeslot_id))

    unscheduled_indices = [
        idx for idx, item in enumerate(schedule)
        if item.room_id is None or item.timeslot_id is None
    ]

    rescued = 0

    for idx in unscheduled_indices:

        item = schedule[idx]
        section = sections[idx] if idx < len(sections) else None

        if idx not in option_cache:
            continue

        viable_rooms, valid_timeslots = option_cache[idx]

        if not viable_rooms or not valid_timeslots:
            continue

        shuffled_rooms = viable_rooms[:]
        shuffled_timeslots = valid_timeslots[:]
        random.shuffle(shuffled_rooms)
        random.shuffle(shuffled_timeslots)

        placed = False

        for ts in shuffled_timeslots:

            instr_blocker = (
                instr_ts_owner.get((item.instructor_id, ts.id))
                if item.instructor_id is not None
                else None
            )

            for room in shuffled_rooms:

                room_blocker = room_ts_owner.get((room.id, ts.id))
                blockers = {b for b in (instr_blocker, room_blocker) if b is not None}

                if not blockers:
                    # slot is actually free? shouldn't really happen since
                    # repair_schedule already grabs free ones, but just in case
                    if section is not None and not passes_hard_constraints(
                        section, room, ts,
                        occupied_instructors=occupied_instructors,
                        occupied_rooms=occupied_rooms,
                    ):
                        continue

                    item.room_id = room.id
                    item.timeslot_id = ts.id
                    room_ts_owner[(room.id, ts.id)] = idx
                    occupied_rooms.add((room.id, ts.id))
                    if item.instructor_id is not None:
                        instr_ts_owner[(item.instructor_id, ts.id)] = idx
                        occupied_instructors.add((item.instructor_id, ts.id))
                    placed = True
                    rescued += 1
                    break

                if len(blockers) > 1:
                    # both the room AND the instructor slot are taken by
                    # different sections - would have to move both, skipping
                    # this one, too messy for now
                    continue

                blocker_idx = next(iter(blockers))

                if blocker_idx not in option_cache:
                    continue

                blocker_item = schedule[blocker_idx]
                blocker_rooms, blocker_timeslots = option_cache[blocker_idx]

                if not blocker_rooms or not blocker_timeslots:
                    continue

                blocker_section = (
                    sections[blocker_idx] if blocker_idx < len(sections) else None
                )

                b_rooms = blocker_rooms[:]
                b_timeslots = blocker_timeslots[:]
                random.shuffle(b_rooms)
                random.shuffle(b_timeslots)

                relocated = False

                # try to find the blocking item a new home somewhere else
                for new_ts in b_timeslots:

                    if new_ts.id == ts.id:
                        continue  # that would just put us back where we started

                    if (
                        blocker_item.instructor_id is not None
                        and (blocker_item.instructor_id, new_ts.id) in occupied_instructors
                    ):
                        continue

                    for new_room in b_rooms:

                        if (new_room.id, new_ts.id) in occupied_rooms:
                            continue

                        if blocker_section is not None and not passes_hard_constraints(
                            blocker_section, new_room, new_ts,
                            occupied_instructors=occupied_instructors,
                            occupied_rooms=occupied_rooms,
                        ):
                            continue

                        # move the blocker out
                        occupied_rooms.discard(
                            (blocker_item.room_id, blocker_item.timeslot_id)
                        )
                        room_ts_owner.pop(
                            (blocker_item.room_id, blocker_item.timeslot_id), None
                        )
                        if blocker_item.instructor_id is not None:
                            occupied_instructors.discard(
                                (blocker_item.instructor_id, blocker_item.timeslot_id)
                            )
                            instr_ts_owner.pop(
                                (blocker_item.instructor_id, blocker_item.timeslot_id),
                                None,
                            )

                        blocker_item.room_id = new_room.id
                        blocker_item.timeslot_id = new_ts.id

                        room_ts_owner[(new_room.id, new_ts.id)] = blocker_idx
                        occupied_rooms.add((new_room.id, new_ts.id))
                        if blocker_item.instructor_id is not None:
                            instr_ts_owner[(blocker_item.instructor_id, new_ts.id)] = blocker_idx
                            occupied_instructors.add(
                                (blocker_item.instructor_id, new_ts.id)
                            )

                        relocated = True
                        break

                    if relocated:
                        break

                if not relocated:
                    # couldn't move the blocker anywhere, oh well, try the
                    # next combo for our stuck section
                    continue

                # blocker moved out, slot is ours now
                if section is not None and not passes_hard_constraints(
                    section, room, ts,
                    occupied_instructors=occupied_instructors,
                    occupied_rooms=occupied_rooms,
                ):
                    continue

                item.room_id = room.id
                item.timeslot_id = ts.id
                room_ts_owner[(room.id, ts.id)] = idx
                occupied_rooms.add((room.id, ts.id))
                if item.instructor_id is not None:
                    instr_ts_owner[(item.instructor_id, ts.id)] = idx
                    occupied_instructors.add((item.instructor_id, ts.id))

                placed = True
                rescued += 1
                break

            if placed:
                break

    return rescued


def deep_repair_schedule(
    schedule, sections, rooms, timeslots, option_cache=None, rescue_rounds=RESCUE_ROUNDS
):
    # normal repair + a few rounds of the rescue pass. more thorough but
    # also more expensive, so only use this at the spots where it's
    # worth spending the extra time (building the population, checking
    # the elites, and the final result) - not on every single crossover
    # or mutation in the main loop, that would slow things down a lot
    # for not much benefit.

    if option_cache is None:
        option_cache = build_option_cache(sections, rooms, timeslots)

    repair_schedule(schedule, sections, rooms, timeslots, option_cache=option_cache)

    for _ in range(rescue_rounds):
        rescued = rescue_unscheduled(schedule, sections, rooms, timeslots, option_cache=option_cache)
        if rescued == 0:
            break
        # moving stuff around in the rescue pass could in theory create a
        # weird edge case conflict, so just run repair again to be safe
        repair_schedule(schedule, sections, rooms, timeslots, option_cache=option_cache)

    return schedule


# =========================
# SELECTION
# =========================
def selection(population, rooms, sections, timeslots, cache):
    # keeping this around in case something else in the codebase still
    # calls it, but genetic_schedule() below does its own scoring now
    # so it doesn't need to call this anymore (saves recalculating
    # fitness twice for the same population)

    population.sort(
        key=lambda s: calculate_fitness(
            s,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=cache,
        ),
        reverse=True,
    )

    return population[: len(population) // 2]


def tournament_selection(population, fitness_lookup, tournament_size=3):
    # fitness_lookup is just a dict of id(schedule) -> fitness that we
    # build once per generation, so we're not recalculating fitness for
    # every competitor every single time this gets called (that was
    # adding up to a LOT of extra calculate_fitness calls before)

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
def mutation(schedule, rooms, timeslots, option_cache=None):
    # returns (schedule, changed) now so the caller knows if it actually
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

    # if it's unscheduled, go all out and check every single option
    # instead of a random sample - this is the one we actually care
    # about fixing, so worth the extra time
    if selected_item.room_id is None or selected_item.timeslot_id is None:

        if cached_rooms and cached_timeslots:

            shuffled_rooms = cached_rooms[:]
            shuffled_timeslots = cached_timeslots[:]
            random.shuffle(shuffled_rooms)
            random.shuffle(shuffled_timeslots)

            for ts in shuffled_timeslots:

                if (
                    selected_item.instructor_id is not None
                    and (selected_item.instructor_id, ts.id) in occupied_instructors
                ):
                    continue

                placed = False

                for room in shuffled_rooms:

                    if (room.id, ts.id) in occupied_rooms:
                        continue

                    selected_item.room_id = room.id
                    selected_item.timeslot_id = ts.id
                    placed = True
                    break

                if placed:
                    return schedule, True

        return schedule, False

    # already scheduled item, just nudge it a bit (this one's fine with
    # a random sample since it's not the priority case)
    changed = False

    if random.random() < 0.5:

        if cached_rooms and selected_item.timeslot_id:

            for room in random.sample(
                cached_rooms,
                min(20, len(cached_rooms)),
            ):

                if (
                    room.id,
                    selected_item.timeslot_id,
                ) not in occupied_rooms:

                    if room.id != selected_item.room_id:
                        changed = True
                    selected_item.room_id = room.id
                    break

    else:

        for ts in random.sample(
            cached_timeslots,
            min(30, len(cached_timeslots)),
        ):

            instructor_free = (
                selected_item.instructor_id,
                ts.id,
            ) not in occupied_instructors

            room_free = (
                selected_item.room_id,
                ts.id,
            ) not in occupied_rooms

            if instructor_free and room_free:
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
            min(20, len(current)),
        )

        idx = random.choice(candidate_indices)

        neighbor = clone_schedule(current)
        selected_item = neighbor[idx]

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

            for ts in random.sample(
                valid_times,
                min(30, len(valid_times)),
            ):

                instructor_free = (
                    selected_item.instructor_id,
                    ts.id,
                ) not in occupied_instructors

                room_free = (
                    selected_item.room_id,
                    ts.id,
                ) not in occupied_rooms

                if instructor_free and room_free:

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

            for room in random.sample(
                valid_rooms,
                min(20, len(valid_rooms)),
            ):

                if (
                    room.id,
                    selected_item.timeslot_id,
                ) not in occupied_rooms:

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
        # and for every tournament selection call below (before, each
        # tournament call was recalculating fitness for 3 schedules
        # every time it ran, which added up to a ton of wasted calls)
        scored_population = [
            (
                s,
                calculate_fitness(
                    s,
                    rooms,
                    sections=sections,
                    timeslots=timeslots,
                    valid_timeslot_cache=cache,
                ),
            )
            for s in population
        ]
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

            # crossover already avoids most conflicts but run repair once
            # more just to catch anything weird that slipped through
            repair_schedule(child, sections, rooms, timeslots, option_cache=option_cache)

            child, mutated = mutation(
                child,
                rooms,
                timeslots,
                option_cache=option_cache,
            )

            # only bother re-repairing if mutation actually did something
            if mutated:
                repair_schedule(child, sections, rooms, timeslots, option_cache=option_cache)

            new_population.append(child)

        scored_new = [
            (
                s,
                calculate_fitness(
                    s,
                    rooms,
                    sections=sections,
                    timeslots=timeslots,
                    valid_timeslot_cache=cache,
                ),
            )
            for s in new_population
        ]
        scored_new.sort(key=lambda x: x[1], reverse=True)
        new_population = [s for s, _ in scored_new]

        current_best = scored_new[0][1]

        if current_best > best_generation_score:
            best_generation_score = current_best
            stagnation = 0
        else:
            stagnation += 1

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

        # this is the part that actually helps with the unscheduled
        # count over time - regular repair can't recover a section that's
        # only stuck because of something else already sitting in its
        # slot, so give the top few individuals a proper deep repair
        # every generation. only doing this for ELITE_SIZE individuals
        # so it doesn't slow down the whole population every round
        for j in range(min(ELITE_SIZE, len(new_population))):
            deep_repair_schedule(
                new_population[j], sections, rooms, timeslots, option_cache=option_cache
            )
            scored_new[j] = (
                new_population[j],
                calculate_fitness(
                    new_population[j],
                    rooms,
                    sections=sections,
                    timeslots=timeslots,
                    valid_timeslot_cache=cache,
                ),
            )

        scored_new.sort(key=lambda x: x[1], reverse=True)
        new_population = [s for s, _ in scored_new]

        population = new_population
        last_scored = scored_new

        if stagnation >= NO_IMPROVEMENT_LIMIT:
            break

    best, _ = last_scored[0]

    # last chance to fix anything - full deep repair, not just the
    # regular pass, so we actually give it a shot at rescuing whatever's
    # left before handing it back
    deep_repair_schedule(best, sections, rooms, timeslots, option_cache=option_cache)

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
    # build this once instead of redoing it 30 times
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

        if score > best_overall_fitness:
            best_overall_fitness = score
            best_overall_schedule = best_schedule

    print("\n=== Results ===")
    print(f"Best fitness: {max(fitness_scores):.4f}")
    print(f"Worst fitness: {min(fitness_scores):.4f}")
    print(f"Average fitness: {sum(fitness_scores)/len(fitness_scores):.4f}")

    return best_overall_schedule