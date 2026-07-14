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
MUTATION_RATE = 0.15  # gate for "nudge an already-scheduled item"; unscheduled
                       # items now bypass this gate entirely (see mutation()).

TABU_ITERATIONS = 5
TABU_TENURE = 7

TS_APPLY_RATE = 0.2  # top 20% of population gets tabu search

ELITE_SIZE = 2

MAX_SEARCH_WIDTH = 20

# CHANGED: this used to hard-cap the final rescue pass at 50 leftover
# items -- meaning if more than 50 sections were unscheduled, the ONE
# exhaustive fallback pass silently never ran at all, and those
# sections stayed unscheduled forever regardless of how many
# generations you ran. Raised substantially, and the skip (if it ever
# happens) now prints a warning instead of failing silently.
RESCUE_MAX_ITEMS = 1000


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
# CREATE POPULATION
# =========================
def create_population(size, sections, rooms, timeslots, option_cache=None):
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

    last_scored = None

    for i in range(GENERATIONS):

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

                # CHANGED: also run the exhaustive rescue pass on each
                # tabu-polished individual, not just at the very end of
                # the whole run. This means leftover unscheduled items in
                # your best individuals get mopped up every time tabu
                # search runs (on stagnation, and on the final
                # generation), rather than only once at the very end --
                # so later generations get to build on fuller schedules.
                if count_unscheduled(improved_schedule) > 0:
                    force_place_remaining(
                        improved_schedule, sections, rooms, timeslots,
                        option_cache=option_cache,
                    )
                    improved_score = calculate_fitness(
                        improved_schedule, rooms, sections=sections,
                        timeslots=timeslots, valid_timeslot_cache=cache,
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
        num_runs,
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