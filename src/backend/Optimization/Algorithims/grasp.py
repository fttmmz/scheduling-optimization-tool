import random
import copy
import statistics

from backend.models.models import ScheduleItem
from backend.Optimization.constraints import (
    get_valid_timeslots,
    get_viable_rooms,
    get_viable_rooms_for_schedule_item,
    passes_hard_constraints,
)
from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
)

# ============================================================
# GRASP Parameters
# ============================================================
RCL_SIZE = 3
MAX_ITERATIONS = 20
LOCAL_SEARCH_ITERATIONS = 30
NO_IMPROVE_LIMIT = 5  # NEW: stop early if no improvement

# Local search tries this many (room, timeslot) candidates per item instead
# of the full room x timeslot cross product. Each candidate costs a full
# calculate_fitness() call over the whole schedule (O(sections)), so
# exhaustively trying every combination for every item, every pass, every
# GRASP restart, every run made runtime blow up combinatorially on
# real-sized datasets. Sampling bounds that cost to a small constant.
NEIGHBORHOOD_ROOM_SAMPLE = 5
NEIGHBORHOOD_TIMESLOT_SAMPLE = 5


# ============================================================
# Candidate Evaluation
# ============================================================
def score_candidate(section, room, timeslot):
    score = 0
    remaining_capacity = room.capacity - section.capacity
    if remaining_capacity >= 0:
        score += 100 - remaining_capacity
    score += 10
    return score


# ============================================================
# Choose Assignment
# ============================================================
def choose_grasp_assignment(
    section,
    rooms,
    timeslots,
    occupied_instructors,
    occupied_rooms,
):
    candidates = []

    instructor_id = section.instructor_id  # cache lookup

    for timeslot in timeslots:

        # Check instructor conflict once per timeslot
        if (
            instructor_id is not None
            and (instructor_id, timeslot.id) in occupied_instructors
        ):
            continue

        for room in rooms:

            if (room.id, timeslot.id) in occupied_rooms:
                continue

            if not passes_hard_constraints(
                section,
                room,
                timeslot,
                occupied_instructors=occupied_instructors,
                occupied_rooms=occupied_rooms,
            ):
                continue

            score = score_candidate(section, room, timeslot)
            candidates.append((score, room, timeslot))  # tuple faster than dict

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)

    rcl = candidates[: min(RCL_SIZE, len(candidates))]
    selected = random.choice(rcl)

    return selected[1], selected[2]


# ============================================================
# Construction Phase
# ============================================================
def construct_grasp_solution(sections, rooms, timeslots, section_candidates):
    schedule = []
    occupied_rooms = set()
    occupied_instructors = set()

    for section, viable_rooms, valid_ts in section_candidates:

        room, timeslot = choose_grasp_assignment(
            section,
            viable_rooms,
            valid_ts,
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
            timeslot_id=timeslot.id if timeslot else None,
            section=str(section.no),
        )

        schedule.append(item)

        if room is not None and timeslot is not None:
            occupied_rooms.add((room.id, timeslot.id))
            if section.instructor_id is not None:
                occupied_instructors.add((section.instructor_id, timeslot.id))

    return schedule


# ============================================================
# Local Search
# KEY FIX: avoid calling calculate_fitness() inside inner loop
# Instead: only evaluate after a swap, restore if worse
# ============================================================
def local_search(
    schedule,
    rooms,
    timeslots,
    sections,
    valid_timeslot_cache,
):
    best_schedule = schedule  # NO deepcopy here, we restore manually
    best_fitness = calculate_fitness(
        best_schedule,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )

    # Pre-shuffle timeslots once per local search call
    timeslot_list = timeslots[:]

    for _ in range(LOCAL_SEARCH_ITERATIONS):

        improved = False
        random.shuffle(timeslot_list)  # shuffle once per iteration, not per item

        for item in best_schedule:

            if item.room_id is None or item.timeslot_id is None:
                continue

            original_room = item.room_id
            original_timeslot = item.timeslot_id

            # Pre-fetch viable rooms once per item, then sample a bounded
            # neighborhood instead of scanning every room x timeslot pair
            viable_rooms = get_viable_rooms_for_schedule_item(item, rooms)
            room_sample = random.sample(
                viable_rooms, min(NEIGHBORHOOD_ROOM_SAMPLE, len(viable_rooms))
            )
            timeslot_sample = random.sample(
                timeslot_list, min(NEIGHBORHOOD_TIMESLOT_SAMPLE, len(timeslot_list))
            )

            found = False

            for room in room_sample:
                for timeslot in timeslot_sample:

                    # Skip if same assignment
                    if room.id == original_room and timeslot.id == original_timeslot:
                        continue

                    # Apply swap
                    item.room_id = room.id
                    item.timeslot_id = timeslot.id

                    new_fitness = calculate_fitness(
                        best_schedule,
                        rooms,
                        sections=sections,
                        timeslots=timeslots,
                        valid_timeslot_cache=valid_timeslot_cache,
                    )

                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        original_room = room.id
                        original_timeslot = timeslot.id
                        improved = True
                        found = True
                        break  # accept first improvement
                    else:
                        # Restore immediately
                        item.room_id = original_room
                        item.timeslot_id = original_timeslot

                if found:
                    break

            # Ensure restored if no improvement found
            if not found:
                item.room_id = original_room
                item.timeslot_id = original_timeslot

        if not improved:
            break  # early exit — no point continuing

    return best_schedule, best_fitness  # return fitness so we don't recompute it


# ============================================================
# GRASP Algorithm
# ============================================================
def grasp_schedule(
    sections,
    timeslots,
    rooms,
    valid_timeslot_cache=None,
    section_candidates=None,  # NEW: pass in precomputed candidates
):
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    # Precompute once and reuse across all iterations
    if section_candidates is None:
        section_candidates = [
            (
                section,
                get_viable_rooms(section, rooms),
                get_valid_timeslots(section, timeslots),
            )
            for section in sections
        ]

    best_schedule = None
    best_fitness = -1
    no_improve_count = 0  # NEW: early stopping counter

    for _ in range(MAX_ITERATIONS):

        schedule = construct_grasp_solution(
            sections,
            rooms,
            timeslots,
            section_candidates,  # reuse precomputed candidates
        )

        schedule, fitness = local_search(  # fitness returned, not recomputed
            schedule,
            rooms,
            timeslots,
            sections,
            valid_timeslot_cache,
        )

        if fitness > best_fitness:
            best_fitness = fitness
            best_schedule = copy.deepcopy(schedule)  # deepcopy only on improvement
            no_improve_count = 0
        else:
            no_improve_count += 1

        # NEW: early stopping
        if no_improve_count >= NO_IMPROVE_LIMIT:
            break

    return best_schedule, best_fitness


# ============================================================
# Multiple GRASP Runs
# ============================================================
def grasp_runs(sections, timeslots, rooms, num_runs=30):

    fitness_scores = []
    best_overall_schedule = None
    best_overall_fitness = -1

    print(f"\n=== GRASP Algorithm: {num_runs} runs ===")
    print(
        f"Total sections: {len(sections)}, "
        f"rooms: {len(rooms)}, "
        f"timeslots: {len(timeslots)}"
    )

    # Build cache ONCE for all runs
    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    # Precompute section candidates ONCE for all runs
    section_candidates = [
        (
            section,
            get_viable_rooms(section, rooms),
            get_valid_timeslots(section, timeslots),
        )
        for section in sections
    ]

    for run in range(num_runs):

        best_schedule, score = grasp_schedule(  # score already computed
            sections,
            timeslots,
            rooms,
            valid_timeslot_cache=valid_timeslot_cache,
            section_candidates=section_candidates,
        )

        fitness_scores.append(score)

        if score > best_overall_fitness:
            best_overall_fitness = score
            best_overall_schedule = copy.deepcopy(best_schedule)

        scheduled = sum(
            1 for item in best_schedule
            if item.room_id is not None and item.timeslot_id is not None
        )

        print(
            f"Run {run + 1:2d}: "
            f"Fitness = {score:.4f} | "
            f"Scheduled = {scheduled}"
        )

    # Statistics
    best_score = max(fitness_scores)
    worst_score = min(fitness_scores)
    avg_score = sum(fitness_scores) / len(fitness_scores)
    std_dev = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0

    print("\n=== Results ===")
    print(f"Best fitness:       {best_score:.4f}")
    print(f"Worst fitness:      {worst_score:.4f}")
    print(f"Average fitness:    {avg_score:.4f}")
    print(f"Standard deviation: {std_dev:.4f}")

    return best_overall_schedule
