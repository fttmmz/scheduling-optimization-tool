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

# Number of best candidates kept in the Restricted Candidate List (RCL)
RCL_SIZE = 3

# Number of GRASP iterations
MAX_ITERATIONS = 20

# Maximum number of local search improvements
LOCAL_SEARCH_ITERATIONS = 30


# ============================================================
# Candidate Evaluation
# This function scores one room-timeslot assignment.
# Higher score = better assignment.
# ============================================================

def score_candidate(section, room, timeslot):
    score = 0

    # Prefer rooms that closely match the section capacity
    remaining_capacity = room.capacity - section.capacity

    if remaining_capacity >= 0:
        score += 100 - remaining_capacity

    # Small bonus for valid assignments
    score += 10

    return score


# ============================================================
# Choose Assignment
# This function creates a Restricted Candidate List (RCL)
# and randomly chooses one assignment from it.
# ============================================================

def choose_grasp_assignment(
    section,
    rooms,
    timeslots,
    occupied_instructors,
    occupied_rooms,
):
    candidates = []

    # Find every feasible room-timeslot pair
    for timeslot in timeslots:

        if (
            section.instructor_id is not None
            and (section.instructor_id, timeslot.id) in occupied_instructors
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

            candidates.append(
                {
                    "room": room,
                    "timeslot": timeslot,
                    "score": score,
                }
            )

    # No feasible assignment found
    if not candidates:
        return None, None

    # Sort candidates by score (highest first)
    candidates.sort(
        key=lambda candidate: candidate["score"],
        reverse=True,
    )

    # Build Restricted Candidate List
    rcl = candidates[: min(RCL_SIZE, len(candidates))]

    # Randomly select one candidate
    selected = random.choice(rcl)

    return selected["room"], selected["timeslot"]


# ============================================================
# Construction Phase
# This function builds one feasible timetable using
# a randomized greedy strategy.
# ============================================================

def construct_grasp_solution(
    sections,
    rooms,
    timeslots,
):
    schedule = []

    occupied_rooms = set()
    occupied_instructors = set()

    # Precompute static candidates
    section_candidates = [
        (
            section,
            get_viable_rooms(section, rooms),
            get_valid_timeslots(section, timeslots),
        )
        for section in sections
    ]

    # Go through every section and assign a room and timeslot
    for section, viable_rooms, valid_timeslots in section_candidates:

        room, timeslot = choose_grasp_assignment(
            section,
            viable_rooms,
            valid_timeslots,
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

        # Update occupied sets
        if room is not None and timeslot is not None:

            occupied_rooms.add(
                (room.id, timeslot.id)
            )

            if section.instructor_id is not None:
                occupied_instructors.add(
                    (section.instructor_id, timeslot.id)
                )

    return schedule


# ============================================================
# Local Search
# This function improves a timetable by trying
# small changes to room or timeslot assignments.
# ============================================================

def local_search(
    schedule,
    rooms,
    timeslots,
    sections,
    valid_timeslot_cache,
):

    best_schedule = copy.deepcopy(schedule)

    best_fitness = calculate_fitness(
        best_schedule,
        rooms,
        sections=sections,
        timeslots=timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )

    # Repeat small improvements
    for iteration in range(LOCAL_SEARCH_ITERATIONS):

        any_improved = False

        # Try improving each scheduled item
        for index, item in enumerate(best_schedule):

            if item.room_id is None or item.timeslot_id is None:
                continue

            original_room = item.room_id
            original_timeslot = item.timeslot_id
            item_improved = False

            viable_rooms = get_viable_rooms_for_schedule_item(
                item,
                rooms,
            )

            random.shuffle(viable_rooms)

            random_timeslots = timeslots[:]
            random.shuffle(random_timeslots)

            # Try different room / timeslot combinations
            for room in viable_rooms:

                for timeslot in random_timeslots:

                    item.room_id = room.id
                    item.timeslot_id = timeslot.id

                    new_fitness = calculate_fitness(
                        best_schedule,
                        rooms,
                        sections=sections,
                        timeslots=timeslots,
                        valid_timeslot_cache=valid_timeslot_cache,
                    )

                    # Keep the change only if fitness improves
                    if new_fitness > best_fitness:

                        best_fitness = new_fitness
                        item_improved = True
                        any_improved = True
                        break

                if item_improved:
                    break

            # Restore if this item found no improvement
            if not item_improved:

                item.room_id = original_room
                item.timeslot_id = original_timeslot

        # Stop if no improvement was found for any item this pass
        if not any_improved:
            break

    return best_schedule


# ============================================================
# GRASP Algorithm
# Main function that runs the whole process
# ============================================================

def grasp_schedule(
    sections,
    timeslots,
    rooms,
    valid_timeslot_cache=None,
):

    # Build the timeslot cache if not already available
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(
            sections,
            timeslots,
        )

    best_schedule = None
    best_fitness = -1

    # Repeat the GRASP process many times
    for iteration in range(MAX_ITERATIONS):

        # Step 1: Construct one randomized greedy solution
        schedule = construct_grasp_solution(
            sections,
            rooms,
            timeslots,
        )

        # Step 2: Improve it using local search
        schedule = local_search(
            schedule,
            rooms,
            timeslots,
            sections,
            valid_timeslot_cache,
        )

        # Evaluate solution
        fitness = calculate_fitness(
            schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )

        # Keep the best solution found
        if fitness > best_fitness:
            best_fitness = fitness
            best_schedule = copy.deepcopy(schedule)

    return best_schedule


# ============================================================
# Multiple GRASP Runs
# Runs the algorithm several times and reports statistics.
# ============================================================

def grasp_runs(
    sections,
    timeslots,
    rooms,
    num_runs=30,
):

    fitness_scores = []

    best_overall_schedule = None
    best_overall_fitness = -1

    print(f"\n=== GRASP Algorithm: {num_runs} runs ===")
    print(
        f"Total sections: {len(sections)}, "
        f"rooms: {len(rooms)}, "
        f"timeslots: {len(timeslots)}"
    )

    valid_timeslot_cache = build_timeslot_guideline_cache(
        sections,
        timeslots,
    )

    for run in range(num_runs):

        best_schedule = grasp_schedule(
            sections,
            timeslots,
            rooms,
            valid_timeslot_cache=valid_timeslot_cache,
        )

        score = calculate_fitness(
            best_schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        )

        fitness_scores.append(score)

        if score > best_overall_fitness:

            best_overall_fitness = score
            best_overall_schedule = copy.deepcopy(best_schedule)

        scheduled = sum(
            1
            for item in best_schedule
            if item.room_id is not None
            and item.timeslot_id is not None
        )

        print(
            f"Run {run + 1:2d}: "
            f"Fitness = {score:.4f} | "
            f"Scheduled = {scheduled}"
        )

    # ========================================================
    # Statistics
    # ========================================================

    best_score = max(fitness_scores)
    worst_score = min(fitness_scores)
    avg_score = sum(fitness_scores) / len(fitness_scores)

    std_dev = (
        statistics.stdev(fitness_scores)
        if len(fitness_scores) > 1
        else 0
    )

    print("\n=== Results ===")
    print(f"Best fitness: {best_score:.4f}")
    print(f"Worst fitness: {worst_score:.4f}")
    print(f"Average fitness: {avg_score:.4f}")
    print(f"Standard deviation: {std_dev:.4f}")

    return best_overall_schedule
