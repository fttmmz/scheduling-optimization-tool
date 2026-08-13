import random
import copy
import statistics
from collections import Counter

from backend.models.models import ScheduleItem
from backend.Optimization.constraints import (
    get_valid_timeslots,
    get_viable_rooms,
    get_viable_rooms_for_schedule_item,
    passes_hard_constraints,
    get_section_campus,
    get_building_campus,
    get_required_room_type,
    room_soft_penalty,
    section_needs,
)
from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
    count_conflicts,
    count_hard_conflicts,
    count_scheduled_sections,
    count_timeslot_guideline_conflicts,
    total_soft_penalty,
    HARD_CONFLICT_WEIGHT,
    UNSCHEDULED_WEIGHT,
    SOFT_PENALTY_WEIGHT,
    SOFT_PENALTY_REFERENCE,
)

# Soft cost charged when a timeslot breaks its course-type guideline
# (duration/day). One "unit" on the same scale as constraints.room_soft_penalty.
TIMESLOT_GUIDELINE_PENALTY = 1.0

# ============================================================
# GRASP Parameters
# ============================================================
RCL_SIZE = 3
MAX_ITERATIONS = 20
LOCAL_SEARCH_ITERATIONS = 30
NO_IMPROVE_LIMIT = 5  # stop early if no improvement

# Local search tries this many (room, timeslot) candidates per item instead
# of the full room x timeslot cross product. Each candidate costs a full
# calculate_fitness() call over the whole schedule (O(sections)), so
# exhaustively trying every combination for every item, every pass, every
# GRASP restart, every run made runtime blow up combinatorially on
# real-sized datasets. Sampling bounds that cost to a small constant.
NEIGHBORHOOD_ROOM_SAMPLE = 5
NEIGHBORHOOD_TIMESLOT_SAMPLE = 5

# choose_grasp_assignment must rank candidates to build the RCL, so unlike
# genetic.py's first-match construction it can't early-exit on the first
# feasible slot -- it was scanning the full room x timeslot cross product
# for every section, every GRASP restart, every run. Sample a bounded
# subset instead, and only fall back to the full scan if the sample turns
# up nothing feasible (keeps hard-to-place sections correctly placeable).
CONSTRUCTION_ROOM_SAMPLE = 15
CONSTRUCTION_TIMESLOT_SAMPLE = 15


# ============================================================
# Candidate Evaluation
# Ranks a (room, timeslot) for the RCL. Higher is better; O(1) per candidate.
#
# This used to carry its own hand-tuned scores for capacity/room type/campus/
# department (+80 here, -150 there), a second and slightly different opinion
# about how the soft rules trade off against each other. It now derives the
# ranking from constraints.room_soft_penalty(), so every part of the system
# ranks rooms by the same weights and retuning them is a one-line change.
# ============================================================
SOFT_SCORE_SCALE = 50          # penalty units -> score points
CAPACITY_FIT_BONUS = 50        # prefer a snug room among equally-penalised ones


def score_candidate(section, room, timeslot, valid_timeslot_cache=None):
    score = -room_soft_penalty(section, room) * SOFT_SCORE_SCALE

    # Tie-breaker: among rooms that fit, prefer the least wasteful. Ignored
    # when the room is too small, since room_soft_penalty already charged for
    # the overflow.
    remaining_capacity = room.capacity - section.capacity
    if remaining_capacity >= 0:
        score += CAPACITY_FIT_BONUS - min(remaining_capacity, CAPACITY_FIT_BONUS)

    if valid_timeslot_cache is not None:
        # build_timeslot_guideline_cache keys on str(sec.no); looking up the
        # raw value missed every time whenever section numbers aren't strings,
        # which silently flattened this term to a constant.
        course_id = getattr(section.course, "id", None)
        section_no = str(getattr(section, "no", None))
        valid_ids = valid_timeslot_cache.get((course_id, section_no), set())
        if timeslot.id in valid_ids:
            score += SOFT_SCORE_SCALE * TIMESLOT_GUIDELINE_PENALTY

    return score


# ============================================================
# Section Ordering
# Schedule hardest sections first (fewest room/timeslot options) so they
# don't get stuck with whatever slots are left over once occupancy fills
# up. Computed once per run (not per construction call) since the
# room/timeslot viability of each section never changes across GRASP
# iterations.
# ============================================================
def build_section_candidates(sections, rooms, timeslots):
    """(section, needs, viable_rooms, valid_timeslots) for every section.

    A section only gets candidates along the axes it actually uses, so
    supervision/thesis sections stop being handed rooms and studio/graduate-lab
    sections stop competing for rooms they never occupy.
    """
    candidates = []
    for section in sections:
        needs = section_needs(section)
        candidates.append((
            section,
            needs,
            get_viable_rooms(section, rooms) if needs.room else [],
            get_valid_timeslots(section, timeslots) if needs.time else [],
        ))
    return candidates


def order_sections_by_difficulty(section_candidates):
    # Hardest (fewest options) first. Sections needing only one of the two axes
    # are sized by that axis alone -- multiplying by an empty list would score
    # them 0 and wrongly float them to the very front.
    def domain_size(entry):
        _section, needs, viable_rooms, valid_ts = entry
        if needs.room and needs.time:
            return len(viable_rooms) * len(valid_ts)
        if needs.room:
            return len(viable_rooms)
        if needs.time:
            return len(valid_ts)
        return 0

    return sorted(section_candidates, key=domain_size)


# ============================================================
# Candidate Scan
# ============================================================
def scan_candidates(
    section, rooms, timeslots,
    occupied_instructors, occupied_rooms,
    valid_timeslot_cache=None,
):
    candidates = []
    instructor_id = section.instructor_id

    for timeslot in timeslots:
        if (
            instructor_id is not None
            and (instructor_id, timeslot.id) in occupied_instructors
        ):
            continue

        for room in rooms:
            if (room.id, timeslot.id) in occupied_rooms:
                continue

            if not passes_hard_constraints(
                section, room, timeslot,
                occupied_instructors=occupied_instructors,
                occupied_rooms=occupied_rooms,
            ):
                continue

            score = score_candidate(
                section, room, timeslot,
                valid_timeslot_cache=valid_timeslot_cache,
            )
            candidates.append((score, room, timeslot))

    return candidates


# ============================================================
# Choose Assignment
# ============================================================
def choose_grasp_assignment(
    section, rooms, timeslots,
    occupied_instructors, occupied_rooms,
    valid_timeslot_cache=None,
):
    room_sample = (
        rooms if len(rooms) <= CONSTRUCTION_ROOM_SAMPLE
        else random.sample(rooms, CONSTRUCTION_ROOM_SAMPLE)
    )
    timeslot_sample = (
        timeslots if len(timeslots) <= CONSTRUCTION_TIMESLOT_SAMPLE
        else random.sample(timeslots, CONSTRUCTION_TIMESLOT_SAMPLE)
    )

    candidates = scan_candidates(
        section, room_sample, timeslot_sample,
        occupied_instructors, occupied_rooms,
        valid_timeslot_cache=valid_timeslot_cache,
    )

    if not candidates:
        # Sampled subset had nothing feasible -- fall back to the full scan
        # so a section doesn't go unscheduled just because sampling missed
        # its only valid slot.
        candidates = scan_candidates(
            section, rooms, timeslots,
            occupied_instructors, occupied_rooms,
            valid_timeslot_cache=valid_timeslot_cache,
        )

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    rcl = candidates[: min(RCL_SIZE, len(candidates))]
    selected = random.choice(rcl)

    return selected[1], selected[2]


# ============================================================
# Construction Phase
# section_candidates must already be ordered (see order_sections_by_
# difficulty) -- ordering is done once by the caller and reused across
# every GRASP iteration/run instead of re-sorting on every call.
# ============================================================
def construct_grasp_solution(
    sections, rooms, timeslots,
    section_candidates,
    valid_timeslot_cache=None,
):
    schedule = []
    occupied_rooms = set()
    occupied_instructors = set()

    for section, needs, viable_rooms, valid_ts in section_candidates:

        if not needs.room and not needs.time:
            room = timeslot = None
        elif not needs.room:
            # Time only: pick the first slot that leaves the instructor free.
            room = None
            timeslot = next(
                (
                    ts for ts in valid_ts
                    if section.instructor_id is None
                    or (section.instructor_id, ts.id) not in occupied_instructors
                ),
                valid_ts[0] if valid_ts else None,
            )
        elif not needs.time:
            room = viable_rooms[0] if viable_rooms else None
            timeslot = None
        else:
            room, timeslot = choose_grasp_assignment(
                section, viable_rooms, valid_ts,
                occupied_instructors, occupied_rooms,
                valid_timeslot_cache=valid_timeslot_cache,
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
            # Classification is keyed on level; without it a detached item
            # looks like an ordinary room+time lecture.
            level=getattr(section.course, "level", None),
            course_class=getattr(section.course, "course_class", None),
        )

        schedule.append(item)

        # A roomless section still consumes its instructor's time, so book the
        # instructor whenever a timeslot was assigned -- not only when a room was.
        if timeslot is not None:
            if room is not None:
                occupied_rooms.add((room.id, timeslot.id))
            if section.instructor_id is not None:
                occupied_instructors.add((section.instructor_id, timeslot.id))

    return schedule


# ============================================================
# Per-item conflict check
# Of the 7 conflict types in count_conflicts(), 5 depend only on a single
# item's own (room, timeslot) -- not on any other item in the schedule:
# campus, room type, department, capacity, timeslot guidelines. This
# computes just those 5 for one item in O(1), instead of the O(sections)
# cost of rescanning the whole schedule via the count_*_conflicts helpers.
# ============================================================
def _item_soft_penalty(item, room, timeslot_id, valid_timeslot_cache):
    """Weighted SOFT cost of one item's own (room, timeslot).

    Was _item_local_conflicts(), which charged exactly 1 per broken rule -- so
    a lab in a classroom cost the same as a room 2 seats short, and five soft
    blemishes outweighed a double-booking. It now uses the shared
    room_soft_penalty() weights, keeping the O(1) cost that makes local search
    affordable while agreeing with how the fitness function scores.
    """
    penalty = room_soft_penalty(item, room)

    valid_ids = valid_timeslot_cache.get((item.course_id, item.section), set())
    if timeslot_id not in valid_ids:
        penalty += TIMESLOT_GUIDELINE_PENALTY

    return penalty


# ============================================================
# Local Search
#
# The only two conflict types that depend on OTHER items in the schedule
# are instructor and room double-booking. Those are tracked incrementally
# via occupancy counters (room_occupancy / instructor_occupancy) instead of
# recounting the whole schedule: count_room_conflicts() sums
# max(0, occupants_at_slot - 1) across slots, so removing an occupant from
# a slot with >=2 occupants reduces the total by exactly 1 (0 otherwise),
# and adding one to a slot with >=1 existing occupant increases it by
# exactly 1 (0 otherwise) -- an exact O(1) equivalent of the O(sections)
# full recount, combined with the O(1) per-item check above. This makes
# evaluating one candidate move O(1) instead of O(sections), which is what
# was making runs take hours on real-sized datasets.
#
# Best-improvement: each item scans its full sampled neighborhood (still
# bounded to NEIGHBORHOOD_ROOM_SAMPLE x NEIGHBORHOOD_TIMESLOT_SAMPLE
# candidates -- same O(1) cost per item as before) and takes the best move
# found instead of the first improving one, which gives noticeably better
# converged fitness for the same per-candidate cost. Item order is
# reshuffled every pass for diversity.
# ============================================================
def local_search(
    schedule,
    rooms,
    timeslots,
    sections,
    valid_timeslot_cache,
):
    best_schedule = schedule
    room_map = {r.id: r for r in rooms}

    total_sections = len(sections)
    scheduled_count = count_scheduled_sections(best_schedule, sections)
    unscheduled = max(0, total_sections - scheduled_count)
    # local_search only relocates already-scheduled items -- it never
    # schedules or unschedules anything -- so this term never changes for
    # the whole call and only needs to be computed once.
    unscheduled_penalty = (
        (unscheduled / total_sections) * UNSCHEDULED_WEIGHT if total_sections else 0.0
    )

    # Hard and soft are tracked SEPARATELY so local search can never trade a
    # double-booking for a handful of soft blemishes -- the flat conflict count
    # this used to run on made those exactly interchangeable.
    total_hard = count_hard_conflicts(best_schedule)
    total_soft = total_soft_penalty(best_schedule, rooms) + (
        count_timeslot_guideline_conflicts(
            best_schedule, sections, timeslots,
            valid_timeslot_cache=valid_timeslot_cache,
        ) * TIMESLOT_GUIDELINE_PENALTY
    )

    def fitness_from(hard, soft):
        """Mirrors evaluation.calculate_fitness so the score GRASP optimizes is
        the score it is ultimately judged by."""
        if not total_sections:
            return round(1.0 / (1.0 + unscheduled_penalty), 4)
        hard_penalty = (hard / total_sections) * HARD_CONFLICT_WEIGHT
        soft_p = (soft / (total_sections * SOFT_PENALTY_REFERENCE)) * SOFT_PENALTY_WEIGHT
        return round(1.0 / (1.0 + unscheduled_penalty + hard_penalty + soft_p), 4)

    best_fitness = fitness_from(total_hard, total_soft)

    room_occupancy = Counter()
    instructor_occupancy = Counter()
    for it in best_schedule:
        if it.room_id is not None and it.timeslot_id is not None:
            room_occupancy[(it.room_id, it.timeslot_id)] += 1
            if it.instructor_id is not None:
                instructor_occupancy[(it.instructor_id, it.timeslot_id)] += 1

    # Pre-shuffle timeslots once per pass
    timeslot_list = timeslots[:]

    for _ in range(LOCAL_SEARCH_ITERATIONS):

        improved = False
        random.shuffle(timeslot_list)

        item_order = list(best_schedule)
        random.shuffle(item_order)

        for item in item_order:

            if item.room_id is None or item.timeslot_id is None:
                continue

            original_room_id = item.room_id
            original_timeslot_id = item.timeslot_id
            original_room = room_map.get(original_room_id)
            if original_room is None:
                continue

            old_local = _item_soft_penalty(
                item, original_room, original_timeslot_id, valid_timeslot_cache
            )

            # Sample a bounded neighborhood instead of scanning every
            # room x timeslot pair
            viable_rooms = get_viable_rooms_for_schedule_item(item, rooms)
            room_sample = random.sample(
                viable_rooms, min(NEIGHBORHOOD_ROOM_SAMPLE, len(viable_rooms))
            )
            timeslot_sample = random.sample(
                timeslot_list, min(NEIGHBORHOOD_TIMESLOT_SAMPLE, len(timeslot_list))
            )

            best_delta = 0
            best_move = None

            for room in room_sample:
                for timeslot in timeslot_sample:

                    if room.id == original_room_id and timeslot.id == original_timeslot_id:
                        continue

                    new_local = _item_soft_penalty(
                        item, room, timeslot.id, valid_timeslot_cache
                    )
                    soft_delta = new_local - old_local
                    hard_delta = 0

                    old_room_count = room_occupancy[(original_room_id, original_timeslot_id)]
                    if old_room_count >= 2:
                        hard_delta -= 1

                    if item.instructor_id is not None:
                        old_instr_count = instructor_occupancy[
                            (item.instructor_id, original_timeslot_id)
                        ]
                        if old_instr_count >= 2:
                            hard_delta -= 1

                    new_room_count = room_occupancy[(room.id, timeslot.id)]
                    if new_room_count >= 1:
                        hard_delta += 1

                    if item.instructor_id is not None:
                        new_instr_count = instructor_occupancy[
                            (item.instructor_id, timeslot.id)
                        ]
                        if new_instr_count >= 1:
                            hard_delta += 1

                    # Compare moves on the same weighting the fitness uses; the
                    # shared 1/total_sections factor cancels out.
                    delta = (
                        hard_delta * HARD_CONFLICT_WEIGHT
                        + soft_delta * SOFT_PENALTY_WEIGHT / SOFT_PENALTY_REFERENCE
                    )

                    if delta < best_delta:
                        best_delta = delta
                        best_move = (room, timeslot, hard_delta, soft_delta)

            if best_move is not None:
                room, timeslot, hard_delta, soft_delta = best_move
                new_total_hard = total_hard + hard_delta
                new_total_soft = total_soft + soft_delta
                new_fitness = fitness_from(new_total_hard, new_total_soft)

                if new_fitness > best_fitness:
                    # Commit: update occupancy, move the item, update the
                    # running totals. Nothing is mutated unless a candidate
                    # is actually accepted, so there's no restore-on-
                    # failure needed.
                    room_occupancy[(original_room_id, original_timeslot_id)] -= 1
                    room_occupancy[(room.id, timeslot.id)] += 1
                    if item.instructor_id is not None:
                        instructor_occupancy[
                            (item.instructor_id, original_timeslot_id)
                        ] -= 1
                        instructor_occupancy[(item.instructor_id, timeslot.id)] += 1

                    item.room_id = room.id
                    item.timeslot_id = timeslot.id

                    total_hard = new_total_hard
                    total_soft = new_total_soft
                    best_fitness = new_fitness
                    improved = True

        if not improved:
            break  # early exit — no point continuing

    return best_schedule, best_fitness


# ============================================================
# GRASP Algorithm
# ============================================================
def grasp_schedule(
    sections,
    timeslots,
    rooms,
    valid_timeslot_cache=None,
    section_candidates=None,  # pass in precomputed, pre-ordered candidates
):
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    if section_candidates is None:
        section_candidates = order_sections_by_difficulty(
            build_section_candidates(sections, rooms, timeslots)
        )

    best_schedule = None
    best_fitness = -1
    no_improve_count = 0

    for _ in range(MAX_ITERATIONS):

        schedule = construct_grasp_solution(
            sections,
            rooms,
            timeslots,
            section_candidates,  # reuse precomputed, pre-ordered candidates
            valid_timeslot_cache=valid_timeslot_cache,
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

    # Precompute section candidates ONCE for all runs, ordered hardest
    # (fewest viable rooms/timeslots) first so they aren't left with
    # whatever slots remain once occupancy fills up.
    section_candidates = order_sections_by_difficulty(
        build_section_candidates(sections, rooms, timeslots)
    )

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
