"""
Constriction Particle Swarm Optimization (SPSO) with elitist
ejection-chain repair, for the university course timetabling problem.

Fits the same shared infrastructure as every other algorithm in this
package (Section / Room / Timeslot / ScheduleItem models, constraints.py,
evaluation.py's calculate_fitness) so results are directly comparable to
greedy / genetic / hybrid / GRASP on the same dataset.

Citations
---------
- Clerc, M., & Kennedy, J. (2002). "The particle swarm - explosion,
  stability, and convergence in a multidimensional complex space." IEEE
  Transactions on Evolutionary Computation, 6(1), 58-73.
  -> constriction-factor velocity update: chi = 2 / |2 - phi - sqrt(phi^2
     - 4*phi)| with phi = c1 + c2 > 4, which guarantees convergence
     without needing an explicit velocity-clamp schedule.
- Chen, R.-M., & Shih, H.-F. (2013). "Solving University Course
  Timetabling Problems Using Constriction Particle Swarm Optimization
  with Local Search." Algorithms, 6(2), 227-244.
  https://doi.org/10.3390/a6020227
  -> the encoding used here: one continuous priority value per section,
     decoded by ranking (argsort, largest-first) into a placement order.
- Moscato, P. (1989). "On Evolution, Search, Optimization, Genetic
  Algorithms and Martial Arts: Towards Memetic Algorithms." Caltech
  Concurrent Computation Program, Report 826.
  -> Lamarckian refinement applied only to the current global-best
     particle (not the whole swarm) each time it improves, which is what
     keeps this affordable at real dataset sizes (see "Why elitist-only"
     below).
- Glover, F. (1996). "Ejection chains, reference structures and
  alternating path methods for traveling salesman problems." Discrete
  Applied Mathematics, 65(1-3), 223-253.
  -> the depth-1 ejection-chain repair used to rescue sections that
     decode() left unscheduled (see rescue_unscheduled()).

Why this isn't just GRASP with extra steps
-------------------------------------------
GRASP gets its diversity from randomizing WHICH candidate is picked (the
RCL) while processing sections in a fixed, precomputed order. PSO here
gets its diversity from evolving the PROCESSING ORDER itself -- each
particle's position is a priority value per section, and decode() always
places each section into its single best still-feasible slot (no
randomness in placement). The swarm's cognitive/social dynamics search
the space of orderings, which matters because construct-in-a-fixed-order
strategies are order-sensitive: a section can end up unscheduled purely
because an earlier-processed section took its only remaining room/
timeslot, not because of any real conflict.

Why there's no interchange local search step (unlike Chen & Shih's SPSOLS)
----------------------------------------------------------------------------
Chen & Shih's decoder can produce clashing placements, so their
interchange local search exists to shuffle conflicts away after decode.
This decoder is stricter: choose_best_assignment() only ever accepts a
candidate that already passes every hard constraint against the current
occupancy, so a freshly decoded schedule has zero conflicts by
construction -- there is nothing for a conflict-shuffling local search to
fix. This was verified empirically, not assumed: running grasp.py's
incremental local_search on a decoded (and on a rescued) schedule at
production scale changed fitness by exactly 0.0 while costing ~0.5-8s per
call. The only thing pulling fitness below 1.0 here is unscheduled
sections left over from the processing order, which is exactly what
rescue_unscheduled()'s ejection chain targets instead.

Why elitist-only rescue
------------------------
Running rescue on every particle every iteration would cost n_particles x
iterations x O(unscheduled) -- at ~4700 sections that's the same kind of
combinatorial blow-up that made GRASP take 4+ hours before it was fixed
(see grasp.py's module docstring/comments). Instead, rescue runs only
when a particle's raw decode fitness beats the current global best, which
self-limits to the number of genuine improvements found (typically far
fewer than n_particles x iterations, especially as the swarm converges)
while still polishing the one schedule that's actually returned.
"""

import random
import copy
import statistics

import numpy as np

from backend.models.models import ScheduleItem
from backend.Optimization.constraints import get_valid_timeslots, get_viable_rooms
from backend.Optimization.evaluation import (
    calculate_fitness,
    build_timeslot_guideline_cache,
)
from backend.Optimization.Algorithims.grasp import scan_candidates

# ============================================================
# PSO Parameters
# ============================================================
# Chen & Shih tuned swarm=30/iterations=6000 on a ~40-event toy instance.
# At real scale (~4700 sections), decode() needs the BEST sampled
# candidate (a full scan of the sample, unlike GRASP/GA's early-exit
# construction), so each particle-evaluation costs more than a GRASP
# construction call. Sized empirically (see the PSO benchmark in
# grasp.py's sibling test suite / commit history) so num_runs x this
# budget stays in the same ballpark as GA/hybrid/GRASP on the same
# dataset, with early stopping so stagnant swarms don't pay for iterations
# that stop helping.
N_PARTICLES = 10
ITERATIONS = 14
NO_IMPROVE_LIMIT = 5

# Bounded candidate sample used by decode()'s per-section placement.
# Smaller than GRASP's CONSTRUCTION_ROOM_SAMPLE/TIMESLOT_SAMPLE (15x15)
# because decode() must score every sampled candidate to find the best one
# (no early exit), so it pays for the full sample on every section, every
# particle, every iteration -- a much larger multiplier than GRASP's
# handful of construction calls per run.
CONSTRUCTION_ROOM_SAMPLE = 8
CONSTRUCTION_TIMESLOT_SAMPLE = 8

# Bounded candidate sample used by rescue_unscheduled() to find statically
# feasible slots for a still-unscheduled section. Kept larger than the
# construction sample above since rescue only runs on the (much smaller)
# set of unscheduled items, not every section.
RESCUE_ROOM_SAMPLE = 10
RESCUE_TIMESLOT_SAMPLE = 10

C1 = 2.05
C2 = 2.05
_PHI = C1 + C2
assert _PHI > 4, "c1 + c2 must exceed 4 for the constriction factor to be defined"
CHI = 2.0 / abs(2 - _PHI - (_PHI * _PHI - 4 * _PHI) ** 0.5)  # ~0.7298 (Clerc & Kennedy 2002)
V_CLAMP = 4.0

# Position/velocity are initialised in this range, matching Chen & Shih's
# ranking decode (only relative order matters, so the absolute scale is
# arbitrary -- this range just keeps floats well-conditioned).
POSITION_RANGE = 9.0
VELOCITY_INIT_RANGE = 0.5


# ============================================================
# Decode: continuous priority vector -> discrete timetable
# ============================================================
def _bounded_candidates(
    section, rooms, timeslots,
    occupied_instructors, occupied_rooms,
    room_sample_size, timeslot_sample_size,
    valid_timeslot_cache=None,
):
    """
    Scan a bounded random sample first, falling back to the full
    room x timeslot space only if the sample turns up nothing feasible
    -- the same bounded-then-fallback pattern grasp.py uses, so a section
    is never left unscheduled just because sampling missed its only
    valid slot.
    """
    room_sample = (
        rooms if len(rooms) <= room_sample_size
        else random.sample(rooms, room_sample_size)
    )
    timeslot_sample = (
        timeslots if len(timeslots) <= timeslot_sample_size
        else random.sample(timeslots, timeslot_sample_size)
    )

    candidates = scan_candidates(
        section, room_sample, timeslot_sample,
        occupied_instructors, occupied_rooms,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    if not candidates:
        candidates = scan_candidates(
            section, rooms, timeslots,
            occupied_instructors, occupied_rooms,
            valid_timeslot_cache=valid_timeslot_cache,
        )
    return candidates


def choose_best_assignment(
    section, rooms, timeslots,
    occupied_instructors, occupied_rooms,
    valid_timeslot_cache=None,
):
    """
    Deterministic best-fit placement: unlike GRASP's randomized RCL
    choice, PSO's diversity comes from the processing order (driven by
    the particle's position), so placement itself should be
    deterministic -- otherwise the fitness landscape over positions
    would be noisy and the swarm couldn't reliably climb it.
    """
    candidates = _bounded_candidates(
        section, rooms, timeslots,
        occupied_instructors, occupied_rooms,
        CONSTRUCTION_ROOM_SAMPLE, CONSTRUCTION_TIMESLOT_SAMPLE,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    if not candidates:
        return None, None

    best = max(candidates, key=lambda c: c[0])
    return best[1], best[2]


def decode(position, section_candidates, valid_timeslot_cache):
    """
    Rank sections by position value (largest first, Chen & Shih's
    "largest position value" decoder) and place each into its best
    still-feasible slot given what's already occupied.
    """
    order = np.argsort(-position)
    n = len(section_candidates)
    schedule = [None] * n
    occupied_rooms = set()
    occupied_instructors = set()

    for idx in order:
        section, viable_rooms, valid_ts = section_candidates[idx]

        room, timeslot = choose_best_assignment(
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
        )
        schedule[idx] = item

        if room is not None and timeslot is not None:
            occupied_rooms.add((room.id, timeslot.id))
            if section.instructor_id is not None:
                occupied_instructors.add((section.instructor_id, timeslot.id))

    return schedule


# ============================================================
# Ejection-chain repair for sections decode() left unscheduled
# ============================================================
def _build_occupancy_maps(schedule):
    room_owner = {}
    instructor_owner = {}
    occupied_rooms = set()
    occupied_instructors = set()

    for idx, item in enumerate(schedule):
        if item.room_id is not None and item.timeslot_id is not None:
            key = (item.room_id, item.timeslot_id)
            room_owner[key] = idx
            occupied_rooms.add(key)
            if item.instructor_id is not None:
                ikey = (item.instructor_id, item.timeslot_id)
                instructor_owner[ikey] = idx
                occupied_instructors.add(ikey)

    return room_owner, instructor_owner, occupied_rooms, occupied_instructors


def _place(item, idx, room, timeslot, room_owner, instructor_owner, occupied_rooms, occupied_instructors):
    item.room_id = room.id
    item.timeslot_id = timeslot.id
    key = (room.id, timeslot.id)
    room_owner[key] = idx
    occupied_rooms.add(key)
    if item.instructor_id is not None:
        ikey = (item.instructor_id, timeslot.id)
        instructor_owner[ikey] = idx
        occupied_instructors.add(ikey)


def _evict(item, room_owner, instructor_owner, occupied_rooms, occupied_instructors):
    key = (item.room_id, item.timeslot_id)
    room_owner.pop(key, None)
    occupied_rooms.discard(key)
    if item.instructor_id is not None:
        ikey = (item.instructor_id, item.timeslot_id)
        instructor_owner.pop(ikey, None)
        occupied_instructors.discard(ikey)
    item.room_id = None
    item.timeslot_id = None


def rescue_unscheduled(schedule, section_candidates, valid_timeslot_cache):
    """
    Depth-1 ejection chain: for every still-unscheduled section, look at
    its statically-feasible (room, timeslot) pairs (ignoring who
    currently holds them) and, for the first one blocked by exactly one
    other section, check whether that section can immediately be
    relocated elsewhere. If so, evict it and place the unscheduled
    section in its spot -- accepted only when both end up scheduled, so
    this can never increase the unscheduled count.

    A freshly decoded schedule already has zero conflicts by
    construction (decode() never accepts a candidate that fails a hard
    constraint), so this is the only remaining lever for raising fitness
    beyond what a fixed processing order reached.
    """
    room_owner, instructor_owner, occupied_rooms, occupied_instructors = (
        _build_occupancy_maps(schedule)
    )

    unscheduled = [
        idx for idx, item in enumerate(schedule)
        if item.room_id is None or item.timeslot_id is None
    ]

    for idx in unscheduled:
        item = schedule[idx]
        section, viable_rooms, valid_ts = section_candidates[idx]

        # Ignore current occupancy here -- we want statically feasible
        # slots so we can identify who (if anyone) is blocking them.
        # Bounded + fallback, same as choose_best_assignment, since a full
        # unsampled scan here was the dominant cost in early benchmarking
        # (~7.6s per call on a ~150-unscheduled-item batch at N=4700).
        candidates = _bounded_candidates(
            section, viable_rooms, valid_ts, set(), set(),
            RESCUE_ROOM_SAMPLE, RESCUE_TIMESLOT_SAMPLE,
            valid_timeslot_cache=valid_timeslot_cache,
        )
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[0], reverse=True)

        for _, room, timeslot in candidates:
            room_blocker = room_owner.get((room.id, timeslot.id))
            instr_blocker = (
                instructor_owner.get((item.instructor_id, timeslot.id))
                if item.instructor_id is not None else None
            )

            if room_blocker is None and instr_blocker is None:
                # Genuinely free right now -- take it.
                _place(item, idx, room, timeslot, room_owner, instructor_owner,
                       occupied_rooms, occupied_instructors)
                break

            if room_blocker is not None and instr_blocker is not None and room_blocker != instr_blocker:
                continue  # two different blockers -- outside depth-1 scope

            blocker_idx = room_blocker if room_blocker is not None else instr_blocker
            if blocker_idx == idx:
                continue

            blocker = schedule[blocker_idx]
            b_section, b_rooms, b_ts = section_candidates[blocker_idx]

            trial_rooms = occupied_rooms - {(blocker.room_id, blocker.timeslot_id)}
            trial_instructors = occupied_instructors
            if blocker.instructor_id is not None:
                trial_instructors = occupied_instructors - {(blocker.instructor_id, blocker.timeslot_id)}

            if room_blocker is not None:
                # Room-block: the blocker sits at exactly (room, timeslot),
                # which is also what we just freed above -- re-mark it
                # occupied for the blocker's OWN search, or a tie would let
                # choose_best_assignment hand it right back (a no-op "move"
                # that silently defeats the rescue).
                b_ts_search = b_ts
                trial_rooms = trial_rooms | {(room.id, timeslot.id)}
            else:
                # Instructor-block: the conflict is with `timeslot` itself,
                # in any room -- excluding just (room, timeslot) wouldn't
                # stop choose_best_assignment from placing the blocker in a
                # DIFFERENT room at the same clashing timeslot.
                b_ts_search = [t for t in b_ts if t.id != timeslot.id]

            new_room, new_timeslot = choose_best_assignment(
                b_section, b_rooms, b_ts_search, trial_instructors, trial_rooms,
                valid_timeslot_cache=valid_timeslot_cache,
            )
            if new_room is None:
                continue  # blocker has nowhere else to go -- leave it alone

            _evict(blocker, room_owner, instructor_owner, occupied_rooms, occupied_instructors)
            _place(blocker, blocker_idx, new_room, new_timeslot, room_owner, instructor_owner,
                   occupied_rooms, occupied_instructors)
            _place(item, idx, room, timeslot, room_owner, instructor_owner,
                   occupied_rooms, occupied_instructors)
            break

    return schedule


# ============================================================
# Elitist refinement: rescue only, applied when a particle's raw decode
# beats the current global best (see module docstring for why there's no
# local-search step here).
# ============================================================
def _refine(schedule, section_candidates, rooms, timeslots, sections, valid_timeslot_cache):
    rescue_unscheduled(schedule, section_candidates, valid_timeslot_cache)
    fitness = calculate_fitness(
        schedule, rooms, sections=sections, timeslots=timeslots,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    return schedule, fitness


# ============================================================
# PSO Algorithm
# ============================================================
def pso_schedule(sections, timeslots, rooms, valid_timeslot_cache=None, section_candidates=None, seed=None):
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    if section_candidates is None:
        section_candidates = [
            (section, get_viable_rooms(section, rooms), get_valid_timeslots(section, timeslots))
            for section in sections
        ]

    n = len(sections)
    rng = np.random.default_rng(seed)

    positions = rng.uniform(0.0, POSITION_RANGE, size=(N_PARTICLES, n))
    velocities = rng.uniform(-VELOCITY_INIT_RANGE, VELOCITY_INIT_RANGE, size=(N_PARTICLES, n))
    pbest_pos = positions.copy()
    pbest_score = np.full(N_PARTICLES, -1.0)

    gbest_pos = positions[0].copy()
    gbest_sched = None
    gbest_score = -1.0
    no_improve_count = 0

    for _ in range(ITERATIONS):
        improved_this_iter = False

        for i in range(N_PARTICLES):
            sched = decode(positions[i], section_candidates, valid_timeslot_cache)
            score = calculate_fitness(
                sched, rooms, sections=sections, timeslots=timeslots,
                valid_timeslot_cache=valid_timeslot_cache,
            )

            if score > pbest_score[i]:
                pbest_score[i] = score
                pbest_pos[i] = positions[i].copy()

            if score > gbest_score:
                # Elitist Lamarckian refinement (Moscato 1989): only the
                # new best pays the rescue cost.
                refined_sched, refined_score = _refine(
                    sched, section_candidates, rooms, timeslots, sections, valid_timeslot_cache
                )
                gbest_score = refined_score
                gbest_pos = positions[i].copy()
                gbest_sched = copy.deepcopy(refined_sched)
                improved_this_iter = True

        no_improve_count = 0 if improved_this_iter else no_improve_count + 1
        if no_improve_count >= NO_IMPROVE_LIMIT:
            break

        # Constriction-factor velocity + position update (Clerc & Kennedy 2002)
        r1 = rng.random((N_PARTICLES, n))
        r2 = rng.random((N_PARTICLES, n))
        cognitive = C1 * r1 * (pbest_pos - positions)
        social = C2 * r2 * (gbest_pos - positions)
        velocities = CHI * (velocities + cognitive + social)
        np.clip(velocities, -V_CLAMP, V_CLAMP, out=velocities)
        positions = positions + velocities

    return gbest_sched, gbest_score


def pso_runs(sections, timeslots, rooms, num_runs=30):
    fitness_scores = []
    best_overall_schedule = None
    best_overall_fitness = -1

    print(f"\n=== PSO Algorithm: {num_runs} runs ===")
    print(f"Total sections: {len(sections)}, rooms: {len(rooms)}, timeslots: {len(timeslots)}")

    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    section_candidates = [
        (section, get_viable_rooms(section, rooms), get_valid_timeslots(section, timeslots))
        for section in sections
    ]

    for run in range(num_runs):
        best_schedule, score = pso_schedule(
            sections, timeslots, rooms,
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

        print(f"Run {run + 1:2d}: Fitness = {score:.4f} | Scheduled = {scheduled}")

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
