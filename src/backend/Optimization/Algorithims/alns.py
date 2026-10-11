"""Adaptive Large Neighbourhood Search (ALNS) for university timetabling.

This implementation uses the Python ``alns`` package:
https://github.com/N-Wouda/ALNS

Reference:

Wouda, N. A. and Lan, L. (2023).
"ALNS: a Python implementation of the adaptive large neighbourhood search
metaheuristic."
Journal of Open Source Software, 8(81), 5028.
DOI: 10.21105/joss.05028

The destroy and repair operators are adapted to the university course
timetabling problem used in this project.
"""

from heapq import nsmallest

import numpy as np
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations

from backend.Optimization.benchmark_seeds import seed_for_run
from backend.Optimization.constraints import (
    finalize_schedule,
    get_valid_timeslots_for_section,
    get_viable_rooms,
    instructor_occupancy_ids,
    passes_hard_constraints,
    room_soft_penalty,
    section_needs,
    time_soft_penalty,
)
from backend.Optimization.evaluation import (
    build_timeslot_guideline_cache,
    calculate_fitness,
)
from backend.models.models import ScheduleItem


# ---------------------------------------------------------------------------
# ALNS parameters
# ---------------------------------------------------------------------------

# Remove 5% of the scheduled sections during each destroy operation.
DESTROY_FRACTION = 0.05

# Stop one ALNS run after 100 destroy-and-repair iterations.
MAX_ITERATIONS = 100

# Greedy repair uses the best choice. Randomized repair selects from the
# three best choices to give the search some variety.
TOP_REPAIR_CHOICES = 3


# ---------------------------------------------------------------------------
# Schedule conversion
# ---------------------------------------------------------------------------

def make_schedule_item(section, room=None, timeslot=None):
    """Convert one assignment to the project's ScheduleItem model."""
    return ScheduleItem.from_section(
        section,
        room_id=room.id if room else None,
        timeslot_id=timeslot.id if timeslot else None,
    )


# ---------------------------------------------------------------------------
# ALNS solution state
# ---------------------------------------------------------------------------

class TimetableState:
    """ALNS solution state: one ``(room, timeslot)`` pair per section."""

    def __init__(
        self,
        sections,
        timeslots,
        rooms,
        assignments=None,
        valid_timeslot_cache=None,
        room_cache=None,
        time_cache=None,
    ):
        self.sections = sections
        self.timeslots = timeslots
        self.rooms = rooms

        # Each list position belongs to the section at the same index.
        # (None, None) means that the section has not been scheduled yet.
        self.assignments = (
            assignments
            if assignments is not None
            else [(None, None) for _ in sections]
        )
        self.valid_timeslot_cache = valid_timeslot_cache

        # These candidate lists never change during one ALNS run.
        self.room_cache = room_cache if room_cache is not None else {}
        self.time_cache = time_cache if time_cache is not None else {}
        self._objective = None

    def copy(self):
        """Create the independent state required by ALNS operators."""
        return TimetableState(
            self.sections,
            self.timeslots,
            self.rooms,
            assignments=self.assignments.copy(),
            valid_timeslot_cache=self.valid_timeslot_cache,
            room_cache=self.room_cache,
            time_cache=self.time_cache,
        )

    def candidate_rooms(self, index):
        """Return and cache ranked candidate rooms for one section."""
        if index not in self.room_cache:
            section = self.sections[index]
            self.room_cache[index] = (
                get_viable_rooms(section, self.rooms)
                if section_needs(section).room
                else [None]
            )
        return self.room_cache[index]

    def candidate_timeslots(self, index):
        """Return and cache valid timeslots for one section."""
        if index not in self.time_cache:
            section = self.sections[index]
            self.time_cache[index] = (
                get_valid_timeslots_for_section(section, self.timeslots)
                if section_needs(section).time
                else [None]
            )
        return self.time_cache[index]

    def unassigned_indices(self):
        """Find sections that are missing a required room or timeslot."""
        result = []
        for index, section in enumerate(self.sections):
            room, timeslot = self.assignments[index]
            needs = section_needs(section)
            if (needs.room and room is None) or (needs.time and timeslot is None):
                result.append(index)
        return result

    def to_schedule(self):
        """Convert this state to the project's ScheduleItem format."""
        return [
            make_schedule_item(section, room, timeslot)
            for section, (room, timeslot) in zip(
                self.sections, self.assignments
            )
        ]

    def objective(self):
        """Convert the maximized project fitness to an ALNS minimum."""
        # A state does not change after evaluation, so calculate once and reuse.
        if self._objective is None:
            fitness = calculate_fitness(
                self.to_schedule(),
                self.rooms,
                sections=self.sections,
                timeslots=self.timeslots,
                valid_timeslot_cache=self.valid_timeslot_cache,
            )
            self._objective = 1.0 - fitness
        return self._objective


# ---------------------------------------------------------------------------
# Repair operators
# ---------------------------------------------------------------------------

def repair(state, rng, randomized=False):
    """Repair removed sections using the best or a random good position."""
    repaired = state.copy()
    occupied_rooms = set()
    occupied_instructors = set()
    sibling_times = {}

    # Rebuild occupancy from assignments that remain after destruction.
    for section, (room, timeslot) in zip(
        repaired.sections, repaired.assignments
    ):
        if room is not None and timeslot is not None:
            occupied_rooms.add((room.id, timeslot.id))

        if timeslot is not None:
            for instructor_id in instructor_occupancy_ids(section):
                occupied_instructors.add((instructor_id, timeslot.id))

            sibling_key = getattr(section, "sibling_key", None)
            if sibling_key is not None:
                sibling_times.setdefault(sibling_key, set()).add(timeslot.id)

    unassigned = repaired.unassigned_indices()

    # A different insertion order helps different iterations explore new states.
    rng.shuffle(unassigned)

    for index in unassigned:
        section = repaired.sections[index]
        sibling_key = getattr(section, "sibling_key", None)
        candidates = []

        for room in repaired.candidate_rooms(index):
            for timeslot in repaired.candidate_timeslots(index):
                # Reject room and instructor conflicts.
                if not passes_hard_constraints(
                    section,
                    room,
                    timeslot,
                    occupied_instructors=occupied_instructors,
                    occupied_rooms=occupied_rooms,
                ):
                    continue

                # Different meeting blocks of one section cannot overlap.
                if (
                    sibling_key is not None
                    and timeslot is not None
                    and timeslot.id in sibling_times.get(sibling_key, set())
                ):
                    continue

                # Soft penalties rank the feasible placements.
                cost = 0.0
                if room is not None:
                    cost += room_soft_penalty(section, room)
                if timeslot is not None:
                    cost += time_soft_penalty(section, timeslot)
                candidates.append((cost, room, timeslot))

        choices = nsmallest(
            TOP_REPAIR_CHOICES,
            candidates,
            key=lambda candidate: candidate[0],
        )
        if not choices:
            # Leave the section unassigned if no feasible position exists.
            continue

        # Greedy repair takes the best. Randomized repair uses a top-three choice.
        choice = int(rng.integers(len(choices))) if randomized else 0
        _, room, timeslot = choices[choice]
        repaired.assignments[index] = (room, timeslot)

        # Record the new assignment to protect later placements from conflicts.
        if room is not None and timeslot is not None:
            occupied_rooms.add((room.id, timeslot.id))

        if timeslot is not None:
            for instructor_id in instructor_occupancy_ids(section):
                occupied_instructors.add((instructor_id, timeslot.id))

            if sibling_key is not None:
                sibling_times.setdefault(sibling_key, set()).add(timeslot.id)

    return repaired


def greedy_repair(state, rng):
    """Insert each removed section in its best feasible position."""
    return repair(state, rng)


def randomized_repair(state, rng):
    """Insert each removed section into one of its three best positions."""
    return repair(state, rng, randomized=True)


# ---------------------------------------------------------------------------
# Destroy operators
# ---------------------------------------------------------------------------

def random_removal(state, rng):
    """Remove a small random group of assigned sections."""
    destroyed = state.copy()
    assigned = [
        index
        for index, (room, timeslot) in enumerate(destroyed.assignments)
        if room is not None or timeslot is not None
    ]

    if assigned:
        number_to_remove = max(
            1, int(len(assigned) * DESTROY_FRACTION)
        )
        selected = rng.choice(
            assigned,
            size=min(number_to_remove, len(assigned)),
            replace=False,
        )

        for index in selected:
            destroyed.assignments[int(index)] = (None, None)

    return destroyed


def worst_removal(state, rng):
    """Remove assignments with the largest room and time penalties."""
    del rng  # Required by the ALNS operator interface.

    destroyed = state.copy()
    scored = []

    for index, (section, assignment) in enumerate(
        zip(destroyed.sections, destroyed.assignments)
    ):
        room, timeslot = assignment
        if room is None and timeslot is None:
            continue

        cost = room_soft_penalty(section, room) if room is not None else 0.0
        if timeslot is not None:
            cost += time_soft_penalty(section, timeslot)
        scored.append((cost, index))

    # Remove the highest-cost assignments so repair can improve them.
    scored.sort(reverse=True)
    number_to_remove = max(1, int(len(scored) * DESTROY_FRACTION))
    for _, index in scored[:number_to_remove]:
        destroyed.assignments[index] = (None, None)

    return destroyed


# ---------------------------------------------------------------------------
# Main ALNS search
# ---------------------------------------------------------------------------

def alns_schedule(
    sections,
    timeslots,
    rooms,
    valid_timeslot_cache=None,
    seed=None,
    max_iterations=MAX_ITERATIONS,
):
    """Run one ALNS search and return its best timetable."""
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(
            sections, timeslots
        )

    # This generator is shared by our operators and the ALNS package.
    rng = np.random.default_rng(seed)

    # Build the initial solution with the same repair logic used by ALNS.
    empty = TimetableState(
        sections,
        timeslots,
        rooms,
        valid_timeslot_cache=valid_timeslot_cache,
    )
    initial = randomized_repair(empty, rng)

    search = ALNS(rng)
    search.add_destroy_operator(random_removal)
    search.add_destroy_operator(worst_removal)
    search.add_repair_operator(greedy_repair)
    search.add_repair_operator(randomized_repair)

    # Better operators receive higher weights and become more likely to run.
    selector = RouletteWheel(
        scores=[5, 3, 1, 0],  # Global best, improvement, accepted, rejected.
        decay=0.8,  # Keep 80% of the old weight and use 20% of the new score.
        num_destroy=2,
        num_repair=2,
    )

    # Allow limited worsening early, then reduce the allowed gap to zero.
    acceptance = RecordToRecordTravel.autofit(
        # Prevent a zero objective from producing a zero acceptance gap.
        init_obj=max(initial.objective(), 1e-6),
        start_gap=0.05,
        end_gap=0.0,
        num_iters=max_iterations,
    )

    result = search.iterate(
        initial,
        selector,
        acceptance,
        MaxIterations(max_iterations),
    )

    # Separate overlapping sibling blocks and try to give them the same room.
    schedule = result.best_state.to_schedule()
    return finalize_schedule(
        schedule,
        sections,
        timeslots,
        valid_timeslot_cache,
        rooms=rooms,
    )


# ---------------------------------------------------------------------------
# Multiple independent runs
# ---------------------------------------------------------------------------

def alns_runs(sections, timeslots, rooms, num_runs=1, seeds=None):
    """Run ALNS several times and return the best timetable."""
    valid_cache = build_timeslot_guideline_cache(sections, timeslots)
    best_schedule = None
    best_fitness = float("-inf")

    for run in range(num_runs):
        # Use the same benchmark-seed protocol as the other algorithms.
        run_seed = seed_for_run(run, seeds)
        schedule = alns_schedule(
            sections,
            timeslots,
            rooms,
            valid_timeslot_cache=valid_cache,
            seed=run_seed,
        )
        fitness = calculate_fitness(
            schedule,
            rooms,
            sections=sections,
            timeslots=timeslots,
            valid_timeslot_cache=valid_cache,
        )

        if fitness > best_fitness:
            best_fitness = fitness
            best_schedule = schedule

    return best_schedule