from functools import partial
from backend.Optimization.Algorithims.greedy import greedy_schedule
from backend.Optimization.constraints import classify_section
from backend.Optimization.Algorithims.genetic import genetic_runs
from backend.Optimization.Algorithims.hybrid import genetic_runs as hybrid_runs
from backend.Optimization.evaluation import count_scheduled_sections

# greedy is fast; genetic/hybrid cap at 1 run so the API responds
# in seconds rather than timing out on the deployed server.
ALGORITHM_REGISTRY = {
    "greedy": greedy_schedule,
    "genetic": partial(genetic_runs, num_runs=1),
    "hybrid": partial(hybrid_runs, num_runs=1),
}


class SchedulingEngine:
    """
    All algorithms must accept (sections, timeslots, rooms) and return a list
    of ScheduleItem.  Any algorithm-specific pre/post processing lives in its
    own module  not here.
    """

    def __init__(self, algorithm_name: str):
        if algorithm_name not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{algorithm_name}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
        self.algorithm_name = algorithm_name
        self.algorithm = ALGORITHM_REGISTRY[algorithm_name]
        

    def run(self, data: dict) -> list:
        """
        Run the selected algorithm against the scheduling data dict.

        Expected keys in `data`:
            sections  — list of Section objects
            timeslots — list of Timeslot objects
            rooms     — list of Room objects

        Returns a list of ScheduleItem.
        """
        if data is None:
            raise ValueError("No data provided")

        missing = [k for k in ("sections", "timeslots", "rooms") if k not in data]
        if missing:
            raise ValueError(f"Data dict is missing required keys: {missing}")

        sections = data["sections"]
        timeslots = data["timeslots"]
        rooms = data["rooms"]

        print(
            f"[SchedulingEngine] Running '{self.algorithm_name}' "
            f"on {len(sections)} sections, "
            f"{len(timeslots)} timeslots, "
            f"{len(rooms)} rooms"
        )

        
        

        result = self.algorithm(sections, timeslots, rooms)

        scheduled = count_scheduled_sections(result, sections)
        unscheduled = len(result) - scheduled
        print(
            f"[SchedulingEngine] Done — {scheduled} scheduled, {unscheduled} unscheduled"
        )

        return result, scheduled, unscheduled
