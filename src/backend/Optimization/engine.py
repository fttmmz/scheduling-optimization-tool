from backend.Optimization.Algorithims.greedy import greedy_schedule
from backend.Optimization.Algorithims.genetic import genetic_runs
from backend.Optimization.Algorithims.hybrid import genetic_runs as hybrid_runs
from backend.Optimization.Algorithims.grasp import grasp_runs
from backend.Optimization.evaluation import count_scheduled_sections

ALGORITHM_REGISTRY = {
    "greedy": greedy_schedule,
    "genetic": genetic_runs,
    "hybrid": hybrid_runs,
    "grasp": grasp_runs,
}


class SchedulingEngine:
    def __init__(self, algorithm_name: str, num_runs: int = 1):
        if algorithm_name not in ALGORITHM_REGISTRY:
            raise ValueError(
                f"Unknown algorithm '{algorithm_name}'. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
        self.algorithm_name = algorithm_name
        self.algorithm = ALGORITHM_REGISTRY[algorithm_name]
        self.num_runs = num_runs

    def run(self, data: dict) -> list:
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
            f"(num_runs={self.num_runs}) on {len(sections)} sections, "
            f"{len(timeslots)} timeslots, {len(rooms)} rooms"
        )

        if self.algorithm_name in ("genetic", "hybrid", "grasp"):
            result = self.algorithm(sections, timeslots, rooms, num_runs=self.num_runs)
        else:
            result = self.algorithm(sections, timeslots, rooms)

        scheduled = count_scheduled_sections(result, sections)
        unscheduled = len(result) - scheduled
        print(f"[SchedulingEngine] Done — {scheduled} scheduled, {unscheduled} unscheduled")

        return result, scheduled, unscheduled
