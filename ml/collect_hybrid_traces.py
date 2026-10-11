"""Run hybrid once per shared benchmark seed and record every placement.

hybrid.genetic_runs() keeps only the best schedule of its runs; this keeps ALL
of them, one JSON file per seed, so outcome-based labels can be built from how
often a (section, room) or (section, timeslot) pair survives into a finished,
conflict-free schedule. Same fixtures and section model as
build_ranking_dataset.py.

Resumable: a seed whose file already exists is skipped, so an interrupted
sweep picks up where it stopped.

    .venv311/Scripts/python.exe ml/collect_hybrid_traces.py            # seeds 1..30
    .venv311/Scripts/python.exe ml/collect_hybrid_traces.py --seeds 1 2 3
"""
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ml"))

from build_ranking_dataset import load_data  # noqa: E402
from backend.Optimization import constraints as C  # noqa: E402
from backend.Optimization.Algorithims import hybrid  # noqa: E402
from backend.Optimization.benchmark_seeds import SEEDS, seed_all  # noqa: E402
from backend.Optimization.evaluation import (  # noqa: E402
    build_timeslot_guideline_cache,
    calculate_fitness,
    count_hard_conflicts,
    total_soft_penalty,
    total_time_penalty,
)

OUT_DIR = ROOT / "ml" / "data" / "traces" / "hybrid"


def run_one(seed, sections, timeslots, rooms, cache):
    seed_all(seed)
    start = time.perf_counter()
    schedule = hybrid.hybrid_schedule(sections, timeslots, rooms, cache)
    runtime = time.perf_counter() - start

    # hybrid builds its schedule as [make_item(s) for s in sections] and every
    # later stage edits in place, so item i IS section i. Asserted, because the
    # labels are joined on that index.
    assert len(schedule) == len(sections)
    for item, section in zip(schedule, sections):
        assert (item.course_id, str(item.section), item.pattern_index) == \
               (section.course.id, str(section.no), section.pattern_index)

    unscheduled = sum(
        not hybrid.is_scheduled(item, C.classify_section(section))
        for item, section in zip(schedule, sections)
    )
    return {
        "algorithm": "hybrid",
        "seed": seed,
        "runtime_s": round(runtime, 2),
        "hard_conflicts": count_hard_conflicts(schedule),
        "unscheduled": unscheduled,
        "room_soft_penalty": total_soft_penalty(schedule, rooms),
        "time_soft_penalty": total_time_penalty(schedule, timeslots),
        "fitness": calculate_fitness(schedule, rooms, sections=sections,
                                     timeslots=timeslots, valid_timeslot_cache=cache),
        # Aligned with build_ranking_dataset's section order: index i here is
        # qid // 2 there.
        "section_ids": [f"{s.course.id}|{s.no}|{s.pattern_index}" for s in sections],
        "placements": [[item.room_id, item.timeslot_id] for item in schedule],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections, rooms, timeslots, _manual = load_data()
    cache = build_timeslot_guideline_cache(sections, timeslots)

    for seed in args.seeds:
        path = out_dir / f"seed_{seed:02d}.json"
        if path.exists():
            print(f"[traces] seed {seed}: exists, skipping", flush=True)
            continue
        trace = run_one(seed, sections, timeslots, rooms, cache)
        path.write_text(json.dumps(trace), encoding="utf-8")
        print(f"[traces] seed {seed}: {trace['runtime_s']}s hard={trace['hard_conflicts']} "
              f"unscheduled={trace['unscheduled']} fitness={trace['fitness']:.4f}", flush=True)


if __name__ == "__main__":
    main()
