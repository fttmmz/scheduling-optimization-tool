"""Outcome-based labels: how often hybrid actually chose each candidate.

Joins the per-seed schedules written by collect_hybrid_traces.py onto the
candidate groups from build_ranking_dataset.py. A candidate's label is the share
of runs in which hybrid's FINISHED schedule placed that section there -- after
GA, tabu search and the sibling repair, and only from runs with zero hard
conflicts. That carries what the formula labels cannot: contention (a room every
section wants is often gone) and whatever the search learned about fitting the
whole timetable together.

Room groups use hybrid's own candidate list (hybrid.ROOM_LIMIT = 30), not
GRASP's 25, so hybrid's picks are inside the group they are labelled in. The
30-list contains the 25-list, so a ranker trained here still covers GRASP's.

    .venv311/Scripts/python.exe ml/build_trace_labels.py

Writes ml/data/hybrid/{room,timeslot}_candidates.csv and summary.json.
"""
import argparse
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ml"))

import build_ranking_dataset as B  # noqa: E402
from backend.Optimization.Algorithims import hybrid  # noqa: E402

TRACE_DIR = ROOT / "ml" / "data" / "traces" / "hybrid"
OUT_DIR = ROOT / "ml" / "data" / "hybrid"


def trace_label(count, runs):
    """0-4 graded relevance from a pick count.

    0 never chosen; then quarters of the runs: 1 = 1-25%, 2 = 26-50%,
    3 = 51-75%, 4 = 76-100%. Same 0-4 scale as the formula labels so the two
    are directly comparable; the raw count stays in m_hybrid_count.
    """
    return 0 if count == 0 else math.ceil(4 * count / runs)


def load_traces(trace_dir, section_ids):
    traces, rejected = [], []
    for path in sorted(trace_dir.glob("seed_*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        # Labels join on section index, so the order must be the builder's.
        assert trace["section_ids"] == section_ids, f"{path.name}: section order differs"
        # A run that left something unplaced or double-booked is not an
        # outcome worth imitating.
        if trace["hard_conflicts"] or trace["unscheduled"]:
            rejected.append(trace["seed"])
            continue
        traces.append(trace)
    return traces, rejected


def relabel(rows, traces, axis, id_column):
    """Swap the formula label for the trace label, keeping it as metadata."""
    runs = len(traces)
    out = []
    for row in rows:
        index = row["qid"] // 2
        count = sum(trace["placements"][index][axis] == row[id_column] for trace in traces)
        new = {"qid": row["qid"], "label": trace_label(count, runs)}
        new.update((k, v) for k, v in row.items() if k not in ("qid", "label"))
        new["m_formula_label"] = row["label"]
        new["m_hybrid_count"] = count
        new["m_hybrid_freq"] = round(count / runs, 4)
        out.append(new)
    return out


def diagnostics(rows, traces, axis):
    """Is the trace label telling us anything the formula did not?"""
    runs = len(traces)
    groups = defaultdict(list)
    for row in rows:
        groups[row["qid"]].append(row)

    picks_outside = 0          # run-picks that landed outside the group
    formula_best_picks = 0     # run-picks on a candidate with the group's best formula label
    modal_share = []           # how consistently hybrid picks one candidate per section
    tie_positions = []         # normalised rank of picks among tied formula-best candidates

    for rows_in_group in groups.values():
        counts = [r["m_hybrid_count"] for r in rows_in_group]
        picks = sum(counts)
        picks_outside += runs - picks
        best = max(r["m_formula_label"] for r in rows_in_group)
        formula_best_picks += sum(r["m_hybrid_count"] for r in rows_in_group
                                  if r["m_formula_label"] == best)
        if picks:
            modal_share.append(max(counts) / picks)

        # Among candidates the formula cannot tell apart, the ONLY other thing
        # hybrid sees is list order (rank_rooms/rank_timeslots sort stably, so
        # ties keep database order). If picks pile up at the front, the labels
        # partly encode that order -- an artifact, not a preference.
        tied = sorted((r for r in rows_in_group if r["m_formula_label"] == best),
                      key=lambda r: r["m_candidate_position"])
        if len(tied) >= 2:
            for rank, r in enumerate(tied):
                tie_positions.extend([rank / (len(tied) - 1)] * r["m_hybrid_count"])

    total = runs * len(groups)
    return {
        "picks_total": total,
        "picks_outside_group": picks_outside,
        "picks_on_formula_best_candidate": round(formula_best_picks / total, 4),
        "mean_modal_share": round(sum(modal_share) / len(modal_share), 4),
        "sections_with_one_candidate_every_run": sum(1 for s in modal_share if s == 1.0),
        "tied_best_pick_mean_normalised_position": (
            round(sum(tie_positions) / len(tie_positions), 4) if tie_positions else None
        ),
        "tied_best_picks_measured": len(tie_positions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace-dir", default=str(TRACE_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)

    sections, room_rows, time_rows, _stats, _by_needs = B.build(room_limit=hybrid.ROOM_LIMIT)
    section_ids = [f"{s.course.id}|{s.no}|{s.pattern_index}" for s in sections]
    traces, rejected = load_traces(pathlib.Path(args.trace_dir), section_ids)
    if not traces:
        sys.exit("no usable traces -- run ml/collect_hybrid_traces.py first")

    room_rows = relabel(room_rows, traces, 0, "m_room_id")
    time_rows = relabel(time_rows, traces, 1, "m_timeslot_id")

    B.write_csv(out_dir / "room_candidates.csv", room_rows)
    B.write_csv(out_dir / "timeslot_candidates.csv", time_rows)

    summary = {
        "label_source": "hybrid run traces",
        "runs_used": len(traces),
        "seeds_used": [t["seed"] for t in traces],
        "seeds_rejected_for_conflicts": rejected,
        "run_fitness": {"min": min(t["fitness"] for t in traces),
                        "max": max(t["fitness"] for t in traces)},
        "room_limit": hybrid.ROOM_LIMIT,
        "room": {**B.group_summary(room_rows), **diagnostics(room_rows, traces, 0)},
        "timeslot": {**B.group_summary(time_rows), **diagnostics(time_rows, traces, 1)},
        "formula_vs_trace_label_crosstab": {
            name: {f"{f}->{t}": n for (f, t), n in sorted(Counter(
                (r["m_formula_label"], r["label"]) for r in rows).items())}
            for name, rows in (("room", room_rows), ("timeslot", time_rows))
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
