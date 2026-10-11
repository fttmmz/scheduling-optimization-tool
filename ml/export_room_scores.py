"""Score every (section, room) candidate with a trained room ranker, offline.

Every room feature is static -- it depends on the section and the room, never
on the rest of the schedule -- so a model's score for a pair never changes
during a run. Scoring every pair once here means GRASP only needs a lookup
table: no xgboost, pandas or feature code in the backend.

    .venv311/Scripts/python.exe ml/export_room_scores.py --source manual
    .venv311/Scripts/python.exe ml/export_room_scores.py --source trace

Writes <data dir>/room_scores.csv: m_section_id, m_room_id, score. A score only
means something compared with the OTHER rooms of the same section (a ranker's
output is relative within its group), which is exactly how GRASP uses it.
"""
import argparse
import json
import pathlib
import sys

import pandas as pd
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from train_ranker import encode  # noqa: E402

SOURCES = {
    # label source: (candidate file dir, model dir, model file)
    "manual": ("ml/data/manual", "ml/models/manual", "room_manual_ranker.json"),
    "trace": ("ml/data/hybrid", "ml/models", "room_trace_ranker.json"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=SOURCES, required=True)
    args = parser.parse_args()
    data, models, model_file = SOURCES[args.source]
    data_dir, model_dir = ROOT / data, ROOT / models

    df = pd.read_csv(data_dir / "room_candidates.csv", dtype={"m_section_no": str},
                     keep_default_na=False,
                     na_values={"f_capacity_slack": [""], "f_capacity_utilisation": [""]})
    spec = json.loads((model_dir / "room_feature_spec.json").read_text())
    model = xgb.XGBRanker()
    model.load_model(model_dir / model_file)

    df["score"] = model.predict(encode(df, spec))
    out = data_dir / "room_scores.csv"
    df[["m_section_id", "m_room_id", "score"]].to_csv(out, index=False)
    print(f"{len(df)} scores, {df['m_section_id'].nunique()} sections -> {out}")


if __name__ == "__main__":
    main()
