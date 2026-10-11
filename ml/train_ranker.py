"""Train XGBoost learning-to-rank models for the room and timeslot decisions.

Four models, one per (axis, label source), all on the trace-labelled files in
ml/data/hybrid/ so every model sees exactly the same groups and the same split:

    label source  column             what it teaches
    trace         label              what hybrid actually chose over 30 runs
    formula       m_formula_label    the banded soft penalty -- a CALIBRATION run:
                                     its labels are a pure function of the features,
                                     so it should recover the hand weights. If it
                                     does, a trace model that weights things
                                     differently is telling us something real.

Objective is rank:ndcg (LambdaMART): the labels are graded 0-4 integers, and
NDCG uses the grade differences where rank:pairwise would only use their order.

The split is by COURSE, not by row or by group. Room features carry no room
identity, so sections with the same profile produce identical groups (786
distinct of 2,722); a per-group split puts exact copies of test groups in the
training set. Splitting by course keeps every group -- and every sibling block
of a section -- whole on one side. A course lands in the same fold on both axes.

Scores that tie are broken RANDOMLY and the metrics averaged over TIE_DRAWS
draws. Breaking ties by list position would quietly hand every scorer the
formula's own ordering.

    .venv311/Scripts/python.exe ml/train_ranker.py

Writes ml/models/{room,timeslot}_{trace,formula}_ranker.json, one
{axis}_feature_spec.json per axis (feature order + category levels -- needed to
encode candidates identically at inference time), and training_report.json.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "ml" / "data" / "hybrid"
MODEL_DIR = ROOT / "ml" / "models"

SEED = 0
TEST_SHARE, VALID_SHARE = 0.2, 0.1
TIE_DRAWS = 10

PARAMS = dict(
    objective="rank:ndcg",
    tree_method="hist",
    enable_categorical=True,
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    lambdarank_pair_method="topk",
    lambdarank_num_pair_per_sample=10,
    random_state=SEED,
)

# Model size is chosen on the VALIDATION courses by NDCG@3 -- GRASP's restricted
# candidate list is RCL_SIZE = 3, so the top three are what it acts on -- with
# ties broken randomly, exactly like the test metrics.
#
# Not XGBoost's built-in early stopping, which failed here: on the formula
# labels NDCG stops moving after one tree, so it kept a single-tree model --
# too coarse to express four graded rules, which is the one thing the
# calibration run exists to show. Its NDCG also breaks score ties by row order.
CHECKPOINTS = [1, 5, 10, 25] + list(range(50, PARAMS["n_estimators"] + 1, 25))
SELECTION_TIE_DRAWS = 3

LABELS = {"trace": "label", "formula": "m_formula_label"}

# rules:   hand-weighted soft rule -> the penalty column that says a candidate
#          breaks it (used to describe where the model and the formula differ).
# implied: the terms of the within-section regression in implied_weights():
#          (name, value from the rows, hand weight or None for terms the hand
#          formula does not have). Capacity is split the way the formula
#          prices it: a flat 2.0 for overflowing at all, plus 4.0 x overflow.
AXES = {
    "room": {
        "file": "room_candidates.csv",
        "formula_score": "m_grasp_room_score",
        "penalty": "m_soft_penalty",
        "rules": {
            "room_type": "m_penalty_room_type",
            "campus": "m_penalty_campus",
            "capacity": "m_penalty_capacity",
            "department": "m_penalty_department",
        },
        "implied": [
            ("room_type", lambda d: d["m_penalty_room_type"] > 0, 4.0),
            ("campus", lambda d: d["m_penalty_campus"] > 0, 3.0),
            ("capacity (overflowing at all)", lambda d: d["m_penalty_capacity"] > 0, 2.0),
            ("capacity (per unit overflow ratio)", lambda d: d["f_capacity_overflow_ratio"], 4.0),
            ("department", lambda d: d["m_penalty_department"] > 0, 1.0),
            ("demand: +100 sections that could use it", lambda d: d["f_room_demand"] / 100, None),
            ("demand: +100 sections it perfectly fits",
             lambda d: d["f_room_perfect_demand"] / 100, None),
            ("+10 spare seats", lambda d: d["f_capacity_slack"].fillna(0) / 10, None),
        ],
        "feature_groups": {
            "room type": ["f_room_type_match", "f_room_type"],
            "campus": ["f_campus_match", "f_room_campus"],
            "capacity": ["f_capacity_overflow_ratio", "f_capacity_slack",
                         "f_capacity_utilisation", "f_capacity_unknown", "f_room_capacity"],
            "department": ["f_dept_match", "f_room_has_dept", "f_room_access"],
            "demand (contention)": ["f_room_demand", "f_room_perfect_demand"],
            "room identity (building, owner)": ["f_room_building", "f_room_dept"],
        },
    },
    "timeslot": {
        "file": "timeslot_candidates.csv",
        "formula_score": "m_grasp_time_score",
        "penalty": "m_time_penalty",
        "rules": {
            "supervision_daytime": "m_penalty_supervision_daytime",
            "teaching_evening": "m_penalty_teaching_evening",
        },
        "implied": [
            ("supervision_daytime", lambda d: d["m_penalty_supervision_daytime"] > 0, 3.0),
            ("teaching_evening", lambda d: d["m_penalty_teaching_evening"] > 0, 2.0),
            ("demand: +100 sections that could use it", lambda d: d["f_slot_demand"] / 100, None),
            ("demand: +100 sections it suits perfectly",
             lambda d: d["f_slot_perfect_demand"] / 100, None),
            ("start 1 hour later", lambda d: d["f_start_minute"] / 60, None),
            ("1 hour longer", lambda d: d["f_duration_minutes"] / 60, None),
        ],
        "feature_groups": {
            "time of day": ["f_is_evening", "f_start_minute", "f_end_minute"],
            "day pattern / length": ["f_day_pattern", "f_n_days", "f_duration_minutes",
                                     "f_day_U", "f_day_M", "f_day_T", "f_day_W",
                                     "f_day_R", "f_day_F"],
            "demand (contention)": ["f_slot_demand", "f_slot_perfect_demand"],
        },
    },
}


# ── Data ─────────────────────────────────────────────────────────────────────

def load(axis):
    df = pd.read_csv(DATA_DIR / AXES[axis]["file"], dtype={"m_section_no": str},
                     keep_default_na=False, na_values={"f_capacity_slack": [""],
                                                       "f_capacity_utilisation": [""]})
    assert df["qid"].is_monotonic_increasing, "XGBoost needs rows sorted by qid"
    return df


def feature_spec(df):
    features = [c for c in df.columns if c.startswith("f_")]
    categories = {c: sorted(df[c].astype(str).unique().tolist())
                  for c in features if not pd.api.types.is_numeric_dtype(df[c])}
    return {"features": features, "categories": categories}


def encode(df, spec):
    """Candidates -> model input. Categories are FIXED from the spec, so a value
    the model never saw becomes missing rather than silently shifting codes."""
    X = df[spec["features"]].copy()
    for col, levels in spec["categories"].items():
        X[col] = pd.Categorical(X[col].astype(str), categories=levels)
    return X


def course_folds(*frames):
    """course_id -> 'train' | 'valid' | 'test', shared by both axes."""
    courses = np.array(sorted(set().union(*(f["m_course_id"].unique() for f in frames))))
    rng = np.random.default_rng(SEED)
    rng.shuffle(courses)
    n_test = round(len(courses) * TEST_SHARE)
    n_valid = round(len(courses) * VALID_SHARE)
    folds = {c: "test" for c in courses[:n_test]}
    folds.update({c: "valid" for c in courses[n_test:n_test + n_valid]})
    folds.update({c: "train" for c in courses[n_test + n_valid:]})
    return folds


# ── Metrics ──────────────────────────────────────────────────────────────────

def _dcg(labels):
    return float(((2.0 ** labels - 1) / np.log2(np.arange(2, len(labels) + 2))).sum())


def select_trees(model, X_valid, valid, label_col):
    """Checkpoint with the best validation (NDCG@3, full NDCG); on an exact tie,
    the LARGER one.

    Tie goes to the larger model because a flat validation curve carries no
    information here, and the smallest-first rule failed on the formula labels:
    the validation courses happen to be easy enough that ONE tree already scores
    a perfect 1.0, so it kept one tree -- which then scored 0.969 NDCG@1 on
    test, where 50+ trees score 1.0. With a 0.05 learning rate, trees added
    along a flat curve do not change the validation ranking at all.
    """
    best_n, best_key, curve = None, None, {}
    for n in CHECKPOINTS:
        scores = model.predict(X_valid, iteration_range=(0, n))
        m = rank_metrics(valid, scores, label_col, np.random.default_rng(SEED),
                         draws=SELECTION_TIE_DRAWS)
        key = (round(m["ndcg@3"], 6), round(m["ndcg"], 6))
        curve[n] = {"ndcg@3": m["ndcg@3"], "ndcg": m["ndcg"]}
        if best_key is None or key >= best_key:
            best_n, best_key = n, key
    return best_n, curve


def rank_metrics(df, scores, label_col, rng, draws=TIE_DRAWS):
    """Per-group ranking quality, ties broken randomly and averaged.

    Only INFORMATIVE groups count (at least two distinct labels): a group where
    every candidate is equally good cannot be ranked well or badly, and
    including it would inflate every scorer toward 1.0 alike.
    """
    labels_all = df[label_col].to_numpy()
    out = {k: [] for k in ("ndcg@1", "ndcg@3", "ndcg@10", "ndcg", "best_at_1", "best_in_top3",
                           "best_in_top5", "rank_of_first_best")}
    for idx in df.groupby("qid", sort=False).indices.values():
        labels = labels_all[idx]
        if labels.min() == labels.max():
            continue
        s = scores[idx]
        ideal = np.sort(labels)[::-1]
        per_draw = {k: [] for k in out}
        for _ in range(draws):
            order = np.lexsort((rng.random(len(s)), -s))
            ranked = labels[order]
            for k in (1, 3, 10):
                per_draw[f"ndcg@{k}"].append(_dcg(ranked[:k]) / _dcg(ideal[:k]))
            per_draw["ndcg"].append(_dcg(ranked) / _dcg(ideal))
            first_best = int(np.argmax(ranked == labels.max())) + 1
            per_draw["rank_of_first_best"].append(first_best)
            per_draw["best_at_1"].append(first_best <= 1)
            per_draw["best_in_top3"].append(first_best <= 3)
            per_draw["best_in_top5"].append(first_best <= 5)
        for k in out:
            out[k].append(float(np.mean(per_draw[k])))
    summary = {k: round(float(np.mean(v)), 4) for k, v in out.items()}
    summary["median_rank_of_first_best"] = float(np.median(out["rank_of_first_best"]))
    summary["groups_evaluated"] = len(out["ndcg@1"])
    return summary


# ── Agreement with the hand-tuned scorer ─────────────────────────────────────

def agreement(df, model_scores, axis, rng):
    """How often the model's #1 is one of the formula's top-scoring candidates,
    and -- where it is not -- what the model chose instead."""
    cfg = AXES[axis]
    formula = df[cfg["formula_score"]].to_numpy()
    rules = cfg["rules"]
    rows, tier_sizes = [], []
    for qid, idx in df.groupby("qid", sort=False).indices.items():
        s = model_scores[idx]
        top = idx[np.lexsort((rng.random(len(s)), -s))[0]]
        f = formula[idx]
        tier = idx[f == f.max()]
        tier_sizes.append(len(tier))
        # The formula's representative pick: its best-scoring candidate that
        # comes first in the scheduler's ranked list.
        formula_pick = tier[np.argmin(df["m_candidate_position"].to_numpy()[tier])]
        rows.append((qid, top, formula_pick, top in set(tier)))

    agree = [r for r in rows if r[3]]
    disagree = [r for r in rows if not r[3]]
    result = {
        "groups": len(rows),
        "model_top1_in_formula_top_tier": round(len(agree) / len(rows), 4),
        "formula_top_tier_size_median": float(np.median(tier_sizes)),
        "disagreements": len(disagree),
    }
    if not disagree:
        return result

    m = df.loc[[r[1] for r in disagree]].reset_index(drop=True)
    f = df.loc[[r[2] for r in disagree]].reset_index(drop=True)
    penalty_col = cfg["penalty"]
    demand_col = "f_room_perfect_demand" if axis == "room" else "f_slot_perfect_demand"
    diff = {
        "model_pick_breaks_rule_formula_pick_keeps": {
            rule: int(((m[col] > 0) & (f[col] == 0)).sum()) for rule, col in rules.items()
        },
        "model_pick_keeps_rule_formula_pick_breaks": {
            rule: int(((m[col] == 0) & (f[col] > 0)).sum()) for rule, col in rules.items()
        },
        "mean_penalty_model_pick": round(float(m[penalty_col].mean()), 3),
        "mean_penalty_formula_pick": round(float(f[penalty_col].mean()), 3),
        "same_penalty_share": round(float((m[penalty_col] == f[penalty_col]).mean()), 4),
        "mean_perfect_demand_model_pick": round(float(m[demand_col].mean()), 1),
        "mean_perfect_demand_formula_pick": round(float(f[demand_col].mean()), 1),
        "course_types": m["f_course_type"].value_counts().head(6).to_dict(),
    }
    # What each label source says about the two picks.
    for col, name in (("m_hybrid_count", "hybrid"), ("m_manual_choice", "manual")):
        if col in df.columns:
            diff[f"model_pick_preferred_by_{name}"] = round(float((m[col] > f[col]).mean()), 4)
            diff[f"formula_pick_preferred_by_{name}"] = round(float((m[col] < f[col]).mean()), 4)
    if axis == "room":
        diff["model_pick_same_building_share"] = None  # filled below
    if axis == "room":
        diff.pop("model_pick_same_building_share")
        diff["mean_capacity_slack_model_pick"] = round(float(m["f_capacity_slack"].mean()), 1)
        diff["mean_capacity_slack_formula_pick"] = round(float(f["f_capacity_slack"].mean()), 1)
    else:
        diff["mean_start_hour_model_pick"] = round(float(m["f_start_minute"].mean() / 60), 2)
        diff["mean_start_hour_formula_pick"] = round(float(f["f_start_minute"].mean() / 60), 2)
    result["when_they_disagree"] = diff
    result["test_course_types"] = df.drop_duplicates("qid")["f_course_type"] \
        .value_counts().head(6).to_dict()
    return result


# ── Importances ──────────────────────────────────────────────────────────────

def importances(model, X, df, axis, spec):
    cfg = AXES[axis]
    booster = model.get_booster()
    gain = booster.get_score(importance_type="total_gain")
    total = sum(gain.values()) or 1.0
    gain_share = {k: round(v / total, 4) for k, v in
                  sorted(gain.items(), key=lambda kv: -kv[1])}

    contribs = booster.predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    col = {name: i for i, name in enumerate(spec["features"])}

    # Share of the model's total |SHAP| per feature group. Everything not in a
    # named group is section context (course type, level, enrolment...), which
    # is constant inside a group and can only act through interactions.
    abs_shap = np.abs(contribs[:, :-1])
    total_abs = abs_shap.sum()
    named = set()
    group_share = {}
    for name, feats in cfg["feature_groups"].items():
        feats = [f for f in feats if f in col]
        named.update(feats)
        group_share[name] = round(float(abs_shap[:, [col[f] for f in feats]].sum() / total_abs), 4)
    rest = [col[f] for f in spec["features"] if f not in named]
    group_share["section context"] = round(float(abs_shap[:, rest].sum() / total_abs), 4)

    return {"gain_share_top": dict(list(gain_share.items())[:12]),
            "shap_share_by_group": group_share}


def _demean_within(values, groups):
    return values - pd.DataFrame(values).groupby(groups).transform("mean").to_numpy()


def implied_weights(df, scores, axis):
    """What the model charges for each rule, read WITHIN one section's list.

    A ranker's score only means something relative to the other candidates of
    the same section, so scores and terms are both centred per group, then the
    score is regressed on the rule indicators plus the terms the hand formula
    does not have (demand, spare seats...). -coefficient = score points the
    model takes off for breaking the rule, other terms held fixed.

    This replaced a SHAP contrast pooled across all rows, which FAILED its own
    calibration: on the model trained on the formula's labels it read
    4 : 12.5 : 11.3 : 9.9 for room type : campus : capacity : department --
    pooling mixes sections, whose score levels are unrelated. The formula-label
    model is the check that this method recovers the hand ordering.

    Charges are also given on the hand scale: multiplied by one factor so the
    yes/no rule charges sum to the same total as their hand weights.
    """
    terms = AXES[axis]["implied"]
    values = [np.asarray(fn(df)) for _, fn, _ in terms]
    X = _demean_within(np.column_stack([v.astype(float) for v in values]), df["qid"].to_numpy())
    y = _demean_within(scores.reshape(-1, 1).astype(float), df["qid"].to_numpy()).ravel()

    # A term that never varies inside any list cannot be estimated.
    usable = X.std(axis=0) > 1e-12
    coef = np.full(len(terms), np.nan)
    coef[usable] = np.linalg.lstsq(X[:, usable], y, rcond=None)[0]

    def r2(cols):
        fit = np.linalg.lstsq(X[:, cols], y, rcond=None)[0]
        return float(1 - ((y - X[:, cols] @ fit) ** 2).sum() / (y ** 2).sum())

    rule_cols = [i for i, (_, _, w) in enumerate(terms) if w is not None and usable[i]]
    indicator = [i for i in rule_cols if values[i].dtype == bool]
    hand_total = sum(terms[i][2] for i in indicator)
    model_total = sum(-coef[i] for i in indicator)
    factor = hand_total / model_total if model_total > 0 else float("nan")

    return {
        "terms": {
            name: {
                "hand_weight": weight,
                "model_charge": None if np.isnan(coef[i]) else round(float(-coef[i]), 4),
                "on_hand_scale": None if np.isnan(coef[i]) else round(float(-coef[i] * factor), 2),
                "rows_nonzero": int((values[i] != 0).sum()),
            }
            for i, (name, _, weight) in enumerate(terms)
        },
        "within_section_r2_rules_only": round(r2(rule_cols), 4),
        "within_section_r2_all_terms": round(r2(list(np.flatnonzero(usable))), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global DATA_DIR, MODEL_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--labels", nargs="+", default=None, metavar="NAME=COLUMN",
                        help="label sources to train, e.g. manual=m_manual_choice "
                             "formula=label (default: %s)" % LABELS)
    args = parser.parse_args()
    DATA_DIR, MODEL_DIR = pathlib.Path(args.data_dir), pathlib.Path(args.model_dir)
    labels = dict(pair.split("=", 1) for pair in args.labels) if args.labels else LABELS
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    frames = {axis: load(axis) for axis in AXES}
    folds = course_folds(*frames.values())
    report = {"data_dir": str(DATA_DIR), "labels": labels,
              "objective": PARAMS["objective"], "xgboost": xgb.__version__,
              "split": "by course, 70/10/20", "seed": SEED, "tie_draws": TIE_DRAWS,
              "params": PARAMS, "tree_selection": "validation NDCG@3, random ties"}

    for axis, df in frames.items():
        spec = feature_spec(df)
        (MODEL_DIR / f"{axis}_feature_spec.json").write_text(json.dumps(spec, indent=2))
        fold = df["m_course_id"].map(folds)
        parts = {name: df[fold == name].reset_index(drop=True)
                 for name in ("train", "valid", "test")}
        X = {name: encode(part, spec) for name, part in parts.items()}
        report[axis] = {"split": {name: {"groups": int(part["qid"].nunique()),
                                         "rows": len(part),
                                         "courses": int(part["m_course_id"].nunique())}
                                  for name, part in parts.items()}}
        test = parts["test"]

        for source, label_col in labels.items():
            full = xgb.XGBRanker(**PARAMS)
            full.fit(X["train"], parts["train"][label_col], qid=parts["train"]["qid"])
            n_trees, curve = select_trees(full, X["valid"], parts["valid"], label_col)

            # Keep only the selected trees, so the saved model IS the evaluated one.
            model = xgb.XGBRanker()
            model._Booster = full.get_booster()[:n_trees]
            path = MODEL_DIR / f"{axis}_{source}_ranker.json"
            model.save_model(path)
            model = xgb.XGBRanker()
            model.load_model(path)

            model_scores = model.predict(X["test"])
            rng = np.random.default_rng(SEED)
            # Two extra baselines separate WHY the model beats score_candidate:
            #   penalty only -- score_candidate without its snug-room bonus, which
            #                   hybrid never uses (it ranks by penalty alone);
            #   list order   -- the scheduler's ranked list as-is. If the model
            #                   only matched this, it would have learned the
            #                   database-order tie artifact, not a preference.
            baselines = {
                "model": model_scores,
                "hand_formula (grasp.score_candidate)": test[AXES[axis]["formula_score"]].to_numpy(),
                "penalty_only": -test[AXES[axis]["penalty"]].to_numpy(),
                "list_order": -test["m_candidate_position"].to_numpy().astype(float),
                "random": np.zeros(len(test)),
            }
            report[axis][source] = {
                "label_column": label_col,
                "trees": n_trees,
                "validation_ndcg@3_by_trees": curve,
                "test_metrics": {name: rank_metrics(test, s, label_col, rng)
                                 for name, s in baselines.items()},
                "agreement_with_formula": agreement(test, model_scores, axis, rng),
                "importances": importances(model, X["test"], test, axis, spec),
                "implied_weights": implied_weights(test, model_scores, axis),
            }
            m = report[axis][source]["test_metrics"]
            print(f"[{axis}/{source}] trees={n_trees} "
                  f"ndcg@1 model={m['model']['ndcg@1']} "
                  f"formula={m['hand_formula (grasp.score_candidate)']['ndcg@1']} "
                  f"random={m['random']['ndcg@1']}", flush=True)

    (MODEL_DIR / "training_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"report -> {MODEL_DIR / 'training_report.json'}")


if __name__ == "__main__":
    main()
