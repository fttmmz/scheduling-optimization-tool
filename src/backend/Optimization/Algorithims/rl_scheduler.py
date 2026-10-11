"""Reinforcement-learning scheduler: a learned Q-function builds the timetable.

Fitted Q iteration (batch Q-learning, Ernst, Geurts & Wehenkel 2005) with an
XGBoost regressor. The design is in q_handover.md section 2; in short:

  Episode   one construction pass over GRASP's fixed difficulty order
            (order_sections_by_difficulty), hardest section first.
  Decision  a (room, timeslot) for each section that needs BOTH. Time-only,
            room-only and supervision sections keep GRASP's rules.
  Actions   viable rooms x valid timeslots that pass the hard rules against
            the current occupancy -- the hard rules are a MASK, never learned
            -- pruned to the TOP_K best by a static scorer:
              mode "xgb"    V3's score (the trace XGBoost room model + time terms)
              mode "noxgb"  grasp.score_candidate (the formula), the ablation
  Reward    minus the soft cost of the placement, in penalty units, exactly as
            GRASP's _item_soft_penalty charges it; UNSCHEDULED_REWARD when a
            section has no allowed action at all.
  Policy    argmax over the TOP_K of Q(features), Q trained offline by
            ml/rl/train_q.py on episodes logged by ml/rl/collect_episodes.py.

The features mix the candidate's own static cost (soft penalties, the XGBoost
room score, capacity slack) with the STATE: how contested its room and hour
still are by the sections not yet placed. That state part is the only thing a
Q-function can add over the static ranker it prunes with, so the static ranker
on its own (model="static") is the baseline every Q model must beat.

The same class runs data collection and inference, so the features the model
was trained on and the ones it is queried with come from one code path.

Running it needs two files, neither of them xgboost:
  * the Q model: xgboost's JSON model file, evaluated here with numpy
    (TreeEnsemble), so xgboost is needed only to TRAIN it -- found in
    $RL_MODEL_DIR, else ml/models/rl/;
  * in mode xgb, the room score table that ml_greedy reads -- found in
    $ML_DATA_DIR, else ml/data/.
If either is missing (synthetic test data, a fresh server) the policy falls
back to the static ranker's top choice, i.e. ml_greedy, and says so.

Method and references (full list, parameter choices and their basis:
ml/RL_SCHEDULER_METHODS.md):
  Ernst, Geurts & Wehenkel (2005), Tree-based batch mode reinforcement
    learning, JMLR 6: fitted Q iteration with tree regressors -- the training
    loop in ml/rl/train_q.py.
  Watkins & Dayan (1992), Q-learning, Machine Learning 8: the Q-function and
    its Bellman target.
  Khalil, Dai, Zhang, Dilkina & Song (2017), Learning combinatorial
    optimization algorithms over graphs, NeurIPS: a greedy construction that
    adds one element at a time by argmax of a learned Q, trained by fitted
    Q-learning -- the same shape as this scheduler.
  Chen & Guestrin (2016), XGBoost, KDD: the regressor.
No RL library code is used; everything here and in ml/rl/ is written for this
project on top of the repo's own constraint, ordering and evaluation code.
"""
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from backend.models.models import ScheduleItem
from backend.Optimization.constraints import (
    finalize_schedule,
    instructor_occupancy_ids,
    room_soft_penalty,
    seats_needed,
    time_soft_penalty,
)
from backend.Optimization.Algorithims import grasp, ml_greedy
from backend.Optimization.benchmark_seeds import seed_all, seed_for_run
from backend.Optimization.evaluation import (
    SOFT_PENALTY_REFERENCE,
    SOFT_PENALTY_WEIGHT,
    UNSCHEDULED_WEIGHT,
    build_timeslot_guideline_cache,
    calculate_fitness,
)

# Candidates the Q-function chooses among, per decision. A trained model
# records the K it was trained with in its spec, and that K is used with it.
TOP_K = 15
# What calculate_fitness charges for one unscheduled section, in soft-penalty
# units: UNSCHEDULED_WEIGHT / (SOFT_PENALTY_WEIGHT / SOFT_PENALTY_REFERENCE)
# = 2.0 / 0.05 = 40. Deriving it keeps the reward on the fitness function's
# own scale. (No logged episode ever had an unscheduled section, so the value
# has not influenced any trained model.)
UNSCHEDULED_REWARD = -UNSCHEDULED_WEIGHT / (SOFT_PENALTY_WEIGHT / SOFT_PENALTY_REFERENCE)
TIMESLOT_GUIDELINE_PENALTY = grasp.TIMESLOT_GUIDELINE_PENALTY
SOFT_SCORE_SCALE = grasp.SOFT_SCORE_SCALE
# The room score table of mode xgb: the model trained on hybrid's schedules,
# as in GRASP V3 and ml_greedy's default.
ROOM_SCORE_SOURCE = "hybrid"

# Model file names in the model directory. Each <name>.json (xgboost's JSON
# model format) has a sidecar <name>.spec.json with its mode, features and K.
# "rl_q" is the production model: a copy of return_xgb_d3 (Monte-Carlo return
# regression, depth-3 trees), chosen by the sensitivity sweep and confirmed on
# two independent training sets -- see ml/RL_SCHEDULER_METHODS.md section 6.
DEFAULT_MODEL = os.environ.get("RL_MODEL", "rl_q")
STATIC_MODELS = ("static", "none")

# Column order of the feature matrix. The "xgb" mode prepends "xgb_score".
BASE_FEATURES = [
    "prune_score",          # the static scorer's value (V3 or formula)
    "prune_rank",           # 0 = the static scorer's favourite
    "room_soft",            # room_soft_penalty of this room
    "time_soft",            # time_soft_penalty of this hour
    "guideline_fit",        # 1 if the hour fits the course-type guideline
    "capacity_slack",       # room capacity - seats needed, clipped
    "seats",                # seats needed
    "progress",             # decision index / number of decisions
    "log_allowed",          # log(1 + allowed actions), a state feature
    "rooms_free_at_ts",     # other viable rooms of this section free at this hour
    "dept_rooms_free_at_ts",  # ... of them owned by the course's department
    "room_free_hours",      # hours this room has left (share of all timeslots)
    "room_demand",          # unplaced decisions still listing this room
    "room_pressure",        # room_demand / free hours of the room
    "ts_demand",            # unplaced decisions still listing this hour
    "instructor_ts_demand",  # unplaced classes of the same teachers wanting this hour
    "own_excess",           # this candidate's cost above the section's own minimum
]


def feature_names(mode):
    return (["xgb_score"] if mode == "xgb" else []) + BASE_FEATURES


# ============================================================
# Static scorer (the pruning) = ml_greedy's score: a room part plus a time
# part, each computed once per section instead of once per pair.
#   mode "xgb"    the hybrid-trained room model's score (GRASP V3's)
#   mode "noxgb"  GRASP's formula room score, the ablation
# ============================================================

class _Decision:
    """Everything static about one room+time section, precomputed once."""

    __slots__ = (
        "index", "section", "rooms", "room_ids", "timeslots", "ts_ids", "instructors",
        "room_part", "ts_part", "room_soft", "time_soft", "guideline",
        "capacity_slack", "seats", "xgb", "same_dept", "ts_cost", "min_cost",
    )


# ============================================================
# Environment
# ============================================================
class SchedulingEnv:
    """One construction pass = one episode. Static data is built once here and
    reused by every episode; run() keeps all state local."""

    def __init__(self, sections, timeslots, rooms, mode="xgb",
                 valid_timeslot_cache=None, section_candidates=None, top_k=TOP_K):
        if mode not in ("xgb", "noxgb"):
            raise ValueError(f"Unknown RL mode {mode!r}")
        self.room_scores = (ml_greedy.load_room_scores(ROOM_SCORE_SOURCE)
                            if mode == "xgb" else {})
        self.mode = mode
        self.top_k = top_k
        self.features = feature_names(mode)
        self.sections, self.timeslots, self.rooms = sections, timeslots, rooms
        self.cache = (valid_timeslot_cache if valid_timeslot_cache is not None
                      else build_timeslot_guideline_cache(sections, timeslots))
        self.order = (section_candidates if section_candidates is not None
                      else grasp.order_sections_by_difficulty(
                          grasp.build_section_candidates(sections, rooms, timeslots)))

        # One _Decision per room+time section, None for the rest.
        self.decisions = []
        self.decision_of = []
        for section, needs, viable_rooms, valid_ts in self.order:
            if needs.room and needs.time:
                dec = self._build_decision(len(self.decisions), section, viable_rooms, valid_ts)
                self.decisions.append(dec)
                self.decision_of.append(dec)
            else:
                self.decision_of.append(None)

        # Per decision: the contention-free minimum cost (see _build_decision).
        # ml/rl/train_q.py --shaped adds it back to the reward as a baseline.
        self.min_costs = np.array([dec.min_cost for dec in self.decisions])

        # Starting demand: every decision still unplaced. run() copies these.
        self.room_demand0 = Counter()
        self.ts_demand0 = Counter()
        for dec in self.decisions:
            self.room_demand0.update(dec.room_ids)
            self.ts_demand0.update(dec.ts_ids)
        # Teachers' wanted hours count time-only classes too: they hold the
        # teacher's hour just the same.
        self.instructor_ts_demand0 = Counter()
        for section, needs, _rooms, valid_ts in self.order:
            if needs.time:
                for instructor_id in instructor_occupancy_ids(section):
                    self.instructor_ts_demand0.update((instructor_id, ts.id) for ts in valid_ts)

    # ---------------------------------------------------------------
    def room_score(self, section, room):
        """Static room part: the model's score in mode xgb, the formula otherwise
        (and for any section the score table does not cover)."""
        return ml_greedy.room_score(section, room, self.room_scores)

    def room_model_score(self, section, room):
        """The room model's 0..1 score, the xgb_score feature; 0 if not covered."""
        return self.room_scores.get(ml_greedy.section_key(section), {}).get(str(room.id), 0.0)

    def _build_decision(self, index, section, viable_rooms, valid_ts):
        guideline_ids = self.cache.get(
            (getattr(section.course, "id", None), str(getattr(section, "no", None))), set()
        )
        seats = seats_needed(section) or 0
        dec = _Decision()
        dec.index = index
        dec.section = section
        dec.rooms = list(viable_rooms)
        dec.room_ids = [r.id for r in viable_rooms]
        dec.timeslots = list(valid_ts)
        dec.ts_ids = [t.id for t in valid_ts]
        dec.instructors = tuple(instructor_occupancy_ids(section))
        dec.room_part = np.array([self.room_score(section, r) for r in viable_rooms], dtype=np.float64)
        dec.ts_part = np.array([ml_greedy.time_score(section, t, guideline_ids) for t in valid_ts],
                               dtype=np.float64)
        dec.room_soft = np.array([room_soft_penalty(section, r) for r in viable_rooms], dtype=np.float64)
        dec.time_soft = np.array([time_soft_penalty(section, t) for t in valid_ts], dtype=np.float64)
        dec.guideline = np.array([1.0 if t.id in guideline_ids else 0.0 for t in valid_ts])
        dec.capacity_slack = np.clip(
            np.array([(r.capacity or 0) - seats for r in viable_rooms], dtype=np.float64), -100, 100
        )
        dec.seats = float(seats)
        dec.xgb = np.array([self.room_model_score(section, r) for r in viable_rooms],
                           dtype=np.float64)
        dec.same_dept = np.array(
            [bool(r.dept_id) and r.dept_id == section.course.dept for r in viable_rooms], dtype=bool
        )
        # The reward charged for an hour, and the cheapest placement this
        # section could have on an EMPTY timetable. Anything it pays above that
        # minimum is caused by contention or by the scorer's preference.
        dec.ts_cost = dec.time_soft + (1.0 - dec.guideline) * TIMESLOT_GUIDELINE_PENALTY
        dec.min_cost = (float(dec.room_soft.min() + dec.ts_cost.min())
                        if len(viable_rooms) and len(valid_ts) else 0.0)
        return dec

    # ---------------------------------------------------------------
    def run(self, policy, seed=0, log=None):
        """Build one schedule. Returns the schedule before finalize_schedule.

        policy(X, rng) -> index into the TOP_K rows of X (row 0 is the static
        scorer's favourite). Ties in the static score are broken by a generator
        seeded from `seed`; the policy gets a SEPARATE generator, so replaying
        a logged episode with the same seed sees the same candidates whatever
        the policy draws.
        """
        tie_rng = np.random.default_rng([seed, 0])
        policy_rng = np.random.default_rng([seed, 1])

        occupied_rooms = set()
        occupied_instructors = set()
        room_used = Counter()
        room_demand = Counter(self.room_demand0)
        ts_demand = Counter(self.ts_demand0)
        instructor_ts_demand = Counter(self.instructor_ts_demand0)
        n_decisions = max(1, len(self.decisions))
        n_timeslots = max(1, len(self.timeslots))

        schedule = []
        for (section, needs, viable_rooms, valid_ts), dec in zip(self.order, self.decision_of):
            # This section no longer competes for anything: take it out of the
            # demand counts BEFORE scoring, so the counts mean "everyone else".
            if needs.time:
                for instructor_id in instructor_occupancy_ids(section):
                    instructor_ts_demand.subtract((instructor_id, ts.id) for ts in valid_ts)
            if dec is not None:
                room_demand.subtract(dec.room_ids)
                ts_demand.subtract(dec.ts_ids)

            if dec is not None:
                room, timeslot = self._decide(
                    dec, policy, tie_rng, policy_rng, log,
                    occupied_rooms, occupied_instructors, room_used,
                    room_demand, ts_demand, instructor_ts_demand,
                    n_decisions, n_timeslots,
                )
            elif not needs.room and not needs.time:
                room = timeslot = None
            elif not needs.room:
                # Time only, as GRASP: the first slot that leaves EVERY
                # instructor free; unscheduled rather than a forced clash.
                room = None
                section_instructors = instructor_occupancy_ids(section)
                timeslot = next(
                    (ts for ts in valid_ts
                     if all((i, ts.id) not in occupied_instructors for i in section_instructors)),
                    None,
                )
                if timeslot is None and valid_ts and not section_instructors:
                    timeslot = valid_ts[0]
            else:
                # Room only: the best room by the static room score. max() keeps
                # the first on a tie, i.e. get_viable_rooms' own order.
                room = (max(viable_rooms, key=lambda r: self.room_score(section, r))
                        if viable_rooms else None)
                timeslot = None

            schedule.append(ScheduleItem.from_section(
                section,
                room_id=room.id if room else None,
                timeslot_id=timeslot.id if timeslot else None,
            ))
            # A roomless class still holds its teachers' hour; supervision
            # holds nothing (instructor_occupancy_ids returns ()).
            if timeslot is not None:
                if room is not None:
                    occupied_rooms.add((room.id, timeslot.id))
                    room_used[room.id] += 1
                for instructor_id in instructor_occupancy_ids(section):
                    occupied_instructors.add((instructor_id, timeslot.id))

        return schedule

    def _decide(self, dec, policy, tie_rng, policy_rng, log,
                occupied_rooms, occupied_instructors, room_used,
                room_demand, ts_demand, instructor_ts_demand,
                n_decisions, n_timeslots):
        # Mask: the two hard rules, against the occupancy so far. Same test as
        # passes_hard_constraints with occupancy maps, unrolled so a timeslot
        # whose teacher is busy is skipped once rather than once per room.
        free = np.zeros((len(dec.room_ids), len(dec.ts_ids)), dtype=bool)
        for j, ts_id in enumerate(dec.ts_ids):
            if any((i, ts_id) in occupied_instructors for i in dec.instructors):
                continue
            for i, room_id in enumerate(dec.room_ids):
                if (room_id, ts_id) not in occupied_rooms:
                    free[i, j] = True
        ri, tj = np.nonzero(free)
        n_allowed = len(ri)

        if n_allowed == 0:
            if log is not None:
                log.add_unscheduled(dec.index)
            return None, None

        score = dec.room_part[ri] + dec.ts_part[tj]
        top = np.lexsort((tie_rng.random(n_allowed), -score))[:self.top_k]
        ri, tj, score = ri[top], tj[top], score[top]
        k = len(top)

        cand_room_ids = [dec.room_ids[i] for i in ri]
        cand_ts_ids = [dec.ts_ids[j] for j in tj]
        free_cols = free[:, tj]
        rooms_free = free_cols.sum(axis=0) - 1
        dept_free = (free_cols & dec.same_dept[:, None]).sum(axis=0) - dec.same_dept[ri]
        room_free_hours = np.array([n_timeslots - room_used[r] for r in cand_room_ids], dtype=np.float64)
        demand = np.array([room_demand[r] for r in cand_room_ids], dtype=np.float64)
        columns = [
            score,
            np.arange(k, dtype=np.float64),
            dec.room_soft[ri],
            dec.time_soft[tj],
            dec.guideline[tj],
            dec.capacity_slack[ri],
            np.full(k, dec.seats),
            np.full(k, dec.index / n_decisions),
            np.full(k, math.log1p(n_allowed)),
            rooms_free,
            dept_free,
            room_free_hours / n_timeslots,
            demand,
            demand / np.maximum(room_free_hours, 1.0),
            np.array([ts_demand[t] for t in cand_ts_ids], dtype=np.float64),
            np.array([sum(instructor_ts_demand[(i, t)] for i in dec.instructors)
                      for t in cand_ts_ids], dtype=np.float64),
            dec.room_soft[ri] + dec.ts_cost[tj] - dec.min_cost,
        ]
        if self.mode == "xgb":
            columns.insert(0, dec.xgb[ri])
        X = np.column_stack(columns).astype(np.float32)

        a = int(policy(X, policy_rng))
        reward = -(dec.room_soft[ri[a]] + dec.ts_cost[tj[a]])
        if log is not None:
            log.add(dec.index, X, a, reward, cand_room_ids, cand_ts_ids)
        return dec.rooms[ri[a]], dec.timeslots[tj[a]]


# ============================================================
# Episode log (data collection)
# ============================================================
class EpisodeLog:
    """Per decision: the TOP_K candidates' features, the chosen index, the
    reward and the candidates' ids. Steps with no allowed action are logged
    with an empty mask, a = -1 and UNSCHEDULED_REWARD."""

    def __init__(self, n_features, top_k=TOP_K):
        self.n_features = n_features
        self.top_k = top_k
        self.rows = []

    def add(self, index, X, a, reward, room_ids, ts_ids):
        self.rows.append((index, X, a, reward, room_ids, ts_ids))

    def add_unscheduled(self, index):
        self.rows.append((index, None, -1, UNSCHEDULED_REWARD, [], []))

    def arrays(self):
        n, k, f = len(self.rows), self.top_k, self.n_features
        X = np.zeros((n, k, f), dtype=np.float32)
        mask = np.zeros((n, k), dtype=bool)
        cand_room = np.full((n, k), -1, dtype=np.int64)
        cand_ts = np.full((n, k), -1, dtype=np.int64)
        a = np.full(n, -1, dtype=np.int16)
        r = np.zeros(n, dtype=np.float32)
        step = np.zeros(n, dtype=np.int32)
        for t, (index, x, chosen, reward, room_ids, ts_ids) in enumerate(self.rows):
            step[t], a[t], r[t] = index, chosen, reward
            if x is not None:
                m = len(x)
                X[t, :m] = x
                mask[t, :m] = True
                cand_room[t, :m] = room_ids
                cand_ts[t, :m] = ts_ids
        return {"X": X, "mask": mask, "a": a, "r": r, "step": step,
                "cand_room": cand_room, "cand_ts": cand_ts}


# ============================================================
# Policies
# ============================================================
def static_policy(X, rng):
    """The static ranker: its own favourite, row 0."""
    return 0


def epsilon_policy(epsilon):
    """Behaviour policy for data collection: the static favourite with
    probability 1 - epsilon, otherwise a uniformly random one of the TOP_K."""
    def policy(X, rng):
        if rng.random() < epsilon:
            return int(rng.integers(len(X)))
        return 0
    return policy


def q_policy(model):
    """Take the candidate with the highest predicted Q. np.argmax keeps the
    first on a tie, i.e. the static scorer's order."""
    def policy(X, rng):
        return int(np.argmax(model.predict(X)))
    return policy


# ============================================================
# Q model: xgboost's JSON model file, evaluated with numpy
# ============================================================
class TreeEnsemble:
    """A trained XGBoost regressor (gbtree, reg:squarederror), evaluated from
    its JSON model file without importing xgboost.

    The format is what xgboost 3.2.0 writes (its formal JSON schema file was
    removed in 3.2, so a test, not a schema, guards it): per tree, parallel arrays
    left_children / right_children (-1 at a leaf), split_indices,
    split_conditions (the threshold, or the leaf value at a leaf) and
    default_left (where a missing value goes). A row goes LEFT when
    x < threshold, as in xgboost. The prediction is base_score plus the sum of
    the leaf values. tests/test_rl_scheduler.py checks it against xgboost.

    All trees are walked at once: the node index of every (tree, row) pair is
    advanced one level per step, so one decision's K rows cost max_depth numpy
    steps rather than a Python loop over the trees.
    """

    def __init__(self, path):
        model = json.loads(Path(path).read_text(encoding="utf-8"))
        learner = model["learner"]
        booster = learner["gradient_booster"]
        if booster.get("name") != "gbtree":
            raise ValueError(f"{path}: only gbtree models are supported")
        objective = learner["objective"]["name"]
        if objective != "reg:squarederror":
            raise ValueError(f"{path}: objective {objective} is not supported")
        trees = booster["model"]["trees"]

        n_nodes = max(len(tree["left_children"]) for tree in trees)
        shape = (len(trees), n_nodes)
        self.left = np.full(shape, -1, dtype=np.int32)
        self.right = np.full(shape, -1, dtype=np.int32)
        self.feature = np.zeros(shape, dtype=np.int32)
        self.threshold = np.zeros(shape, dtype=np.float32)
        self.default_left = np.zeros(shape, dtype=bool)
        for t, tree in enumerate(trees):
            n = len(tree["left_children"])
            self.left[t, :n] = tree["left_children"]
            self.right[t, :n] = tree["right_children"]
            self.feature[t, :n] = tree["split_indices"]
            self.threshold[t, :n] = tree["split_conditions"]
            self.default_left[t, :n] = np.asarray(tree["default_left"], dtype=bool)
        # At a leaf, split_conditions holds the leaf value.
        self.value = np.where(self.left < 0, self.threshold, 0.0).astype(np.float64)
        self.max_depth = max(self._depth(t) for t in range(len(trees)))
        # base_score is stored as a string: "5E-1" or, in xgboost 3, "[5E-1]".
        base = learner["learner_model_param"]["base_score"]
        self.base_score = float(str(base).strip("[]").split(",")[0])
        self.n_features = int(learner["learner_model_param"]["num_feature"])

    def _depth(self, t):
        depth, frontier = 0, [0]
        while True:
            frontier = [c for n in frontier for c in (self.left[t, n], self.right[t, n]) if c >= 0]
            if not frontier:
                return depth
            depth += 1

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        trees = np.arange(self.left.shape[0])[:, None]
        rows = np.arange(len(X))[None, :]
        node = np.zeros((self.left.shape[0], len(X)), dtype=np.int32)
        for _ in range(self.max_depth):
            left = self.left[trees, node]
            x = X[rows, self.feature[trees, node]]
            go_left = np.where(np.isnan(x), self.default_left[trees, node],
                               x < self.threshold[trees, node])
            child = np.where(go_left, left, self.right[trees, node])
            node = np.where(left < 0, node, child)
        return self.base_score + self.value[trees, node].sum(axis=0)


def _model_dir():
    """$RL_MODEL_DIR if set, else ml/models/rl found by walking up from here."""
    if os.environ.get("RL_MODEL_DIR"):
        return Path(os.environ["RL_MODEL_DIR"])
    relative = Path("ml") / "models" / "rl"
    for parent in Path(__file__).resolve().parents:
        if (parent / relative).exists():
            return parent / relative
    return Path(__file__).resolve().parents[4] / relative


class LoadedPolicy:
    """What load_policy returns: the choice function and the settings the
    environment must be built with for it."""

    def __init__(self, choose, mode, top_k, description):
        self.choose, self.mode, self.top_k, self.description = choose, mode, top_k, description


def load_policy(model=None, mode=None):
    """The policy for a model name (a file in the model directory) or path.

    "static" gives the static ranker in `mode` (default xgb), i.e. ml_greedy.
    A missing model file, or in mode xgb a missing room score table, falls back
    to the static ranker with a warning: a Q model fed a zero xgb_score column
    would be queried far outside what it was trained on.
    """
    model = model or DEFAULT_MODEL
    if model in STATIC_MODELS:
        mode = mode or "xgb"
        return LoadedPolicy(static_policy, mode, TOP_K, f"static ranker ({mode})")

    path = Path(model)
    if not path.suffix:
        path = _model_dir() / f"{model}.json"
    spec_path = path.with_name(path.stem + ".spec.json")
    if not path.exists() or not spec_path.exists():
        print(f"Warning: RL model not found ({path}); using the static ranker")
        return LoadedPolicy(static_policy, mode or "xgb", TOP_K,
                            "static ranker (fallback: no model)")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["features"] != feature_names(spec["mode"]):
        raise ValueError(f"{spec_path}: feature list does not match this code")
    if spec["mode"] == "xgb" and not ml_greedy.load_room_scores(ROOM_SCORE_SOURCE):
        print("Warning: room score table missing; RL uses the static ranker")
        return LoadedPolicy(static_policy, "noxgb", TOP_K,
                            "static ranker (fallback: no room score table)")
    ensemble = TreeEnsemble(path)
    if ensemble.n_features != len(spec["features"]):
        raise ValueError(f"{path}: model has {ensemble.n_features} features, spec lists "
                         f"{len(spec['features'])}")
    return LoadedPolicy(q_policy(ensemble), spec["mode"], int(spec.get("top_k", TOP_K)),
                        f"Q model {path.name}")


# ============================================================
# Entry points
# ============================================================
def rl_schedule(sections, timeslots, rooms, seed=1, model=None, local_search=False,
                valid_timeslot_cache=None, env=None, policy=None):
    """One schedule from the Q policy, finalized. Returns (schedule, fitness)."""
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    if policy is None or env is None:
        loaded = load_policy(model)
        policy = loaded.choose
        env = SchedulingEnv(sections, timeslots, rooms, mode=loaded.mode, top_k=loaded.top_k,
                            valid_timeslot_cache=valid_timeslot_cache)

    schedule = env.run(policy, seed=seed)
    if local_search:
        # Optional polish, reported as its own variant: the standalone claim
        # is about the plain construction.
        schedule, _ = grasp.local_search(schedule, rooms, timeslots, sections,
                                         valid_timeslot_cache)
    schedule = finalize_schedule(schedule, sections, timeslots, valid_timeslot_cache,
                                 rooms=rooms)
    fitness = calculate_fitness(schedule, rooms, sections=sections, timeslots=timeslots,
                                valid_timeslot_cache=valid_timeslot_cache)
    return schedule, fitness


def rl_runs(sections, timeslots, rooms, num_runs=1, seeds=None, model=None,
            local_search=None):
    """Best of num_runs schedules. The policy is deterministic; runs differ only
    in how ties in the static score are broken (and in local search)."""
    if local_search is None:
        local_search = os.environ.get("RL_LOCAL_SEARCH") == "1"
    loaded = load_policy(model)
    print(f"\n=== RL scheduler: {num_runs} runs | {loaded.description} | mode {loaded.mode}"
          f"{' + local search' if local_search else ''} ===")

    valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)
    env = SchedulingEnv(sections, timeslots, rooms, mode=loaded.mode, top_k=loaded.top_k,
                        valid_timeslot_cache=valid_timeslot_cache)
    policy = loaded.choose

    best_schedule, best_fitness, scores = None, -1.0, []
    for run in range(num_runs):
        run_seed = seed_all(seed_for_run(run, seeds))
        schedule, fitness = rl_schedule(
            sections, timeslots, rooms, seed=run_seed or 0, local_search=local_search,
            valid_timeslot_cache=valid_timeslot_cache, env=env, policy=policy,
        )
        scores.append(fitness)
        print(f"Run {run + 1:2d} (seed {run_seed}): Fitness = {fitness:.4f}")
        if fitness > best_fitness:
            best_schedule, best_fitness = schedule, fitness

    if len(scores) > 1:
        print(f"Best {max(scores):.4f} | mean {statistics.mean(scores):.4f} | "
              f"std {statistics.stdev(scores):.4f}")
    return best_schedule
