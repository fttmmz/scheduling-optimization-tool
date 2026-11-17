"""The RL scheduler's model loading, which has to work without xgboost.

rl_scheduler evaluates the trained Q model from xgboost's JSON file with numpy
(TreeEnsemble), so the server never imports xgboost. If that evaluator drifted
from xgboost's own predictions, the scheduler would silently follow a different
policy from the one that was trained and benchmarked -- hence a test.
"""
import numpy as np
import pytest

from backend.Optimization.Algorithims import rl_scheduler as RL


def test_tree_ensemble_matches_xgboost(tmp_path):
    xgboost = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 7)).astype(np.float32)
    y = X[:, 0] * 2 - X[:, 1] ** 2 + (X[:, 2] > 0.3) + rng.normal(scale=0.1, size=2000)
    X[rng.random(X.shape) < 0.05] = np.nan   # missing values take default_left
    model = xgboost.XGBRegressor(n_estimators=60, learning_rate=0.1, max_depth=5,
                                 subsample=0.8, random_state=0)
    model.fit(X, y)
    path = tmp_path / "model.json"
    model.get_booster().save_model(str(path))

    ours = RL.TreeEnsemble(path).predict(X[:500])
    theirs = model.get_booster().inplace_predict(X[:500])
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-4)


def test_missing_model_falls_back_to_the_static_ranker(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_MODEL_DIR", str(tmp_path))
    loaded = RL.load_policy("fqi_xgb")
    assert loaded.choose is RL.static_policy
    assert "fallback" in loaded.description


def _q_tools():
    """load_q_model and epsilon_q_policy are experiment helpers in ml/rl/q_tools.py, not rl_scheduler."""
    _ml_rl_scripts()
    import q_tools
    return q_tools


def test_load_q_model_has_no_fallback(tmp_path, monkeypatch):
    """Data collection must not log the static ranker's episodes under a
    behaviour model's name because the model file was missing."""
    monkeypatch.setenv("RL_MODEL_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        _q_tools().load_q_model("rl_q")


class _Prefers:
    """A stand-in Q model: anything with predict(X) can drive q_policy."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, X):
        return self.scores(len(X))


def _one_hot(row):
    return _Prefers(lambda n: np.eye(n)[min(row, n - 1)])


def test_epsilon_q_policy_follows_the_model_at_epsilon_zero():
    rng = np.random.default_rng(0)
    X = np.zeros((15, 4), dtype=np.float32)
    policy = _q_tools().epsilon_q_policy(_one_hot(7), 0.0)
    assert {policy(X, rng) for _ in range(50)} == {7}


def test_epsilon_q_policy_explores_only_inside_the_candidates():
    rng = np.random.default_rng(0)
    X = np.zeros((6, 4), dtype=np.float32)
    policy = _q_tools().epsilon_q_policy(_one_hot(0), 1.0)
    assert {policy(X, rng) for _ in range(300)} == set(range(6))


def test_episode_driven_by_a_q_model_has_no_hard_conflicts():
    """The hard rules are a mask: whatever the Q model prefers, it only ever
    chooses among allowed placements. Here it prefers the static ranker's LAST
    row, so the run takes options ml_greedy would not."""
    from backend.Optimization.constraints import finalize_schedule
    from backend.Optimization.evaluation import count_hard_conflicts
    from test_algorithms import make_rooms, make_sections, make_timeslots

    sections, timeslots, rooms = make_sections(), make_timeslots(), make_rooms()
    env = RL.SchedulingEnv(sections, timeslots, rooms, mode="noxgb")
    last = _Prefers(lambda n: np.arange(n, dtype=np.float64))
    schedule = env.run(_q_tools().epsilon_q_policy(last, 0.3), seed=1)
    schedule = finalize_schedule(schedule, sections, timeslots, env.cache, rooms=rooms)
    assert count_hard_conflicts(schedule, rooms) == 0


def _ml_rl_scripts():
    """Make ml/rl/ importable (its scripts import each other as `common`)."""
    import pathlib
    import sys
    path = str(pathlib.Path(__file__).resolve().parents[1] / "ml" / "rl")
    if path not in sys.path:
        sys.path.insert(0, path)


def test_iterate_refuses_training_seeds_that_are_validation_or_test_seeds():
    _ml_rl_scripts()
    import iterate
    with pytest.raises(SystemExit):
        iterate.check_seeds(1801, 4)          # round 3 would be 2001-2100
    with pytest.raises(SystemExit):
        iterate.check_seeds(1, 1)             # the benchmark seeds
    iterate.check_seeds(3001, 4)              # clear of both


def test_pooled_training_data_must_be_distinct_and_alike():
    pytest.importorskip("xgboost")
    _ml_rl_scripts()
    import train_q

    def episode(seed, k=15, f=18):
        return {"seed": seed, "X": np.zeros((3, k, f), dtype=np.float32)}

    train_q.check_pool([episode(1), episode(2)])
    with pytest.raises(SystemExit):
        train_q.check_pool([episode(1), episode(1)])
    with pytest.raises(SystemExit):
        train_q.check_pool([episode(1), episode(2, k=5)])


def test_double_target_values_ones_choice_by_the_other():
    """V(t) = Q_B(t, argmax_a Q_A(t, a)); a step with no action passes its
    reward through: V(t) = r_t + gamma V(t+1)."""
    pytest.importorskip("xgboost")
    _ml_rl_scripts()
    import train_q

    q_a = np.array([[1.0, 5.0], [-np.inf, -np.inf], [3.0, 2.0]])
    q_b = np.array([[10.0, -4.0], [-np.inf, -np.inf], [-1.0, 7.0]])
    ep = {"mask": np.isfinite(q_a), "r": np.array([0.0, -2.0, 0.0])}
    v = train_q.state_values(q_a, ep, 0.5, q_eval=q_b)
    # step 0: A prefers row 1, B values it -4; step 2: A prefers row 0, B says -1;
    # step 1 has no action: -2 + 0.5 * V(2).
    np.testing.assert_allclose(v, [-4.0, -2.5, -1.0])
    # Without q_eval it is the plain max, as before.
    np.testing.assert_allclose(train_q.state_values(q_a, ep, 0.5), [5.0, -0.5, 3.0])


def test_related_credit_charges_only_cheaper_displacement():
    """Decision 0 takes cell (room 10, hour 1). Decision 1 cannot use room 10:
    not charged. Decision 2 could have had (10, 1) for 0 but paid 3: charged 3.
    Decision 3 could use (10, 1) but it costs what it paid anyway: not charged."""
    pytest.importorskip("xgboost")
    _ml_rl_scripts()
    import train_q

    cells = [
        ({10: 0.5}, {1: 0.5}),
        ({11: 0.0}, {1: 0.0}),
        ({10: 0.0, 12: 0.0}, {1: 0.0, 2: 3.0}),
        ({10: 0.0}, {1: 0.5, 3: 0.5}),
    ]
    ep = {
        "step": np.array([0, 1, 2, 3]),
        "a": np.array([0, 0, 0, 0]),
        "r": np.array([-1.0, 0.0, -3.0, -0.5], dtype=np.float32),
        "cand_room": np.array([[10], [11], [12], [10]]),
        "cand_ts": np.array([[1], [1], [2], [3]]),
    }
    y, charged = train_q.related_credit_targets(ep, cells)
    np.testing.assert_allclose(y, [-4.0, 0.0, -3.0, -0.5])
    assert charged == 1


def test_unscheduled_reward_is_the_fitness_price():
    """One unscheduled section, in soft-penalty units, as calculate_fitness
    weighs it: UNSCHEDULED_WEIGHT / (SOFT_PENALTY_WEIGHT / SOFT_PENALTY_REFERENCE)."""
    from backend.Optimization.evaluation import (
        SOFT_PENALTY_REFERENCE, SOFT_PENALTY_WEIGHT, UNSCHEDULED_WEIGHT,
    )
    assert RL.UNSCHEDULED_REWARD == pytest.approx(
        -UNSCHEDULED_WEIGHT * SOFT_PENALTY_REFERENCE / SOFT_PENALTY_WEIGHT
    )
