"""The shared seed list for the benchmark protocol.

Every algorithm's `*_runs()` benchmark uses the SAME list, in the same order, so run *i* of
one algorithm is reproducible by anyone and no author can pick a seed set that flatters their
own algorithm. See ALGORITHM_GUIDE.md section 9.

Do not edit SEEDS. Changing it silently invalidates every number already recorded by every
member of the team. If 30 runs is not enough, extend the list at the END and say so in the
paper's methodology.

What the shared list does and does not buy you:

  * It DOES make each run reproducible, and it removes seed selection as a free parameter.
  * It does NOT give the algorithms identical random draws. They consume randomness in
    different amounts and different orders, so seed 7 does not mean "the same conditions"
    for a GA and a PSO. Do not claim common-random-numbers variance reduction, and do not
    pair runs by index in a statistical test -- use an unpaired rank test (Mann-Whitney U)
    for a single instance.
"""

import random

try:
    import numpy as _np
except ImportError:              # numpy is only needed by the algorithms that use it
    _np = None


# 1..30. The specific values do not matter -- what matters is that they were fixed before
# anyone looked at a result, and that all five (and the five Phase 3) algorithms share them.
SEEDS = tuple(range(1, 31))


def seed_all(seed):
    """Seed every global RNG this project's algorithms draw from, and return the seed.

    Call this ONCE at the top of each benchmark run, before any randomness is consumed.

    Modules that build their own generator (`np.random.default_rng(...)`, `random.Random(...)`)
    are NOT covered by this -- a local generator ignores the global seed. Thread the returned
    value into those explicitly, as pso.py does with `pso_schedule(..., seed=seed)`.
    """
    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed % (2 ** 32))
    return seed


def seed_for_run(run_index, seeds=None):
    """The seed for run `run_index` (0-based), from `seeds` or the shared list.

    Wraps around if more runs are requested than there are seeds, so a 50-run exploratory
    sweep still works -- but a reported benchmark should use exactly len(SEEDS) runs.
    """
    pool = tuple(seeds) if seeds else SEEDS
    if not pool:
        return None
    return pool[run_index % len(pool)]
