# Phase 1 backup — original constraint model

Frozen copies of the constraint model **as it was before the Phase 2 hard/soft
re-architecture began (2026-08-09)**. Kept so we can revert or diff against the
"everything-is-hard" version any time.

- `constraints.ORIGINAL.py` — copy of `Optimization/constraints.py`
- `evaluation.ORIGINAL.py`  — copy of `Optimization/evaluation.py`

These are **not imported anywhere**. To restore, copy one back up a level,
overwriting the working file. See `CONSTRAINT_INVESTIGATION.md` for why the
model is changing.
