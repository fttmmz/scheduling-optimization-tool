# Project Layout

- **Backend** (this repo, GA / hybrid scheduling optimization algorithm): `D:\uniDB\scheduling-optimization-tool`
  - Deployed via **Render**, under the **fttmz** account.
- **Frontend**: `D:\UCT frontend\v0-university-timetable-tool-2`
  - Deployed via **Vercel**, under the **logiphic** account.

The frontend calls this backend's `/api/optimize` endpoint (see the frontend's `NEXT_PUBLIC_API_URL` and `lib/api.ts`).

## Long-running jobs (GA / hybrid)

GA and hybrid runs can take 20+ minutes to over an hour (30 runs), which is why the optimize flow is split:
- `POST /api/optimize` — synchronous, only fine for `greedy` or a quick single genetic/hybrid run.
- `POST /api/optimize/async` + `GET /api/jobs/{job_id}` — starts the run in a background thread and returns a `job_id` immediately; the frontend polls every 3s. Any algorithm other than `greedy` should go through this path (see the frontend's `runGeneration`).
- A GitHub Actions workflow (`.github/workflows/keep-alive.yml`) pings `/api/health` every 10 minutes so Render's free tier doesn't spin the service down from inactivity during idle periods.

## Known gotcha: shared Supabase client + auth session expiry

`supabase` in `database/db.py` is a single module-level client reused across every request from every user. Its `ClientOptions` are explicitly set to `auto_refresh_token=False, persist_session=False` — **do not remove this**. With the defaults (both `True`), any login or token operation silently registers a background auto-refresh timer on that shared client, which rotates a user's Supabase refresh token on the server without the frontend knowing. Since the frontend (not the backend) owns token lifecycle via `localStorage`, that silent rotation invalidates the frontend's copy of the refresh token — surfacing as a forced logout roughly an hour after login, which was especially painful mid-way through a long GA/hybrid run. `GET/POST` auth endpoints (login, signup, `/api/auth/refresh`) all go through this same client, so this setting must stay off for all of them.

## Algorithms: `Optimization/Algorithims/` (registered in `Optimization/engine.py`'s `ALGORITHM_REGISTRY`)

`greedy`, `genetic`, `hybrid` (aliases `genetic.py`'s function names but is actually a from-scratch MRV + LNS + min-conflicts + ejection-chain repair pipeline — read the module docstring in `hybrid.py`, don't assume it's just GA+tabu), `grasp`, `pso`. All of `genetic`/`hybrid`/`grasp`/`pso` are called as `fn(sections, timeslots, rooms, num_runs=N)` and return the single best schedule across `num_runs` independent restarts.

### Fitness ceiling is often data-limited, not algorithm-limited

On the real production dataset (~4700 sections), GRASP topped out around fitness **0.8239** (502 unscheduled). This was reproduced almost exactly by a synthetic hard-contention benchmark used while building PSO (0.827-0.828, ~489-491 unscheduled) — and critically, **running more repair passes on that instance changed nothing** (5 rounds of PSO's ejection-chain rescue all converged to the exact same 489 unscheduled). That's strong evidence the ~0.82-0.85 ceiling on tightly-constrained instances is a genuine supply/demand imbalance in the data (not enough rooms of the right type/campus/dept combination for every section that needs one), not a weakness in any particular algorithm. Easier instances (less contention) comfortably clear 0.94-0.98 with the same code. Don't chase "get algorithm X above 0.90" as a universal target without first checking whether the specific dataset has this kind of structural shortage — if it does, no metaheuristic will close that gap; only adding/reallocating rooms or loosening a hard constraint will.

### GRASP (`grasp.py`) — history worth knowing before touching it again

Went through three real performance bugs before reaching its current form, each only visible at real dataset scale (~4700 sections), not in small tests: (1) `local_search` calling the O(sections) `calculate_fitness()` per candidate instead of tracking conflicts incrementally via `Counter`-based occupancy maps — fixed, now O(1) per candidate; (2) `choose_grasp_assignment`/construction having no early-exit (unlike `genetic.py`'s first-feasible-match), forcing a full room×timeslot scan per section — fixed via `CONSTRUCTION_ROOM_SAMPLE`/`CONSTRUCTION_TIMESLOT_SAMPLE` bounded sampling with fallback to a full scan only if the sample is empty; (3) a teammate's later rewrite added richer candidate scoring (`score_candidate`, now also considers room type/campus/dept/timeslot guideline, not just capacity) but had a bug where it read `section.course_type` (doesn't exist — it's `section.course.type`), silently making the room-type scoring term a no-op — fixed. `scan_candidates` (was `_scan_candidates`) was promoted to a public, cross-module helper for exactly this reason: `pso.py` reuses it directly rather than re-implementing the same bounded-scan-with-fallback logic.

### PSO (`pso.py`)

Encoding follows Chen & Shih (2013): one continuous priority value per section, decoded by ranking (argsort, largest-first) into a placement order, with each section greedily placed into its single best still-feasible slot (deterministic, unlike GRASP's randomized RCL — PSO's diversity comes from the swarm exploring different *orderings*, not different placements for a fixed order). Two things worth knowing if you touch this again:

- **There is deliberately no interchange local search step**, unlike the SPSOLS paper's design. This decoder only ever accepts a candidate that already passes every hard constraint against current occupancy, so a freshly decoded schedule has zero conflicts by construction — there's nothing for a conflict-shuffling local search to fix. Verified empirically (not assumed): running `grasp.py`'s incremental `local_search` on a decoded/rescued PSO schedule at production scale changed fitness by exactly 0.0 while costing 0.5-8s per call. The only lever left is unscheduled sections, which is what `rescue_unscheduled()`'s depth-1 ejection chain targets instead.
- **Rescue only runs on the current global-best particle**, not every particle every iteration (`n_particles x iterations x O(unscheduled)` would reproduce GRASP's original blow-up). This is an intentional Lamarckian/memetic pattern (Moscato 1989), not an oversight — the cost self-limits to the number of genuine improvements the swarm actually finds.
- Sample-size constants (`CONSTRUCTION_ROOM_SAMPLE`/`TIMESLOT_SAMPLE=8`, `RESCUE_ROOM_SAMPLE`/`TIMESLOT_SAMPLE=10`, `N_PARTICLES=10`, `ITERATIONS=14`) were tuned empirically against a 4700-section synthetic benchmark to land a single `pso_schedule()` call around 35-60s (comparable to GRASP's per-run cost) — decode() needs the *best* sampled candidate (no early exit, unlike GRASP/GA's first-feasible construction), so it's inherently pricier per call and needed a smaller sample to compensate.
