# Writing a New Algorithm for This System

Reference for anyone — human or AI agent — adding a scheduling algorithm to this repository.

**If you are an AI agent:** this file is designed to be sufficient on its own. You should not need
the rest of the repository read into context to write a correct algorithm. Everything in §3–§7 is
verified against the current code. Follow §4 exactly; those four requirements are enforced by the
test suite and three of them have been violated by every algorithm written here so far.

**If you are a human:** read §1, §2 and §7 before you design anything. §7 is a list of mistakes that
have already been made in this codebase, each of which cost days.

---

## 0. Getting set up, and getting your code in

### You do not need database credentials to write or test an algorithm

The test suite runs **fully offline**, against a committed snapshot in `tests/fixtures/`. Nothing in
`tests/` imports `backend.database.db`. So you can clone, install, write your algorithm, and run the
entire suite with **no environment variables and no database access at all**.

Credentials are only needed for two things, and neither is part of writing an algorithm:

- running the API server (`src/backend/main.py`) against live data
- running the scripts in `scripts/`, which query Supabase directly

### Option A — work through GitHub in the browser (no clone, no credentials)

Fine for small edits. You cannot run the tests this way, so the algorithm must still be tested by
someone who has cloned, before the pull request is merged.

1. On the repository page, use the branch dropdown → type a new branch name → **Create branch**.
   Name it for what it does: `add-simulated-annealing`, not `test2`.
2. Navigate to the folder, **Add file → Create new file** (or **Edit** the pencil icon on an
   existing one).
3. At the bottom, choose **Commit directly to the new branch** — never to `main`.
4. **Compare & pull request** → describe what you changed → request a review.

### Option B — clone and work locally (required to run the tests)

```bash
git clone https://github.com/<owner>/scheduling-optimization-tool.git
cd scheduling-optimization-tool

# Python 3.11. A Windows Store Python will fail with an "untrusted mount point"
# error -- install a standard Python 3.11 and use its full path.
python -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt

# verify your setup -- this needs no credentials
.\.venv311\Scripts\python.exe -m pytest
```

Always call the interpreter by its full path (`.\.venv311\Scripts\python.exe`). A bare `python` may
resolve to a broken stub.

**Only if you also need to run the API or the `scripts/` diagnostics**, create a `.env` in the repo
root:

```
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=...
```

Ask the repository owner for the values. **Never commit `.env`** — it is in `.gitignore`, and a
previous `.env` committed to this repository put a live database password in public history that had
to be rotated. If you ever see credentials in a diff, stop and say so before pushing.

### Branch and pull request — always, even though we can push to `main`

The repository is private, so `main` is not protected and direct commits to it will succeed. **Do
not.** A direct push to `main` skips review, skips the test suite, and has already broken this
repository once: a new `hybrid.py` was merged with a different function signature than the one
`tests/test_algorithms.py` calls, and `main` sat red without anyone noticing.

```bash
git checkout -b add-simulated-annealing
# ... work, run the tests ...
git add src/backend/Optimization/Algorithims/simulated_annealing.py
git commit -m "Add simulated annealing scheduler"
git push -u origin add-simulated-annealing
```

Then open a pull request on GitHub and request a review.

**Before you open it:** pull the latest `main` into your branch and run the full suite again. If
someone else changed a shared file — the `constraints/` package, `evaluation.py`, `models.py`, or another
algorithm — your branch can pass alone and still break once merged. That is exactly how the hybrid
collision above happened.

```bash
git fetch origin
git merge origin/main
.\.venv311\Scripts\python.exe -m pytest
```

**Never force-push a shared branch**, and never rewrite history on `main`.

---

## 1. What you are actually solving

This is a **University Course Timetabling (UCT)** problem on a real institutional dataset.

### The entity being scheduled

**One entity = one `(course, section, meeting pattern)` triple.** Not one database row.

The source table stores **one row per instructor**, so 4,698 rows describe **3,270 real meetings**.
777 rows are exact repeats and 651 are supervision-roster rows. **Never treat a row count as
demand.** The `Section` objects you are handed are already collapsed correctly — you do not have to
do this yourself, but you must not undo it.

Scale: **3,270 sections, 375 rooms, 466 timeslots.** Of the sections, 106 have multiple instructors,
11 have multiple meeting patterns, and 313 are supervision.

### Not every section needs a room and a time

This is the single most important modelling fact in the system. Four categories:

| category | meaning | example |
| --- | --- | --- |
| `NEEDS_ROOM_AND_TIME` | an ordinary class | undergraduate lecture |
| `NEEDS_TIME_ONLY` | meets at an hour but holds no room | clinical practice, some labs |
| `NEEDS_ROOM_ONLY` | rare | — |
| `NEEDS_NOTHING` | no fixed meeting at all | thesis / dissertation supervision |

Call `section_needs(section)` and **respect the answer**. Assigning a room to a thesis supervision
makes 2,700 sections compete for rooms that 330 of them never needed, which is how an earlier
version of this system produced ~502 unplaceable sections.

### Two other data shapes that break naive code

- **A `day` value encodes several days at once.** `"MTR"` means Mon+Tue+Thu in a *single* timeslot
  row. Anything assuming one day per slot will miscount overlap.
- **A section can have several meeting patterns.** Clinical sections genuinely meet in five separate
  blocks across the week. Each block is its own `Section` sharing a `sibling_key`. Deduplicating on
  `(course, section)` deletes real meetings.

### Supervision is not teaching

The `instructor_id` column mixes two unrelated relationships:

- **Teaching** — this person is in this room at this hour. Two at once is impossible.
- **Supervision** — this person is nominally attached to a student's thesis or project.

Twenty-two supervisors on a senior project are not twenty-two people in a room. Counting the second
as the first invented 453 double-bookings that nobody experiences. **`instructor_occupancy_ids()`
already encodes this distinction — always go through it, never read `.instructor_id`.**

---

## 2. The objective: three tiers

Constraints are **tiered**, and the tiers are priced so they can never trade against each other
wrongly:

| tier | weight | what |
| --- | --- | --- |
| **HARD** | 3.0 | a room, an instructor, or one section's own two blocks in two places at one hour |
| **UNSCHEDULED** | 2.0 | a section left with no placement |
| **SOFT** | 0.5 | room type, campus, department, capacity overflow, and the time axis |

### The rule that follows from those weights

**Never leave a section unscheduled to satisfy a soft rule, and never force a hard conflict to avoid
leaving one unscheduled.** Unscheduled (2.0) is cheaper than a hard conflict (3.0) and dearer than
any soft violation. Two algorithms here got this backwards by falling back to `valid_ts[0]` when no
free slot existed — creating a guaranteed double-booking to avoid a cheaper outcome.

If no conflict-free placement exists, **leave the section unplaced.** That is the correct answer.

### Why the tiers exist at all

An earlier model treated the university's *soft preferences* as *hard constraints*. The real manual
timetable breaks those "hard" rules 11–32% each, so the model had outlawed placements the university
makes routinely. That — not a facility shortage, not algorithm weakness — caused a ceiling of ~502
unscheduled sections and 0.82 fitness. After the split, every algorithm places every section.

**The lesson generalises: if your algorithm hits a wall, check whether the constraint model is
refusing something legal before you blame the search.**

### The soft rules live on two axes

- `room_soft_penalty(obj, room)` prices **the room** (type 4.0, campus 3.0, capacity 2.0 + overflow
  scaling 4.0, department 1.0).
- `time_soft_penalty(obj, ts)` prices **the hour** (supervision in daytime 3.0, teaching in evening
  2.0; cutoff 17:00).

**A candidate scorer that reads only the first is blind to half the objective.** That is exactly how
three algorithms here came to put ~450 undergraduate classes in the evening while the fitness they
were judged by charged for every one.

---

## 3. The interface

### Signature

```python
def yourname_schedule(sections, timeslots, rooms, valid_timeslot_cache=None):
    """One run. Returns a list[ScheduleItem]."""

def yourname_runs(sections, timeslots, rooms, num_runs=30, seeds=None):
    """N independent restarts. Returns the single best schedule by fitness."""
```

The engine calls the `_runs` form as `fn(sections, timeslots, rooms, num_runs=N)`. A deterministic
algorithm (like `greedy`) may expose only the single-run form.

**`seeds=` is required on the `_runs` form** and is how the shared seed list reaches your
algorithm — §9. Default it to `None` (meaning "use the shared list") so the API keeps working
unchanged; all five existing algorithms already accept it.

### Registration — both places required

1. **`src/backend/Optimization/engine.py`** — import it and add to `ALGORITHM_REGISTRY`:
   ```python
   ALGORITHM_REGISTRY = {
       "greedy": greedy_schedule,
       "genetic": genetic_runs,
       "hybrid": hybrid_runs,
       "grasp": grasp_runs,
       "pso": pso_runs,
       "yourname": yourname_runs,        # <- add
   }
   ```
   Also add `"yourname"` to the tuple in `SchedulingEngine.run()` that decides which algorithms get
   `num_runs=` passed.

2. **`tests/test_algorithms.py`** — add `"yourname"` to `ALGORITHMS` **and** a branch to `_run()`.
   This is the definition of done. See §8.

---

## 4. The four non-negotiable requirements

Every one of these has been violated in this codebase, most of them more than once.

### 4.1 Call `finalize_schedule()` on what you return

```python
return finalize_schedule(schedule, sections, timeslots, valid_timeslot_cache, rooms=rooms)
```

It separates a section's own meeting blocks in time, then puts them in one shared room. The order
matters and is handled for you. It only ever moves a block into a slot/room that is genuinely free,
so it can improve a schedule but never break one. **Pass `rooms=`** — without it the room
unification can only reuse rooms the blocks already sit in, and gives up when those are busy.

### 4.2 Build occupancy through `instructor_occupancy_ids()`

```python
for instructor_id in instructor_occupancy_ids(section):   # plural. always.
    occupied_instructors.add((instructor_id, ts.id))
```

Returns a **tuple** — empty for supervision and for the 56% of rows with no instructor, and
multiple ids for the 106 team-taught sections. The singular `instructor_occupancy_id()` exists for
sort keys and display only. **Anything counting or reserving must use the plural form.**

**Instructor occupancy is not conditional on having a room.** Time-only classes hold no room but do
hold their teachers' hours. Forgetting this made clinical teachers invisible to one algorithm's
local search, which then moved classes onto hours their instructors were already teaching.

If you track occupancy incrementally (build → decrement on move → increment on move), **all three
paths must agree**. If they disagree about supervision or multi-instructor classes, the counter
drifts negative and your conflict delta silently stops meaning anything.

### 4.3 Carry every field into every `ScheduleItem`

```python
item = ScheduleItem(
    course_id=section.course.id,
    course_name=section.course.name,
    course_type=section.course.type,
    course_dept=section.course.dept,
    capacity=section.capacity,
    instructor_id=section.instructor_id,     # scalar, for reporting/DB only
    room_id=room.id if room else None,
    timeslot_id=timeslot.id if timeslot else None,
    section=str(section.no),
    instructor_ids=section.instructor_ids,   # REQUIRED
    pattern_index=section.pattern_index,     # REQUIRED
    level=getattr(section.course, "level", None),          # REQUIRED
    course_class=getattr(section.course, "course_class", None),  # REQUIRED
)
```

**`level` and `course_class` drive classification.** An item that loses them re-classifies
downstream as an ordinary room+time lecture, and the whole four-category model silently undoes
itself. This bug has been introduced **three separate times**. It is pinned by
`test_greedy_carries_level_into_every_item`.

### 4.4 Pass the shared test suite

`pytest tests/test_algorithms.py` with your algorithm in `ALGORITHMS`. Not negotiable — see §8.

---

## 5. API reference

Everything you need. You should not have to read the implementation of any of these.

### Classification — `backend.Optimization.constraints`

```python
section_needs(section) -> Needs        # .room and .time booleans. Ask before placing.
classify_section(section) -> str       # "NEEDS_ROOM_AND_TIME" | "NEEDS_TIME_ONLY" | ...
needs_for(course_type, level) -> Needs # same, from a ScheduleItem's fields
is_supervision(obj) -> bool            # does this person hold a room at this hour (no)
```

### Candidates — these **RANK**, they do not filter

```python
get_viable_rooms(section, rooms, limit=25) -> list        # best-first, nothing dropped
get_viable_rooms_for_schedule_item(item, rooms, limit=25) -> list
get_valid_timeslots_for_section(section, timeslots) -> list
rank_timeslots(obj, timeslots) -> list                    # preferred hours first
```

**This is the most misunderstood contract in the codebase.** These return candidates ordered
best-first with **nothing removed for a soft reason**. Consequences:

- If you **walk** the list and take the first free entry, you get the soft preferences for free.
- If you **sample** the list and score candidates yourself, you must price the soft rules in your own
  scorer — otherwise you are blind to them and your fitness suffers for a reason no column shows.

`get_valid_timeslots_for_section()` is the composition of a hard filter (does a 75-minute lecture
fit this block?) and a soft ranking. The sort is **stable** and keyed only on penalty, so
equally-preferred slots keep their original order — do not add a finer sort key, or every section of
one kind funnels onto the same few hours.

### Feasibility

```python
passes_hard_constraints(section, room, timeslot,
                        schedule=None,
                        occupied_instructors=None,
                        occupied_rooms=None) -> bool
```

Hard rules only: no double-booked room, no double-booked instructor. Pass the occupancy sets if you
have them (O(1)); otherwise it scans the partial schedule (O(n)).

```python
instructor_occupancy_ids(obj) -> tuple    # see 4.2
sibling_key(obj)                          # blocks of one section share this
is_room_free_map(occupied_rooms, room_id, timeslot_id) -> bool
is_instructor_free_map(occupied_instructors, instructor_id, timeslot_id) -> bool
```

### Soft costs — for your own candidate scorer

```python
room_soft_penalty(obj, room) -> float     # type, campus, capacity, department
time_soft_penalty(obj, ts) -> float       # supervision-daytime, teaching-evening
```

If your algorithm scores candidates rather than calling `calculate_fitness`, **it must use both.**

### Repair

```python
finalize_schedule(schedule, sections, timeslots, valid_timeslot_cache=None, rooms=None)
```

### Evaluation — `backend.Optimization.evaluation`

```python
calculate_fitness(schedule, rooms, sections=None, total_sections=None,
                  timeslots=None, valid_timeslot_cache=None) -> float
```
The objective. **Always pass `sections=` and `timeslots=`** — without `sections` it cannot count
unscheduled correctly, and without `timeslots` it cannot price the time axis.

```python
count_hard_conflicts(schedule) -> int     # "can this be run?" — the only feasibility test
total_soft_penalty(schedule, rooms) -> float
total_time_penalty(schedule, timeslots) -> float
soft_violation_counts(schedule, rooms, timeslots=None) -> dict
build_timeslot_guideline_cache(sections, timeslots) -> dict   # build ONCE, pass everywhere
```

> **`count_all_violations_flat()` is a display number only.** It adds hard and soft at one point
> each, so a schedule with 3 double-bookings scores 3 and one with 40 campus mismatches scores 40 —
> the runnable schedule gets the worse number. Never use it as a feasibility test and never put it
> in a report.

### Seeding — `backend.Optimization.benchmark_seeds`

```python
SEEDS                              # the shared list, 30 seeds. Do not edit.
seed_all(seed) -> seed              # seeds random + numpy globals; returns the seed
seed_for_run(run_index, seeds=None) # seed for run i, from `seeds` or SEEDS
```

Call `seed_all(seed_for_run(run, seeds))` once at the top of each run in your `_runs`. See §9.

### Models — `backend.models.models`

`Section`: `.course` (`.id .name .type .dept .level .course_class`), `.no`, `.capacity`,
`.instructor_ids` (tuple), `.instructor_id` (first, reporting only), `.pattern_index`,
`.pattern_count`, `.sibling_key`, `.id` = `(course_id, section_no, pattern_index)`.

`Room`: `.id .capacity .type .building .dept_id .no`
`Timeslot`: `.id .day .start .end` — `.start`/`.end` are `"HH:MM:SS"` **strings**, not `time`
objects. (Assuming otherwise silently returned duration 0 for every section, letting lectures land
in 3-hour blocks. It went unnoticed for months.)

---

## 6. Skeleton

```python
import random
from backend.Optimization.constraints import (
    section_needs, get_viable_rooms, get_valid_timeslots_for_section,
    passes_hard_constraints, instructor_occupancy_ids, finalize_schedule,
    room_soft_penalty, time_soft_penalty,
)
from backend.Optimization.evaluation import (
    calculate_fitness, build_timeslot_guideline_cache,
)
from backend.Optimization.benchmark_seeds import seed_all, seed_for_run
from backend.models.models import ScheduleItem


def yourname_schedule(sections, timeslots, rooms, valid_timeslot_cache=None):
    if valid_timeslot_cache is None:
        valid_timeslot_cache = build_timeslot_guideline_cache(sections, timeslots)

    # Precompute candidates once. Sections that need no room get an empty list,
    # so they never compete for one.
    candidates = []
    for s in sections:
        needs = section_needs(s)
        candidates.append((
            s, needs,
            get_viable_rooms(s, rooms) if needs.room else [],
            get_valid_timeslots_for_section(s, timeslots) if needs.time else [],
        ))

    schedule = []
    occupied_rooms = set()          # (room_id, timeslot_id)
    occupied_instructors = set()    # (instructor_id, timeslot_id)

    for section, needs, viable_rooms, valid_slots in candidates:
        room = timeslot = None

        # ---- your search goes here. The contract, whatever the method: ----
        #  * only place what section_needs() says is needed
        #  * check passes_hard_constraints() before committing
        #  * if you SAMPLE rather than walk the ranked lists, score candidates
        #    with room_soft_penalty() AND time_soft_penalty()
        #  * if nothing fits, leave room/timeslot as None. Do NOT force a
        #    conflicting placement -- unscheduled (2.0) < hard conflict (3.0).

        schedule.append(ScheduleItem(
            course_id=section.course.id,
            course_name=section.course.name,
            course_type=section.course.type,
            course_dept=section.course.dept,
            capacity=section.capacity,
            instructor_id=section.instructor_id,
            room_id=room.id if room else None,
            timeslot_id=timeslot.id if timeslot else None,
            section=str(section.no),
            instructor_ids=section.instructor_ids,
            pattern_index=section.pattern_index,
            level=getattr(section.course, "level", None),
            course_class=getattr(section.course, "course_class", None),
        ))

        if room is not None and timeslot is not None:
            occupied_rooms.add((room.id, timeslot.id))
        if timeslot is not None:
            # plural, and NOT conditional on having a room
            for iid in instructor_occupancy_ids(section):
                occupied_instructors.add((iid, timeslot.id))

    return finalize_schedule(schedule, sections, timeslots,
                             valid_timeslot_cache, rooms=rooms)


def yourname_runs(sections, timeslots, rooms, num_runs=30, seeds=None):
    cache = build_timeslot_guideline_cache(sections, timeslots)
    best, best_fit = None, -1.0
    for run in range(num_runs):
        # The shared benchmark seeds (§9). Once per run, before any randomness
        # is consumed. If you build your own generator instead of using the
        # `random` module, thread run_seed into it -- seed_all cannot reach it.
        run_seed = seed_all(seed_for_run(run, seeds))

        sched = yourname_schedule(sections, timeslots, rooms, valid_timeslot_cache=cache)
        fit = calculate_fitness(sched, rooms, sections=sections,
                                timeslots=timeslots, valid_timeslot_cache=cache)
        # Per-run line for the benchmark table -- the seed belongs in it.
        print(f"Run {run + 1:2d} (seed {run_seed}): fitness = {fit:.4f}")
        if fit > best_fit:
            best, best_fit = sched, fit
    return best
```

---

## 7. Traps — every one of these has actually happened here

1. **Verify your randomness**: build a
   population, compare the members, and assert they differ. A `random.choice` in a fallback branch
   that never fires is not randomness.

4. **No elitism.** Replacing the population wholesale and reading the best from the *final*
   generation lets a search end worse than it was at generation 3. Track the global best.
5. **Scoring candidates without the time axis.** See §2 and §5.
6. **Forcing a conflict instead of leaving a section unplaced.** See §2.
7. **Seeding an occupancy map only from roomed items.** Time-only classes hold their teachers'
   hours. See §4.2.
8. **Reading `.instructor_id` for occupancy.** Silently ignores every instructor after the first.
9. **Dropping `level` / `course_class`.** See §4.3.
10. **Keying a course rule on `level`.** It carries eight values, and 356 of 1,959 timed
    `Lecture Undergraduate` rows are filed under `Fine Art`, `Diploma`, `Intensive English` or
    `Foundation Year` — all silently exempted from a rule their type obeys at 94%. **Key on
    `course_type`**, like every other rule in the module.
11. **Rooms with capacity 0 mean "unknown", not "no seats".** They are charged the full overflow
    penalty so they never look like a perfect fit.
12. **A synthetic test dataset small enough to read is usually small enough to pass by accident.**
    Three of our guards first passed with the bug reintroduced. See §8.
13. **Seeding one generator when your module draws from two.** `seed_all()` reaches the `random`
    module and numpy's global state, and nothing else. A `np.random.default_rng(...)` or
    `random.Random(...)` you built yourself ignores it, so the run half-repeats and the benchmark
    is not reproducible. PSO draws from both — see how `pso_runs` handles it, and §9.

---

## 8. Definition of done

```bash
.\.venv311\Scripts\python.exe -m pytest
```

All tests must pass **with your algorithm's name in `ALGORITHMS`** in `tests/test_algorithms.py`.
That parametrised suite runs every algorithm over synthetic datasets built to contain the awkward
shapes — team-taught time-only sections, a multi-block section with no instructor, a two-block studio
needing a shared room — and checks:

| test | what it catches |
| --- | --- |
| `test_no_hard_conflicts` | all three hard rules |
| `test_every_instructor_is_carried_through` | the singular-instructor bug |
| `test_sibling_blocks_share_a_room` | missing `finalize_schedule` |
| `test_all_levels_survive` | dropped `level` |
| `test_honours_the_evening_preference` | a scorer blind to the time axis |
| `test_the_preference_never_costs_a_placement` | a preference used as a filter |

It would have caught three of the four bugs found during the last rework. Treat "that suite passes
with my algorithm listed" as the definition of done.

**Verify your own regression guards.** If you add a test, reintroduce the bug and confirm the test
fails. A guard that does not catch its own bug is worse than none, because it looks like coverage.

Note: the `tests/test_data_invariants.py` tests assert facts about the **dataset**, not the code.
If they fail, the data moved — run `tests/snapshot_refresh.py`, read the diff, and update the
expected numbers deliberately. **Do not "fix" them by editing the number to match.**

---

## 9. Benchmark protocol

Fixed before anyone writes code, so ten algorithms from four people stay comparable. Do not deviate
without telling the team, because a deviation invalidates the comparison for everyone.

- **30 independent runs** per algorithm on the full dataset. Greedy is deterministic; one run is its
  whole distribution.
- **Everyone uses the shared seed list** — see below. This is not optional.
- **Report per run:** fitness, hard conflicts, unscheduled, room soft penalty, time penalty,
  wall-clock, and the seed.
- **Report across runs:** mean ± sample standard deviation, and the median for time.
- **Identical stopping criterion** across algorithms — agree wall-clock or evaluation count, not
  "whatever each one happens to do".
- **A standard deviation of exactly zero is a red flag, not a strength.** It means your search is not
  searching. See §7.1.

### The shared seed list

`src/backend/Optimization/benchmark_seeds.py` holds one list, `SEEDS`, and every algorithm's
benchmark uses it in the same order. Run 1 of the GA and run 1 of your algorithm use the same seed.

```python
from backend.Optimization.benchmark_seeds import SEEDS, seed_all, seed_for_run

def yourname_runs(sections, timeslots, rooms, num_runs=30, seeds=None):
    cache = build_timeslot_guideline_cache(sections, timeslots)
    best, best_fit = None, -1.0

    for run in range(num_runs):
        # ONCE per run, before any randomness is consumed.
        run_seed = seed_all(seed_for_run(run, seeds))

        sched = yourname_schedule(sections, timeslots, rooms, valid_timeslot_cache=cache)
        fit = calculate_fitness(sched, rooms, sections=sections,
                                timeslots=timeslots, valid_timeslot_cache=cache)
        print(f"Run {run + 1:2d} (seed {run_seed}): fitness = {fit:.4f}")
        ...
```

**`seed_all()` covers the global generators only** — `random` and `numpy.random`. If your module
builds its own generator (`np.random.default_rng(...)`, `random.Random(...)`), that generator does
not see the global seed and you must thread the value in yourself. `pso.py` is the worked example:
it draws from *both*, so `pso_runs` calls `seed_all(...)` **and** passes `seed=run_seed` down into
`pso_schedule`. Seeding only one of two generators leaves half the run unreproducible, and the
symptom — runs that almost repeat — is easy to miss.

**Verify it, don't assume it.** Run your `_runs` twice with the same `seeds=` and compare the
schedules item by item; they must be identical. Then run it with a different list and confirm the
output changes. The second half matters as much as the first: a run that is identical under *every*
seed list is not seeded, it is deterministic, which is §7.1 all over again.

**Do not edit `SEEDS`.** It was fixed before anyone had results, which is the whole point — it
removes seed choice as something an author can tune in their own favour, after the fact, without
anyone being able to tell. If you need more than 30 runs, extend the list at the end and say so in
the methodology.

### What the shared list does and does not prove

Be precise about this in the paper, because it is an easy thing to overclaim and a cheap thing for a
reviewer to attack.

- It **does** give reproducibility — anyone can re-run run 7 and get run 7 — and it removes seed
  cherry-picking.
- It **does not** give the algorithms the same random draws. A GA and a PSO consume randomness in
  different amounts and in a different order, so "seed 7" is not a shared condition. Do **not**
  claim common-random-numbers variance reduction, and do **not** pair run *i* of A against run *i*
  of B in a statistical test — the pairing would be arbitrary. For one instance and 30 runs each,
  compare the distributions with an unpaired rank test (Mann–Whitney U).

---

## 10. Academic standards

This work is going into a paper. The algorithm you write must be defensible as well as functional.

### Implement from a real, named source

- **Choose a published algorithm and cite the paper you implemented from.** Put the citation in the
  module docstring: authors, year, title, venue, and a DOI or arXiv ID.
- **Verify every reference exists before you use it.** Open it. Check the authors, year and venue
  against the actual publisher page. Language models fabricate plausible-looking citations —
  correct-sounding author names attached to papers that were never written. A fabricated citation
  discovered by a reviewer discredits the entire paper, not just that line.
- **Do not describe a method you did not implement.** If you simplified, say which part and why, in
  the docstring. 
- **If the docstring describes something other than what the code does, fix one of them.** 

### Choosing what to implement

- **Prefer taxonomy coverage over novelty.** We already have a constructive heuristic, a GA, a PSO,
  a GRASP and a hybrid. A fourth population-based swarm method adds nothing to the paper.
  What is missing is trajectory-based local search (simulated annealing, tabu search, late acceptance
  hill climbing), an exact or hybrid-exact reference, and a hyper-heuristic.
- **Be aware of the metaphor critique.** There is a substantial literature criticising "novel"
  metaphor-named metaheuristics as rediscoveries of existing methods, published with weak
  benchmarking. Read it before picking an animal-named 2023 optimizer, and be ready to justify your
  choice on mechanism rather than novelty.
- **An exact method on a reduced sub-instance is worth more than another metaheuristic.** "We are
  0.4% from the proven optimum" is a far stronger claim than "ours scored highest among the five we
  wrote."

### Reporting

- Report the tiers separately — hard conflicts, unscheduled, room soft, time soft — not just the
  scalar fitness. The scalar compresses every interesting difference into the last 1%.
- Report negative and messy results. An algorithm that underperforms is a finding; an algorithm that
  underperforms for a reason you diagnosed is a contribution.
- State known weaknesses before a reviewer does.

---

## 11. Checklist

Before opening a pull request:

- [ ] `section_needs()` consulted; nothing placed that does not need placing
- [ ] `instructor_occupancy_ids()` (plural) used everywhere occupancy is built
- [ ] occupancy recorded for time-only sections, not just roomed ones
- [ ] `level`, `course_class`, `instructor_ids`, `pattern_index` on every `ScheduleItem`
- [ ] `finalize_schedule(..., rooms=rooms)` called on the returned schedule
- [ ] nothing forced into a conflicting placement to avoid leaving it unscheduled
- [ ] if candidates are sampled: both `room_soft_penalty` and `time_soft_penalty` in the scorer
- [ ] randomness verified — repeated runs actually differ
- [ ] `_runs` accepts `seeds=None` and calls `seed_all(seed_for_run(run, seeds))` once per run
- [ ] every generator seeded, including any `default_rng` / `random.Random` you built yourself
- [ ] reproducibility verified — same `seeds=` twice gives identical schedules, a different list
      gives a different one
- [ ] registered in `ALGORITHM_REGISTRY` and in `SchedulingEngine.run()`'s `num_runs` tuple
- [ ] added to `ALGORITHMS` and `_run()` in `tests/test_algorithms.py`
- [ ] full suite passes
- [ ] any new regression guard verified by reintroducing its bug
- [ ] module docstring names the paper implemented from, with a DOI/arXiv ID that you opened
- [ ] benchmark run under §9 and the numbers recorded with their seeds
