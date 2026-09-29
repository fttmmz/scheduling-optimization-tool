"""Which timeslots a section may use, and in what order of preference.

The hard part (duration and day pattern) is _candidate_timeslots(); the soft
part (the evening preference) is time_soft_penalty() and rank_timeslots()."""

import re

from .classification import _course_identity, is_supervision


SINGLE_DAY = {"M", "T", "W", "R"}

# Duration constants (in minutes) — used to filter timeslots by expected length
DURATION_75  = 75    # 1h15m  — standard lecture
DURATION_100 = 100   # 1h40m  — Intro to IT combined lec/lab
DURATION_110 = 110   # 1h50m  — connected lab (X / Y suffix)
DURATION_170 = 170   # 2h50m  — independent lab (L suffix) or Training
DURATION_410 = 410   # 6h50m  — intensive English / Training (daily block)

# Special course name fragment used for Intro-to-IT detection
INTRO_IT_NAME_FRAGMENT = "introduction to it"


# internal helpers

def _section_suffix(section_no: str) -> str:
    """
    Return the trailing alphabetic suffix of a section number, upper-cased.
    Examples: '04L' → 'L', '02X' → 'X', '04' → '', '04T' → 'T'
    """
    return re.sub(r"^[^A-Za-z]*", "", str(section_no)).upper()


def _is_independent_lab(section) -> bool:
    """Section number ends with 'L' — standalone lab, 2h50m."""
    return _section_suffix(section.no) == "L"


def _is_connected_lab(section) -> bool:
    """Section number ends with 'X' or 'Y' — lab paired with a lecture, 1h50m."""
    return _section_suffix(section.no) in {"X", "Y"}


def _is_tutorial(section) -> bool:
    """Section number ends with 'T' — tutorial linked to a lecture."""
    return _section_suffix(section.no) == "T"


def _is_intro_it(section) -> bool:
    """
    True if this section belongs to an 'Intro to IT' course.
    Matching is case-insensitive on the course name.
    """
    name = getattr(section.course, "name", "") or ""
    return INTRO_IT_NAME_FRAGMENT in name.lower()


def _is_intensive_english(section) -> bool:
    """
    True for intensive-English sections that meet every day for 6h50m.
    Detected via course name fragment; extend as needed.
    """
    name = getattr(section.course, "name", "") or ""
    return "intensive english" in name.lower()


def _parse_time_to_minutes(value):
    """
    Convert a time value to minutes-since-midnight. Accepts a
    datetime.time-like object (.hour/.minute) or a string in "HH:MM[:SS]"
    format -- Timeslot.start/.end (models.py) are loaded straight from
    Supabase as "HH:MM:SS" strings, not datetime.time objects. Returns
    None if the value is missing or unparseable.
    """
    if value is None:
        return None
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return value.hour * 60 + value.minute
    try:
        parts = str(value).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return None


def _timeslot_duration(ts) -> int:
    """
    Return the duration of a timeslot in minutes. Falls back to 0 if
    start/end are missing or unparseable.

    Previously looked for ts.start_time/.end_time (the actual attributes
    are ts.start/.end -- see Timeslot in models.py) and assumed
    datetime.time objects, when they're actually "HH:MM:SS" strings. Both
    mismatches independently guaranteed an AttributeError on every real
    timeslot, so this always returned 0 -- meaning every duration-based
    rule below (lecture=75min, lab=170min, etc.) silently never matched
    and every course type fell through to its day-only fallback,
    regardless of the timeslot's actual length.
    """
    start = _parse_time_to_minutes(getattr(ts, "start", None))
    end = _parse_time_to_minutes(getattr(ts, "end", None))
    if start is None or end is None:
        return 0
    return max(0, end - start)


def is_placeholder_timeslot(ts) -> bool:
    """True for rows that encode "no real meeting time" rather than a slot.

    Two encodings exist in this dataset:
      * day IS NULL            -- timeslot_id=1, the TBA bucket (already
                                  filtered out by loader.load_timeslots)
      * start == end           -- 13 zero-duration slots

    The zero-duration ones are not a modelling guess: across the whole manual
    schedule they carry 50 rows and NOT ONE of them has a room. They hold
    supervision-style activities (Prof. Exp. placements, thesis), where several
    sections share the same nominal time under one instructor. Counting them as
    real bookings invented instructor double-bookings that nobody experiences.
    """
    if getattr(ts, "day", None) is None:
        return True
    return _timeslot_duration(ts) <= 0


# -- The evening preference (HANDOVER 4.7) -----------------------------------
# A time-axis soft rule, the counterpart of the room-axis weights in rooms.py.
#
# Measured, not assumed. Only 3 of the 148 timed Senior Project Supervision rows
# start before 17:00, against 6% of undergraduate lectures -- far too lopsided to
# be an accident. The university keeps daytime hours for taught contact and runs
# supervision around them in the evening. (Full per-type rates in the table
# further down, which is what the sets are actually chosen from.)
#
# It is the one Phase 3 preference recoverable from existing data -- every other
# preference (who likes mornings, which students clash) describes what someone
# WANTED, and the schedule table only records what WAS.
#
# Three things about how it is encoded:
#
#   * SOFT, not a filter. get_valid_timeslots_for_section() used to HARD-filter
#     supervision down to slots at/after 15:00. That is exactly the mistake the
#     hard/soft split exists to prevent -- a preference that can strand a
#     section is priced at infinity, and an unscheduled class (2.0) is worse
#     than a badly-timed one. It now ranks instead, so the preference is
#     honoured while candidates last and degrades quietly once they run out.
#
#   * 17:00, not 15:00. The old cutoff was a guess made before anyone measured;
#     the data puts the break at 17:00.
#
#   * Graduate lectures are NEUTRAL. At 52% evening they express no preference
#     either way, so they are charged nothing in either direction. Penalising
#     them would be inventing a rule the data does not contain. Same for
#     Seminar Graduate (27%), Office Hours (47%) and Project (40%).
EVENING_CUTOFF_HOUR = 17

# Priced against the room weights in rooms.py (room_type 4.0, campus 3.0, capacity
# 2.0, department 1.0), and deliberately asymmetric:
#
#   supervision in a daytime slot is the more expensive direction, because it
#   consumes a scarce daytime hour that taught contact needs -- the 98%/6% split
#   says the institution treats daytime as lecture territory.
#
#   an undergraduate lecture pushed into the evening is a real inconvenience to
#   students but takes nothing away from anyone else, so it is cheaper.
SOFT_WEIGHT_SUPERVISION_DAYTIME = 3.0
SOFT_WEIGHT_TEACHING_EVENING = 2.0

# The three sets below are chosen from measured evening rates, NOT from the old
# hard-coded low-priority list. Rates are over timed rows of schedule_id=1 --
# the local test fixture, so tests/test_data_invariants.py can pin them
# offline and a dataset change shows up as a failure rather than as silent drift.
#
# A type only gets a preference when there are at least 20 timed rows to judge
# it by AND they fall at least 80/20 one way. Anything weaker stays neutral:
# charging a penalty for a pattern the data does not show is an opinion with a
# number attached, and this module already carries enough of those.
#
#     course type                        timed   evening   verdict
#     Senior Project Supervision           148       98%   EVENING
#     Lecture Undergraduate              1,959        6%   DAYTIME
#     Laboratory                           293        4%   DAYTIME
#     Clinical Practice                    142        7%   DAYTIME
#     Lecutre / Studio Undergraduate        77        0%   DAYTIME
#     Studio Undergraduate                  21       14%   DAYTIME
#     Training                              21       14%   DAYTIME
#     Lecture Graduate                     477       52%   neutral
#     Seminar Graduate                      56       27%   neutral
#     Office Hours                          34       47%   neutral
#     Project                               20       40%   neutral
#
# Two of those verdicts overturn what the old list assumed, and both were only
# found by measuring:
#
#   Office Hours was treated as low-priority evening work. It is a 47% coin flip.
#   Project is supervision, and still runs in the daytime 60% of the time.

# Supervision by is_supervision() (classification.py), but with no measured preference about
# WHEN it meets. Being supervision answers "does this person occupy a room at
# this hour" (no), which is a different question from "when is this scheduled"
# -- and for Project the two answers diverge. Hence a list rather than
# prefers_evening() simply being is_supervision().
TIME_NEUTRAL_COURSE_TYPES = frozenset({
    "Project",                          # 40% of 20
})

# Not supervision -- these consume their instructors' time and are still checked
# for clashes -- but scheduled around teaching for the same reason.
EVENING_PREFERRED_EXTRA_TYPES = frozenset({
    # 4 timed rows, all 4 in the evening. Below the 20-row bar, kept because it
    # has no counter-evidence and sat beside the supervision types in the old
    # list. Drop it if it ever grows enough rows to be judged properly.
    "Internship",
})

# Taught contact that holds the daytime.
#
# Keyed on course_type, NOT on `level`. `level` looked like the natural field and
# is not usable here: it carries eight values, and 356 of the 1,959 timed
# 'Lecture Undergraduate' rows are filed under 'Fine Art' (157), 'Diploma' (94),
# 'Intensive English' (55) or 'Foundation Year' (50). Keying on
# level == 'Undergraduate' silently exempted all 356 from a rule the type obeys
# at 94%. course_type is also what MINED_NEEDS, SUPERVISION_COURSE_TYPES and
# get_required_room_type() key on, so this stays consistent with the module.
DAYTIME_COURSE_TYPES = frozenset({
    "Lecture Undergraduate",            # 6% evening of 1,959
    "Laboratory",                       # 4% of 293
    "Clinical Practice",                # 7% of 142
    "Lecutre / Studio Undergraduate",   # 0% of 77   (typo is in the source data)
    "Studio Undergraduate",             # 14% of 21
    "Training",                         # 14% of 21
    # Below the 20-row bar. Included because it is undergraduate taught contact
    # agreeing in direction with every type above, over a population of 11.
    "Seminar Undergraduate",            # 18% of 11
})


def is_evening_timeslot(ts) -> bool:
    """True when the slot starts at or after EVENING_CUTOFF_HOUR."""
    start = _parse_time_to_minutes(getattr(ts, "start", None))
    if start is None:
        return False
    return start >= EVENING_CUTOFF_HOUR * 60


def prefers_evening(obj) -> bool:
    """True for supervision and the other activities scheduled around teaching.

    Reuses is_supervision() rather than keeping a second list, so the five
    miscategorised courses of SUPERVISION_COURSE_IDS are covered here too. The
    old hard-coded low-priority list missed all five, and also missed three real
    supervision types (Proposal Doctorate, Comprehensive Exam, Qualifying Exam)
    that the supervision detector already had.

    TIME_NEUTRAL_COURSE_TYPES is subtracted first, for the types that are
    supervision but show no preference about when they meet.
    """
    course_type = _course_identity(obj)[1]
    if course_type in TIME_NEUTRAL_COURSE_TYPES:
        return False
    if is_supervision(obj):
        return True
    return course_type in EVENING_PREFERRED_EXTRA_TYPES


def prefers_daytime(obj) -> bool:
    """True for undergraduate taught contact -- the population that holds the day.

    Checked AFTER prefers_evening() so the two can never both fire: the five
    courses in SUPERVISION_COURSE_IDS carry a taught course_type (one of them is
    typed 'Lecture Undergraduate') and would otherwise match both sets and be
    charged for whichever slot they were given.
    """
    if prefers_evening(obj):
        return False
    return _course_identity(obj)[1] in DAYTIME_COURSE_TYPES


def time_soft_penalty_parts(obj, ts) -> dict:
    """Per-rule soft cost of putting *obj* (Section or ScheduleItem) at *ts*."""
    parts = {"supervision_daytime": 0.0, "teaching_evening": 0.0}
    if ts is None:
        return parts

    evening = is_evening_timeslot(ts)
    if prefers_evening(obj):
        if not evening:
            parts["supervision_daytime"] = SOFT_WEIGHT_SUPERVISION_DAYTIME
    elif prefers_daytime(obj):
        if evening:
            parts["teaching_evening"] = SOFT_WEIGHT_TEACHING_EVENING
    return parts


def time_soft_penalty(obj, ts) -> float:
    """Total soft cost of this timeslot for this section/item. 0.0 == preferred."""
    return sum(time_soft_penalty_parts(obj, ts).values())


def rank_timeslots(obj, timeslots: list) -> list:
    """Preferred timeslots first. Never drops a candidate.

    The sort is STABLE and keyed only on the penalty, so slots that are equally
    preferred keep their original relative order. That matters: greedy, GRASP
    and hybrid walk this list and take the first slot that is free, so a finer
    key would funnel every section of a given kind onto the same few hours.
    """
    return sorted(timeslots, key=lambda ts: time_soft_penalty(obj, ts))


# timeslot functions

def get_valid_timeslots(section, timeslots: list) -> list:

    return get_valid_timeslots_for_section(section, timeslots)


def _candidate_timeslots(section, timeslots: list) -> list:
    """
    Return the subset of *timeslots* that are valid for *section*, applying
    ALL of the rules below.  Falls back to a random single-day slot if no
    rule produces a match (so nothing is silently dropped).

    ┌─────────────────────────────────┬──────────┬───────────────────────────┐
    │ Case                            │ Duration │ Days                      │
    ├─────────────────────────────────┼──────────┼───────────────────────────┤
    │ Independent lab  (suffix L)     │ 2h50m    │ single day                │
    │ Regular lecture                 │ 1h15m    │ MW or TR                  │
    │ Connected lab    (suffix X/Y)   │ 1h50m    │ single day                │
    │ Intensive English               │ 6h50m    │ daily (MTWR or similar)   │
    │ Training (course type)          │ 6h50m    │ MW or TR (twice a week)   │
    │ Intro to IT — lec component     │ 1h40m    │ once a week, single day   │
    │ Intro to IT — lab component     │ 1h40m    │ once a week, single day,  │
    │                                 │          │ ≥2 days apart from lec    │
    │ prefers_evening() (supervision, │ any      │ single day; evening is    │
    │   Internship)                   │          │ preferred, not required   │
    │ Everything else                 │ any      │ single day (random)       │
    └─────────────────────────────────┴──────────┴───────────────────────────┘

    Parameters
    ----------
    section   : Section object (must expose .no, .course.type, .course.name)
    timeslots : full list of Timeslot objects available in the semester

    Returns
    -------
    list of Timeslot — may be a subset of *timeslots*; never empty (falls
    back to the full single-day list so the caller always has candidates).
    """
    ct   = section.course.type
    sno  = str(section.no)

    # ── 1. SCHEDULED-AROUND-TEACHING (supervision, Office Hours, Internship) ─
    # These belong in the evening, but that is a PREFERENCE, not a statement
    # about what FITS. This used to hard-filter the list down to slots at/after
    # 15:00 and return only those, so a preference could leave a section
    # unscheduled -- 2.0 penalty units spent to save at most 0.3 of one. The
    # shape rule (single day) is all that survives here; the evening half is
    # priced in time_soft_penalty() and applied by the ranking below.
    if prefers_evening(section):
        return [ts for ts in timeslots if ts.day in SINGLE_DAY]

    # ── 2. INTENSIVE ENGLISH — every day, 6h50m block ────────────────────────
    if _is_intensive_english(section):
        daily_sets = {"MTWRF", "MTWR"}          # accept whatever the DB calls it
        slots = [
            ts for ts in timeslots
            if ts.day in daily_sets
            and abs(_timeslot_duration(ts) - DURATION_410) <= 10   # ±10 min tolerance
        ]
        if slots:
            return slots
        # fallback: any daily multi-day slot
        return [ts for ts in timeslots if ts.day in daily_sets] or \
               [ts for ts in timeslots if ts.day in SINGLE_DAY]

    # ── 3. TRAINING — twice a week (MW / TR), 6h50m ──────────────────────────
    if ct == "Training":
        slots = [
            ts for ts in timeslots
            if ts.day in {"MW", "TR"}
            and abs(_timeslot_duration(ts) - DURATION_410) <= 10
        ]
        if slots:
            return slots
        return [ts for ts in timeslots if ts.day in {"MW", "TR"}]

    # ── 4. INTRO TO IT — lec + lab share room/timeslot; 1h40m each ───────────
    if _is_intro_it(section):
        # Both the lecture component (e.g. '04') and the lab component
        # (e.g. '04 Lab' or suffix 'L') follow the same duration rule.
        # The "≥2 days apart" pairing constraint is enforced externally
        # (see preprocessing.tag_intro_it_pairs() / intro_it_days_ok()).
        slots = [
            ts for ts in timeslots
            if ts.day in SINGLE_DAY
            and abs(_timeslot_duration(ts) - DURATION_100) <= 10
        ]
        if slots:
            return slots
        return [ts for ts in timeslots if ts.day in SINGLE_DAY]

    # ── 5. INDEPENDENT LAB (suffix L) — single day, 2h50m ───────────────────
    if _is_independent_lab(section):
        slots = [
            ts for ts in timeslots
            if ts.day in SINGLE_DAY
            and abs(_timeslot_duration(ts) - DURATION_170) <= 10
        ]
        if slots:
            return slots
        return [ts for ts in timeslots if ts.day in SINGLE_DAY]

    # ── 6. CONNECTED LAB (suffix X or Y) — single day, 1h50m ────────────────
    if _is_connected_lab(section):
        slots = [
            ts for ts in timeslots
            if ts.day in SINGLE_DAY
            and abs(_timeslot_duration(ts) - DURATION_110) <= 10
        ]
        if slots:
            return slots
        return [ts for ts in timeslots if ts.day in SINGLE_DAY]

    # ── 7. REGULAR LECTURE — MW or TR, 1h15m ─────────────────────────────────
    lecture_types = {
        "Lecture Undergraduate",
        "Lecture Graduate",
        "Lecutre / Studio Undergraduate",
        "Seminar Graduate",
    }
    if ct in lecture_types:
        slots = [
            ts for ts in timeslots
            if ts.day in {"MW", "TR"}
            and abs(_timeslot_duration(ts) - DURATION_75) <= 10
        ]
        if slots:
            return slots
        # fallback: correct days, any duration
        return [ts for ts in timeslots if ts.day in {"MW", "TR"}]

    # ── 8. LECTURE/LAB COMBINED — single day, treat as lab ───────────────────
    if ct in {"Lecture/Lab", "Laboratory"}:
        slots = [
            ts for ts in timeslots
            if ts.day in SINGLE_DAY
            and abs(_timeslot_duration(ts) - DURATION_170) <= 10
        ]
        if slots:
            return slots
        return [ts for ts in timeslots if ts.day in SINGLE_DAY]

    # ── 9. TUTORIAL (suffix T) — follow parent lecture's day preference ───────
    # Tutorials are typically 1h15m, single day.
    if _is_tutorial(section):
        slots = [
            ts for ts in timeslots
            if ts.day in SINGLE_DAY
            and abs(_timeslot_duration(ts) - DURATION_75) <= 10
        ]
        if slots:
            return slots
        return [ts for ts in timeslots if ts.day in SINGLE_DAY]

    # ── 10. CATCH-ALL FALLBACK — any single-day slot ─────────────────────────
    fallback = [ts for ts in timeslots if ts.day in SINGLE_DAY]
    return fallback if fallback else list(timeslots)


def get_valid_timeslots_for_section(section, timeslots: list) -> list:
    """Timeslots this section may use, PREFERRED ONES FIRST. Never empty.

    Same split as get_viable_rooms(): _candidate_timeslots() decides what FITS
    (duration and day pattern -- a 1h15m lecture cannot use a 6h50m block), and
    rank_timeslots() then orders what survives by preference. Nothing is dropped
    for a soft reason.

    Ranking here rather than inside each algorithm is what makes the evening
    preference apply everywhere at once: all five funnel their timeslot choice
    through this one function, and greedy, GRASP and hybrid take the first free
    slot they are handed. Genetic and PSO sample rather than scan, so for those
    two it is the penalty in the objective that does the work.
    """
    return rank_timeslots(section, _candidate_timeslots(section, timeslots))
