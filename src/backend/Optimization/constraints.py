# constraints.py
# Single source of truth for ALL scheduling constraints and preprocessing.
# Every algorithm (greedy, genetic, etc.) imports from here


import re
from collections import defaultdict
from typing import NamedTuple

# 1. COURSE-TYPE / LEVEL CLASSIFICATION
#
# What a section needs is described by two independent booleans, because the
# real (manual) schedule uses all four combinations:
#
#     Needs(room=True,  time=True)   normal lecture / undergrad lab
#     Needs(room=False, time=True)   design studio, GRADUATE lab, office hours,
#                                    training, clinical practice, seminars...
#                                    (meets at a scheduled hour, no tracked room)
#     Needs(room=True,  time=False)  room-only (rare)
#     Needs(room=False, time=False)  thesis / dissertation, doctorate exams
#                                    (no room, no timeslot)
#
# The single most important fact: the need depends on LEVEL, not just
# course_type. 'Laboratory' at Undergraduate level needs a real lab room, but
# 'Laboratory' at Master/Doctorate level needs neither (research lab). So
# classification is keyed by (level, course_type).


class Needs(NamedTuple):
    """What a section needs from the scheduler."""
    room: bool
    time: bool


# ── Mined classification table ────────────────────────────────────────────────
# Derived directly from the real manual schedule (schedule_id=10) on 2026-08-09:
# for each (level, course_type) with n>=5 placements, a category is treated as
# "no room" when the manual schedule gave it a room <20% of the time, and
# "no timeslot" when it landed in the placeholder slot (ts=1) >80% of the time.
# Only the rows that DIVERGE from the (room=True, time=True) default are listed;
# everything else falls through to that safe default so nothing is dropped.
# Regenerate with scratchpad/open_questions2.py-style mining if the dataset changes.
MINED_NEEDS: dict[tuple, Needs] = {
    ("Doctorate", "Comprehensive Exam"):             Needs(False, False),  # n=12 noroom=100% ts1=92%
    ("Doctorate", "Proposal Doctorate"):             Needs(False, False),  # n=6  noroom=100% ts1=100%
    ("Doctorate", "Thesis"):                         Needs(False, False),  # n=5  noroom=100% ts1=80%
    ("Doctorate", "Thesis / Dissertation Doctorat"): Needs(False, False),  # n=75 noroom=100% ts1=100%
    ("Doctorate", "Thesis / Dissertation Master"):   Needs(False, False),  # n=5  noroom=100% ts1=100%
    ("Master",    "Clinical Practice"):              Needs(False, True),   # n=59 noroom=95%  ts1=24%
    ("Master",    "Laboratory"):                     Needs(False, True),   # n=9  noroom=100% ts1=0%
    ("Master",    "Office Hours"):                   Needs(False, True),   # n=7  noroom=100% ts1=0%
    ("Master",    "Project"):                        Needs(False, True),   # n=9  noroom=89%  ts1=22%
    ("Master",    "Seminar Graduate"):               Needs(False, True),   # n=37 noroom=81%  ts1=11%
    ("Master",    "Thesis"):                         Needs(False, True),   # n=7  noroom=100% ts1=71%
    ("Master",    "Thesis / Dissertation Master"):   Needs(False, False),  # n=115 noroom=100% ts1=97%
    ("Undergraduate", "Clinical Practice"):          Needs(False, True),   # n=13 noroom=100% ts1=23%
    ("Undergraduate", "Internship"):                 Needs(False, True),   # n=17 noroom=82%  ts1=76%
    ("Undergraduate", "Lecutre / Studio Undergraduate"): Needs(False, True),  # n=77 noroom=100% ts1=0%
    ("Undergraduate", "Office Hours"):               Needs(False, True),   # n=44 noroom=98%  ts1=50%
    ("Undergraduate", "Training"):                   Needs(False, True),   # n=51 noroom=94%  ts1=35%
}

# Safe default for any (level, course_type) not in the mined table: schedule it
# fully (room + time) so an unrecognised section is never silently dropped.
DEFAULT_NEEDS = Needs(True, True)


def needs_for(course_type, level) -> Needs:
    """Core classifier: what does a (course_type, level) section need?

    Looks up the mined (level, course_type) table; falls back to the safe
    room+time default. This is the single place both the section path
    (section_needs) and the item path (hybrid._item_requirement) share.
    """
    return MINED_NEEDS.get((level, course_type), DEFAULT_NEEDS)


def section_needs(section) -> Needs:
    """Canonical classifier for a Section object -> Needs(room, time)."""
    level = getattr(section.course, "level", None)
    return needs_for(section.course.type, level)


# ── Legacy string-enum compatibility ──────────────────────────────────────────
# Older callers (greedy.py, hybrid.py) branch on these four string values.
# classify_section maps the two-boolean Needs onto them, adding NEEDS_TIME_ONLY
# for the "meets at a scheduled hour but needs no room" case the old 3-value
# scheme could not express.
NEEDS_ROOM_AND_TIME = "NEEDS_ROOM_AND_TIME"
NEEDS_TIME_ONLY     = "NEEDS_TIME_ONLY"
NEEDS_ROOM_ONLY     = "NEEDS_ROOM_ONLY"
NEEDS_NOTHING       = "NEEDS_NOTHING"


def needs_to_label(needs: Needs) -> str:
    """Map Needs(room, time) onto the legacy string-enum label."""
    if needs.room and needs.time:
        return NEEDS_ROOM_AND_TIME
    if not needs.room and needs.time:
        return NEEDS_TIME_ONLY
    if needs.room and not needs.time:
        return NEEDS_ROOM_ONLY
    return NEEDS_NOTHING


def classify_section(section) -> str:
    """Backward-compatible label for a Section: one of NEEDS_ROOM_AND_TIME,
    NEEDS_TIME_ONLY, NEEDS_ROOM_ONLY, NEEDS_NOTHING. Prefer section_needs()
    in new code."""
    return needs_to_label(section_needs(section))


# 2. COURSE-TYPE → ROOM-TYPE MAPPING

COURSE_TYPE_TO_ROOM_TYPE = {
    "Lecture Undergraduate": "classroom",
    "Lecture Graduate": "classroom",
    "Lecutre / Studio Undergraduate": "classroom",
    "Seminar Graduate": "classroom",
    "Office Hours": "classroom",
    "Senior Project Supervision": "classroom",
    "Project": "classroom",
    "Thesis": "classroom",
    "Thesis / Dissertation Master": "classroom",
    "Thesis / Dissertation Doctorat": "classroom",
    "Independent Study": "classroom",
    "Lecture/Lab": "lab",
    "Laboratory": "lab",
}


def get_required_room_type(course_type: str) -> str | None:
    """Return the room type string required by this course type, or None."""
    return COURSE_TYPE_TO_ROOM_TYPE.get(course_type)


# 3. TIMESLOT HELPERS

MULTI_DAY = {"MW", "TR", "MTWR"}
SINGLE_DAY = {"M", "T", "W", "R"}

# Duration constants (in minutes) — used to filter timeslots by expected length
DURATION_75  = 75    # 1h15m  — standard lecture
DURATION_100 = 100   # 1h40m  — Intro to IT combined lec/lab
DURATION_110 = 110   # 1h50m  — connected lab (X / Y suffix)
DURATION_170 = 170   # 2h50m  — independent lab (L suffix) or Training
DURATION_410 = 410   # 6h50m  — intensive English / Training (daily block)

# Special course name fragment used for Intro-to-IT detection
INTRO_IT_NAME_FRAGMENT = "introduction to it"

# Day ordering used to enforce the "one day gap" rule for Intro-to-IT pairs
DAY_ORDER = {"M": 0, "T": 1, "W": 2, "R": 3, "F": 4}

# Scheduling-priority tags written onto sections by tag_scheduling_priority()
PRIORITY_LATE  = "LATE"   # schedule after main courses, later in the day
PRIORITY_MAIN  = "MAIN"   # normal scheduling window


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


def _is_later_in_day(ts, cutoff_hour: int = 15) -> bool:
    """
    Return True if the timeslot starts at or after cutoff_hour (24-h clock).
    Used to push low-priority sections (Office Hours, Project, …) later.
    Default cutoff: 15:00 (3 PM). Same start_time/.start mismatch as
    _timeslot_duration() above -- fixed the same way.
    """
    start = _parse_time_to_minutes(getattr(ts, "start", None))
    if start is None:
        return False
    return start >= cutoff_hour * 60


def _days_apart(day_a: str, day_b: str) -> int:
    """
    Return the absolute calendar-day distance between two single-day codes.
    Returns 99 if either day is unknown.
    """
    a = DAY_ORDER.get(day_a, 99)
    b = DAY_ORDER.get(day_b, 99)
    return abs(a - b)


# timeslot functions

def get_valid_timeslots(section, timeslots: list) -> list:

    return get_valid_timeslots_for_section(section, timeslots)


def get_valid_timeslots_for_section(section, timeslots: list) -> list:
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
    │ Office Hours / Project /        │ any      │ single day, ≥ 15:00       │
    │   Internship / …low-priority    │          │                           │
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

    # ── 1. LOW-PRIORITY SECTIONS (Office Hours, Project, Internship, …) ──────
    # These should be scheduled last and only in afternoon/evening slots.
    low_priority_types = {
        "Office Hours", "Project", "Internship",
        "Senior Project Supervision", "Independent Study",
        "Thesis", "Thesis / Dissertation Master",
        "Thesis / Dissertation Doctorat",
    }
    if ct in low_priority_types:
        late_slots = [
            ts for ts in timeslots
            if ts.day in SINGLE_DAY and _is_later_in_day(ts)
        ]
        return late_slots if late_slots else [
            ts for ts in timeslots if ts.day in SINGLE_DAY
        ]

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
        # (see pair_intro_it_sections()), but we expose a helper below.
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


# ─────────────────────────────────────────────────────────────────────────────
# INTRO-TO-IT PAIRING HELPER
# Call once before scheduling to mark which Intro-to-IT sections are
# lecture vs. lab, and record their pairing so the algorithm can enforce
# the "≥2 calendar days apart" rule.
# ─────────────────────────────────────────────────────────────────────────────

def tag_intro_it_pairs(all_sections: list) -> None:
    """
    For every Intro-to-IT course, locate the base lecture section (e.g. '04')
    and its matching lab section (e.g. '04 Lab' or suffix 'L'), then annotate
    both in-place:

      section.intro_it_role         — 'lecture' | 'lab' | None
      section.intro_it_pair_id      — Section.id of the counterpart, or None

    The scheduling algorithm must then ensure the two paired sections are
    placed on days that are at least 2 apart (e.g. M and W are fine; M and T
    are not, because |0-1| = 1 < 2).

    Helper:  intro_it_days_ok(day_a, day_b) → bool
    """
    # Group Intro-to-IT sections by course_id and base number
    groups: dict[tuple, dict] = defaultdict(lambda: {"lecture": None, "lab": None})

    for sec in all_sections:
        if not _is_intro_it(sec):
            sec.intro_it_role    = None
            sec.intro_it_pair_id = None
            continue

        base_no = _base_section_no(str(sec.no))
        key     = (sec.course.id, base_no)
        suffix  = _section_suffix(sec.no)

        if suffix == "L" or "lab" in str(sec.no).lower():
            groups[key]["lab"] = sec
        else:
            groups[key]["lecture"] = sec

    for (course_id, base_no), pair in groups.items():
        lec = pair["lecture"]
        lab = pair["lab"]

        if lec:
            lec.intro_it_role    = "lecture"
            lec.intro_it_pair_id = lab.id if lab else None
        if lab:
            lab.intro_it_role    = "lab"
            lab.intro_it_pair_id = lec.id if lec else None


def intro_it_days_ok(day_a: str, day_b: str) -> bool:
    """
    Return True if the two single-day codes are at least 2 calendar days
    apart — enforcing the Intro-to-IT lec/lab separation rule.
    """
    return _days_apart(day_a, day_b) >= 2


# 4. CAMPUS HELPERS
def get_section_campus(section_no) -> str:
    """
    Derive campus from section number.
    Accepts int or string (strips any trailing letters first).
    """
    if isinstance(section_no, str):
        # strip trailing letters e.g. "02T", "02X", "02Y" → "02"
        section_no = re.sub(r"[A-Za-z]+$", "", section_no).strip()

    try:
        n = int(section_no)
    except (TypeError, ValueError):
        return "MAIN"

    if 1 <= n <= 29:
        return "MEN"
    if 30 <= n <= 59:
        return "WOMEN"
    return "MAIN"


def get_building_campus(building) -> str:
    """
    Derive campus from building code (e.g. 'M3', 'W8').

    Annex buildings (e.g. 'M3A', 'W7A') follow the same numbering as their
    base building and belong to the same campus, so the trailing letter
    suffix is stripped before parsing -- without this, int("3A") raised
    ValueError and silently misrouted every annex building into the
    MEDICAL fallback below, even though they're regular MEN/WOMEN/MAIN
    buildings.
    """
    if not isinstance(building, str):
        return "MEDICAL"

    building = building.upper()
    numeric_part = re.sub(r"[A-Za-z]+$", "", building[1:])

    if building.startswith("M"):
        try:
            num = int(numeric_part)
        except ValueError:
            return "MEDICAL"
        if 1 <= num <= 6:
            return "MEN"
        if 7 <= num <= 12:
            return "MAIN"

    elif building.startswith("W"):
        try:
            num = int(numeric_part)
        except ValueError:
            return "MEDICAL"
        if 1 <= num <= 6:
            return "WOMEN"
        if 7 <= num <= 12:
            return "MAIN"

    # Building numbers above 12 (e.g. 'M23', 'M31') are a genuine separate
    # medical campus at this university, confirmed against real data --
    # this is an intentional 4th campus identity, not an accidental catch-all.
    return "MEDICAL"



# 5. SECTION-LINKING PREPROCESSING  (run once, before any algorithm)

def _base_section_no(section_no: str) -> int:
    """Strip trailing letter(s) to get the base section number as an integer."""
    base_str = re.sub(r"[A-Za-z]+$", "", str(section_no)).strip()
    try:
        return int(base_str) if base_str else 0
    except ValueError:
        return 0


def tag_section_links(all_sections: list) -> None:
    """
    Annotate every Section object in-place with two optional attributes:

      section.tutorial_parent_id  — set on tutorial sections (e.g. '02T')
                                    points to the Section.id of the parent
                                    lecture with the same course + base number

      section.lab_parent_id       — set on lab sub-sections (e.g. '02X','02Y')
                                    points to the Section.id of the parent
                                    lecture with the same course + base number

    All other sections get both attributes set to None.

    SOFT constraints only — algorithms should TRY to honour these links
    but are not required to.
    """
    # Build lookup: (course_id, base_section_no) → Section
    base_lookup: dict[tuple, object] = {}
    for sec in all_sections:
        sno = str(sec.no)
        base_no = _base_section_no(sno)
        key = (sec.course.id, base_no)
        # Extract base string to check if section has no suffix
        base_str = re.sub(r"[A-Za-z]+$", "", sno).strip()
        # Only index sections that are themselves "base" (no trailing letter)
        if sno.strip() == base_str:
            base_lookup[key] = sec

    for sec in all_sections:
        sec.tutorial_parent_id = None
        sec.lab_parent_id = None

        sno = str(sec.no)
        base_no = _base_section_no(sno)
        base_str = re.sub(r"[A-Za-z]+$", "", sno).strip()
        suffix = sno[len(base_str):]  # whatever was stripped

        if not suffix:
            continue  # base section — no link needed

        parent = base_lookup.get((sec.course.id, base_no))
        parent_id = parent.id if parent else None  # None if parent missing

        if suffix.upper() == "T":
            sec.tutorial_parent_id = parent_id
        elif suffix.upper() in {"X", "Y"}:
            sec.lab_parent_id = parent_id

# 6. CONSTRAINT TIERS
#
# THE HARD/SOFT SPLIT (2026-08-13). Validated against the real manual schedule
# (schedule_id=10, 4698 sections, 0 unscheduled): that schedule is in active use,
# yet it breaks the rules this module used to enforce as HARD —
#
#     campus 32%   department 31%   duration 27%   room_type 13%   capacity 11%
#
# Treating those as hard meant forbidding ~30% of the placements the university
# actually makes, which is why the algorithms hit a ~502-unscheduled ceiling: not
# a facility shortage and not algorithm weakness, but a model that outlawed the
# real answer. So the tiers are now:
#
#   HARD  — physical impossibility only: a room or an instructor in two places at
#           the same time. Never violated; still gates every placement.
#   SOFT  — campus / department / room_type / capacity. Strongly preferred and
#           priced into the objective, but NEVER a reason to leave a section
#           unscheduled. An unscheduled class is worse than an imperfect room.
#
# Consequence for callers: get_viable_rooms() no longer filters, it RANKS. It
# returns the cheapest rooms first and only drops candidates to bound the list
# size, never because a soft rule was broken.

# ── Soft-constraint weights ───────────────────────────────────────────────────
# Penalty units, all relative to each other. Rationale (revisit in Phase 2 once
# the fair evaluator separates our own measurement bugs from genuine manual
# suboptimality — several of these rates are inflated by known data defects):
#
#   room_type  — a lab course in a plain classroom is a real teaching failure,
#                so it is the most expensive. Held below "impossible" because
#                room.type is a `"lab" in description` substring flatten that
#                misfiles Design Studios as classrooms.
#   campus     — MEN/WOMEN campus separation is institutionally real, so it is
#                priced high. The 32% manual "violation" rate is largely our
#                crude section-number→campus proxy being wrong, so it must not
#                be cheap just because the proxy is noisy.
#   capacity   — graded, not binary: 5 seats short is not 50 short. The manual
#                schedule overbooks deliberately (planned enrolment exceeds the
#                room), so a small overflow is nearly free.
#   department — weakest. Dept-locked rooms are a courtesy, and 142/375 rooms
#                have no dept_id at all.
SOFT_WEIGHT_ROOM_TYPE = 4.0
SOFT_WEIGHT_CAMPUS = 3.0
SOFT_WEIGHT_DEPARTMENT = 1.0
SOFT_WEIGHT_CAPACITY = 2.0          # flat cost of overflowing at all
SOFT_WEIGHT_CAPACITY_OVERFLOW = 4.0  # extra, scaled by how badly it overflows

# How many ranked rooms get_viable_rooms() returns when soft rules force it to
# choose. Every zero-penalty room is always returned; this only bounds the
# imperfect tail, so downstream random sampling (hybrid's _sample_pairs, GRASP's
# construction sample) stays concentrated on good rooms instead of being diluted
# by hundreds of bad ones.
DEFAULT_ROOM_CANDIDATES = 25


def _soft_profile(obj):
    """Extract (course_type, dept, capacity, section_no) from either a Section
    or a ScheduleItem, so one set of soft-penalty rules serves both paths.

    The two used to drift: get_viable_rooms() scored campus and department while
    get_viable_rooms_for_schedule_item() silently ignored both, so a section
    could be rescued into a room the constructor would never have chosen.
    """
    course = getattr(obj, "course", None)
    if course is not None:                      # Section
        return course.type, course.dept, obj.capacity, obj.no
    return obj.course_type, obj.course_dept, obj.capacity, obj.section  # ScheduleItem


def capacity_penalty(needed, available) -> float:
    """Graded capacity cost: free when the room fits, then a flat cost plus a
    term proportional to the overflow fraction (capped at 1.0 so a wildly
    undersized room is expensive but still finite/comparable)."""
    if needed is None or available is None:
        return 0.0
    if available <= 0:
        # Capacity 0 means "unknown", not "a room with no seats" -- it is a data
        # gap. Charging the full overflow cost keeps such rooms usable as a last
        # resort while stopping them from scoring as a perfect fit and
        # outranking real rooms (a zero-capacity room otherwise sorts first).
        return SOFT_WEIGHT_CAPACITY + SOFT_WEIGHT_CAPACITY_OVERFLOW
    if available >= needed:
        return 0.0
    overflow_ratio = min(1.0, (needed - available) / available)
    return SOFT_WEIGHT_CAPACITY + SOFT_WEIGHT_CAPACITY_OVERFLOW * overflow_ratio


def room_soft_penalty_parts(obj, room) -> dict:
    """Per-rule soft cost of putting *obj* (Section or ScheduleItem) in *room*.
    Returned split out so the evaluator can report which rule was bent."""
    course_type, course_dept, capacity, section_no = _soft_profile(obj)

    required_type = get_required_room_type(course_type)
    type_cost = (
        SOFT_WEIGHT_ROOM_TYPE
        if required_type is not None and room.type != required_type
        else 0.0
    )

    # An unrestricted room (no dept_id) is open to everyone — no cost.
    dept_cost = (
        SOFT_WEIGHT_DEPARTMENT
        if room.dept_id and room.dept_id != course_dept
        else 0.0
    )

    campus_cost = (
        SOFT_WEIGHT_CAMPUS
        if get_section_campus(section_no) != get_building_campus(room.building)
        else 0.0
    )

    return {
        "room_type": type_cost,
        "department": dept_cost,
        "campus": campus_cost,
        "capacity": capacity_penalty(capacity, room.capacity),
    }


def room_soft_penalty(obj, room) -> float:
    """Total soft cost of this room for this section/item. 0.0 == perfect fit."""
    return sum(room_soft_penalty_parts(obj, room).values())


# ── HARD CONSTRAINT VALIDATORS  (item-level — used inside algorithm loops) ────
# Each function returns True / False (violation).


def is_instructor_free(schedule: list, instructor_id, timeslot_id) -> bool:

    """No instructor may teach two sections at the same timeslot."""

    if instructor_id is None or timeslot_id is None:
        return True
    for item in schedule:
        if item.instructor_id == instructor_id and item.timeslot_id == timeslot_id:
            return False
    return True


def is_room_free(schedule: list, room_id, timeslot_id) -> bool:

    """No room may host two sections at the same timeslot."""

    if room_id is None or timeslot_id is None:
        return True
    for item in schedule:
        if item.room_id == room_id and item.timeslot_id == timeslot_id:
            return False
    return True


def is_instructor_free_map(occupied_instructors: set, instructor_id, timeslot_id) -> bool:
    """instructor availability check using a prebuilt set."""
    if instructor_id is None or timeslot_id is None:
        return True
    return (instructor_id, timeslot_id) not in occupied_instructors


def is_room_free_map(occupied_rooms: set, room_id, timeslot_id) -> bool:
    """room availability check using a prebuilt set."""
    if room_id is None or timeslot_id is None:
        return True
    return (room_id, timeslot_id) not in occupied_rooms


# ── SOFT predicates ───────────────────────────────────────────────────────────
# These are the boolean form of the four soft rules priced by
# room_soft_penalty(). They are NOT gates: nothing may refuse a placement on
# their say-so. They survive because the evaluator and the UI report violation
# COUNTS per rule, which needs a crisp yes/no per rule.

def is_room_type_match(section, room) -> bool:
    """SOFT: does the room's type match the one the course type implies?
    True for course types that imply no particular room."""
    required = get_required_room_type(section.course.type)
    if required is None:
        return True
    return room.type == required


def is_department_match(section, room) -> bool:
    """SOFT: is this room available to the course's department?

    A room with no dept_id is unrestricted and always matches; a dept-assigned
    room 'belongs' to that department. Formerly enforced as hard (labs
    exclusive, classrooms 'soft treated as hard'), which locked sections out of
    rooms the real schedule uses freely — the manual schedule breaks this on 31%
    of placements, and 142/375 rooms have no dept_id at all.
    """
    room_dept = room.dept_id  # may be None / ''
    if not room_dept:
        return True
    return room_dept == section.course.dept


def is_capacity_ok(section, room) -> bool:
    """SOFT: does the room seat the section's planned enrolment?

    Overbooking is deliberate in the real data (sec_capacity is planned
    enrolment, and the manual schedule overflows the room 11% of the time), so
    the graded capacity_penalty() is the better signal — this stays for counting.
    """
    return room.capacity >= section.capacity


def is_campus_match(section, room) -> bool:
    """SOFT: does the room's campus match the one implied by the section number?"""
    section_campus = get_section_campus(section.no)
    building_campus = get_building_campus(room.building)
    return section_campus == building_campus


def passes_hard_constraints(
    section,
    room,
    timeslot,
    schedule: list = None,
    occupied_instructors: set = None,
    occupied_rooms: set = None,
) -> bool:
    """
    Run the HARD constraints for a candidate (section, room, timeslot):
    no double-booked room, no double-booked instructor. That is all — see the
    tier note at the top of section 6.

    This used to also require room_type/department/capacity/campus to match,
    which made a merely-imperfect room indistinguishable from a physically
    impossible one and left ~30% of the real schedule's placements unreachable.
    Those four are now priced by room_soft_penalty() instead, so a section takes
    an imperfect room rather than going unscheduled.

    If occupancy maps are provided, use those for instructor/room checks
    instead of scanning the partial schedule.
    """
    if occupied_instructors is not None:
        instructor_free = is_instructor_free_map(
            occupied_instructors,
            section.instructor_id,
            timeslot.id if timeslot else None,
        )
    else:
        instructor_free = is_instructor_free(
            schedule, section.instructor_id, timeslot.id if timeslot else None
        )

    if occupied_rooms is not None:
        room_free = is_room_free_map(
            occupied_rooms,
            room.id if room else None,
            timeslot.id if timeslot else None,
        )
    else:
        room_free = is_room_free(
            schedule, room.id if room else None, timeslot.id if timeslot else None
        )

    return instructor_free and room_free

#  FULL-SCHEDULE VALIDATORS  (used for final verification / testing)

def check_instructors(schedule: list, data=None) -> bool:
    seen = set()
    for item in schedule:
        if item.instructor_id is None:
            continue
        key = (item.instructor_id, item.timeslot_id)
        if key in seen:
            return False
        seen.add(key)
    return True


def check_room(schedule: list, data=None) -> bool:
    seen = set()
    for item in schedule:
        if item.room_id is None or item.timeslot_id is None:
            continue
        key = (item.room_id, item.timeslot_id)
        if key in seen:
            return False
        seen.add(key)
    return True


def check_capacity(schedule: list, data=None) -> bool:
    rooms = {room.id: room for room in data["rooms"]}
    for item in schedule:
        if item.room_id is None:
            continue
        room = rooms.get(item.room_id)
        if room and room.capacity < item.capacity:
            return False
    return True


def check_room_type(schedule: list, data=None) -> bool:
    rooms = {room.id: room for room in data["rooms"]}
    for item in schedule:
        if item.room_id is None:
            continue
        room = rooms.get(item.room_id)
        if not room:
            continue
        required = get_required_room_type(item.course_type)
        if required and room.type != required:
            return False
    return True


def check_department(schedule: list, data=None) -> bool:
    rooms = {room.id: room for room in data["rooms"]}
    for item in schedule:
        if item.room_id is None:
            continue
        room = rooms.get(item.room_id)
        if not room:
            continue
        if room.dept_id and room.dept_id != item.course_dept:
            return False
    return True


def check_campus(schedule: list, data=None) -> bool:
    rooms = {room.id: room for room in data["rooms"]}
    for item in schedule:
        if item.room_id is None:
            continue
        room = rooms.get(item.room_id)
        if not room:
            continue
        if get_section_campus(item.section) != get_building_campus(room.building):
            return False
    return True


def check_all(schedule: list, data: dict) -> bool:
    """Run every hard constraint validator. Returns True only if all pass."""
    checks = [
        check_instructors,
        check_room,
        check_capacity,
        check_room_type,
        check_department,
        check_campus,
    ]
    return all(fn(schedule, data) for fn in checks)


# 8. HELPER FUNCTIONS FOR GENETIC ALGORITHM

def rank_rooms(obj, rooms, limit=DEFAULT_ROOM_CANDIDATES):
    """
    Rank every room for *obj* (a Section or a ScheduleItem) cheapest-first by
    soft penalty, and return the best ones. This REPLACES the old filter-based
    viable-room logic — see the tier note in section 6.

    Two properties make the switch safe:

      * Every zero-penalty (perfect) room is always returned, and the sort is
        stable, so for any section that already had viable rooms the old viable
        set is exactly the prefix of this list, in its original order. Nothing
        that used to be chosen becomes unreachable or worse-ranked.
      * *limit* only ever truncates the imperfect tail. It exists so downstream
        random sampling stays concentrated on good rooms rather than being
        diluted by hundreds of bad ones; pass limit=None to get all of them.

    So a section whose department/campus/type/capacity rules previously left it
    with NO viable room — the direct cause of the unscheduled ceiling — now gets
    the least-bad rooms instead of an empty list.
    """
    if not rooms:
        return []

    scored = [(room_soft_penalty(obj, room), room) for room in rooms]
    scored.sort(key=lambda pair: pair[0])  # stable: ties keep input order

    if limit is None:
        return [room for _, room in scored]

    perfect = sum(1 for penalty, _ in scored if penalty == 0.0)
    keep = max(perfect, limit)
    return [room for _, room in scored[:keep]]


def get_viable_rooms(section, rooms, limit=DEFAULT_ROOM_CANDIDATES):
    """Rooms for a Section, best-fit first. Never empty unless *rooms* is."""
    return rank_rooms(section, rooms, limit=limit)


def get_viable_rooms_for_schedule_item(item, rooms, limit=DEFAULT_ROOM_CANDIDATES):
    """Rooms for a ScheduleItem, best-fit first.

    Now scores campus and department too. The old version checked only room type
    and capacity, so the repair/rescue paths that use it (genetic, GRASP) could
    place a section in a room the constructor would have rejected outright.
    """
    return rank_rooms(item, rooms, limit=limit)


# greedy helpers
def _build_room_lookup(rooms):
    """
    Returns (dept_typed, open_typed, all_rooms).

    dept_typed: (room_type, dept_id, campus) → rooms
    open_typed: (room_type, campus)          → rooms
    all_rooms:  the full list, used for the soft fallback tail in _viable_rooms.

    Type, department, and campus are static properties, so bucketing them once
    here means the inner scheduling loop only checks availability (O(1)).
    """
    dept_typed = defaultdict(list)
    open_typed = defaultdict(list)

    for room in rooms:
        campus = get_building_campus(room.building)
        if room.dept_id:
            dept_typed[(room.type, room.dept_id, campus)].append(room)
        else:
            open_typed[(room.type, campus)].append(room)

    return dept_typed, open_typed, list(rooms)


def _greedy_room_penalty(room, room_type, course_dept, section_campus) -> float:
    """Soft penalty for greedy's lookup path, which knows only the
    (type, dept, campus) triple rather than the Section object. Capacity is
    scored separately by the caller, which knows the section's enrolment."""
    penalty = 0.0
    if room_type is not None and room.type != room_type:
        penalty += SOFT_WEIGHT_ROOM_TYPE
    if room.dept_id and room.dept_id != course_dept:
        penalty += SOFT_WEIGHT_DEPARTMENT
    if get_building_campus(room.building) != section_campus:
        penalty += SOFT_WEIGHT_CAMPUS
    return penalty


def _viable_rooms(
    dept_typed,
    open_typed,
    room_type,
    course_dept,
    section_campus,
    all_rooms=None,
    limit=DEFAULT_ROOM_CANDIDATES,
):
    """Yield rooms best-fit first for a given (type, dept, campus) triple.

    Tiers 1 and 2 are the exact matches this function used to return, in the
    same order. Tier 3 is new: the least-bad remaining rooms, ranked. Without
    it a section whose triple had no exact match simply went unscheduled, which
    is the greedy-side half of the unscheduled ceiling.
    """
    exact_dept = dept_typed.get((room_type, course_dept, section_campus), [])
    exact_open = open_typed.get((room_type, section_campus), [])

    # 1. dept-assigned rooms matching this course's department
    yield from exact_dept
    # 2. open rooms (no dept restriction)
    yield from exact_open

    if not all_rooms:
        return

    # 3. soft fallback — everything else, cheapest first, so an imperfect room
    #    beats no room at all.
    already = {id(room) for room in exact_dept}
    already.update(id(room) for room in exact_open)

    rest = [room for room in all_rooms if id(room) not in already]
    rest.sort(
        key=lambda room: _greedy_room_penalty(
            room, room_type, course_dept, section_campus
        )
    )
    yield from rest[:limit] if limit is not None else rest