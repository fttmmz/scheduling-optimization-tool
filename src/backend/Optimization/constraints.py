# constraints.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for ALL scheduling constraints and preprocessing.
# Every algorithm (greedy, genetic, etc.) imports from here 
# ─────────────────────────────────────────────────────────────────────────────

import re
from collections import defaultdict

# 1. COURSE-TYPE CLASSIFICATION

NEEDS_ROOM_AND_TIME = {
    "Lecture Undergraduate",
    "Lecture Graduate",
    "Lecutre / Studio Undergraduate",  
    "Lecture/Lab",
    "Laboratory",
    "Seminar Graduate",
    "Office Hours",
}

NEEDS_ROOM_ONLY = {
    "Senior Project Supervision",
    "Project",
    "Thesis",
    "Thesis / Dissertation Master",
    "Thesis / Dissertation Doctorat",
    "Independent Study",
}

NEEDS_NOTHING = {
    "Clinical Practice",
    "Field Training",
    "Internship",
    "Qualifying Exam",
    "Comprehensive Exam",
    "Training",
}


def classify_section(section) -> str:
    """
    Return one of 'NEEDS_ROOM_AND_TIME', 'NEEDS_ROOM_ONLY', 'NEEDS_NOTHING'.
    Falls back to 'NEEDS_ROOM_AND_TIME' for unknown types so they are
    scheduled rather than silently dropped.
    """
    ct = section.course.type
    if ct in NEEDS_NOTHING:
        return "NEEDS_NOTHING"
    if ct in NEEDS_ROOM_ONLY:
        return "NEEDS_ROOM_ONLY"
    return "NEEDS_ROOM_AND_TIME"  # default / unknown


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


def _timeslot_duration(ts) -> int:
    """
    Return the duration of a timeslot in minutes.
    Assumes timeslot objects expose .start_time and .end_time as
    datetime.time (or anything with .hour / .minute).
    Falls back to 0 if attributes are missing.
    """
    try:
        start = ts.start_time.hour * 60 + ts.start_time.minute
        end   = ts.end_time.hour   * 60 + ts.end_time.minute
        return max(0, end - start)
    except AttributeError:
        return 0


def _is_later_in_day(ts, cutoff_hour: int = 15) -> bool:
    """
    Return True if the timeslot starts at or after cutoff_hour (24-h clock).
    Used to push low-priority sections (Office Hours, Project, …) later.
    Default cutoff: 15:00 (3 PM).
    """
    try:
        return ts.start_time.hour >= cutoff_hour
    except AttributeError:
        return False


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

        base_no = _base_section_no(sec.no)
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
    """Derive campus from building code (e.g. 'M3', 'W8')."""
    if not isinstance(building, str):
        return "MEDICAL"

    building = building.upper()

    if building.startswith("M"):
        try:
            num = int(building[1:])
        except ValueError:
            return "MEDICAL"
        if 1 <= num <= 6:
            return "MEN"
        if 7 <= num <= 12:
            return "MAIN"

    elif building.startswith("W"):
        try:
            num = int(building[1:])
        except ValueError:
            return "MEDICAL"
        if 1 <= num <= 6:
            return "WOMEN"
        if 7 <= num <= 12:
            return "MAIN"

    return "MEDICAL"



# 5. SECTION-LINKING PREPROCESSING  (run once, before any algorithm)



def _base_section_no(section_no: str) -> str:
    """Strip trailing letter(s) to get the base section number string."""
    return re.sub(r"[A-Za-z]+$", "", str(section_no)).strip()


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
        base_no = _base_section_no(sec.no)
        key = (sec.course.id, base_no)
        # Only index sections that are themselves "base" (no trailing letter)
        if str(sec.no) == base_no:
            base_lookup[key] = sec

    for sec in all_sections:
        sec.tutorial_parent_id = None
        sec.lab_parent_id = None

        sno = str(sec.no)
        base_no = _base_section_no(sno)
        suffix = sno[len(base_no):]  # whatever was stripped

        if not suffix:
            continue  # base section — no link needed

        parent = base_lookup.get((sec.course.id, base_no))
        parent_id = parent.id if parent else None  # None if parent missing

        if suffix.upper() == "T":
            sec.tutorial_parent_id = parent_id
        elif suffix.upper() in {"X", "Y"}:
            sec.lab_parent_id = parent_id



# 6. HARD CONSTRAINT VALIDATORS  (item-level — used inside algorithm loops)
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


def is_room_type_match(section, room) -> bool:
    """
    The required room type (derived from course type) must equal the
    actual room type.  Returns True for course types that need no room.
    """
    required = get_required_room_type(section.course.type)
    if required is None:
        return True
    return room.type == required


def is_department_match(section, room) -> bool:
    """
    Department matching rules:

    For LABS (room.type == 'lab'):
      HARD — room.dept_id must equal section.course.dept OR room.dept_id is None/''
      (a lab with no dept assignment is open to everyone; a dept-assigned
       lab is exclusive to that department)

    For CLASSROOMS (room.type == 'classroom'):
      SOFT treated as HARD here — prefer dept match, but allow undept-rooms.
      Rooms assigned to a dept are RESERVED for that dept (hard).
      Rooms with no dept can be used by anyone .

    Returns True  → assignment is allowed
    Returns False → hard violation (dept-locked room, wrong dept)
    """
    room_dept = room.dept_id  # may be None / ''
    course_dept = section.course.dept

    # Room has no department restriction → always allowed
    if not room_dept:
        return True

    # Room IS dept-restricted → must match course dept
    return room_dept == course_dept


def is_capacity_ok(section, room) -> bool:
    """Room capacity must be >= section enrolment capacity."""
    return room.capacity >= section.capacity


def is_campus_match(section, room) -> bool:
    """Section campus (derived from section number) must match room campus."""
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
    Convenience: run ALL hard constraints for a candidate (section, room, timeslot).

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

    return (
        instructor_free
        and room_free
        and is_room_type_match(section, room)
        and is_department_match(section, room)
        and is_capacity_ok(section, room)
        and is_campus_match(section, room)
    )



# 7. SOFT CONSTRAINT SCORERS  (schedule-level — used for evaluation)



def score_tutorial_links(schedule: list, sections_by_id: dict) -> int:
    """
    Returns the number of tutorial sections whose room differs from their
    parent lecture's room.  0 = perfect.
    """
    scheduled_room = {item.course_id: item.room_id for item in schedule}
    # Build map: section_id → room_id from schedule
    room_by_section: dict = {}
    for item in schedule:
        room_by_section[(item.course_id, item.section)] = item.room_id

    violations = 0
    for sec in sections_by_id.values():
        if not getattr(sec, "tutorial_parent_id", None):
            continue
        parent = sections_by_id.get(sec.tutorial_parent_id)
        if not parent:
            continue
        tutorial_room = room_by_section.get((sec.course.id, sec.no))
        parent_room = room_by_section.get((parent.course.id, parent.no))
        if tutorial_room and parent_room and tutorial_room != parent_room:
            violations += 1
    return violations



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

def get_viable_rooms(section, rooms):
    """Return rooms that satisfy the section's static constraints."""
    return [
        room
        for room in rooms
        if is_room_type_match(section, room)
        and is_department_match(section, room)
        and is_capacity_ok(section, room)
        and is_campus_match(section, room)
    ]


def get_viable_rooms_for_schedule_item(item, rooms):
    """Return rooms that satisfy a schedule item's constraints."""
    return [
        room
        for room in rooms
        if _schedule_item_room_type_match(item, room)
        and _schedule_item_capacity_ok(item, room)
    ]


def _schedule_item_room_type_match(item, room) -> bool:
    """Check if room type matches schedule item's course type."""
    required_room_type = get_required_room_type(item.course_type)
    if required_room_type is None:
        return True
    return room.type == required_room_type


def _schedule_item_capacity_ok(item, room) -> bool:
    """Check if room capacity is sufficient for schedule item."""
    return item.capacity <= room.capacity


# greedy helpers
def _build_room_lookup(rooms):
    """
    Returns a dict:
        (room_type, course_dept, section_campus) → [room, ...]

    Rooms are ordered dept-match first, then open (dept_id=None).
    Type, department, and campus are all static properties — checking them
    once here means the inner loop only sees genuinely viable candidates and
    only needs to check availability (O(1) set lookup) and capacity.
    """
    # Intermediate buckets
    dept_typed = defaultdict(list)  # (room_type, dept_id, campus) → rooms
    open_typed = defaultdict(list)  # (room_type, campus) → rooms

    for room in rooms:
        campus = get_building_campus(room.building)
        if room.dept_id:
            dept_typed[(room.type, room.dept_id, campus)].append(room)
        else:
            open_typed[(room.type, campus)].append(room)

    # Collect all (room_type, course_dept, campus) combos that will be queried
    # We can't know course_dept ahead of time, so we build on first access via cache
    return dept_typed, open_typed


def _viable_rooms(dept_typed, open_typed, room_type, course_dept, section_campus):
    """Yield rooms in priority order for a given (type, dept, campus) triple."""
    # 1. dept-assigned rooms matching this course's department
    yield from dept_typed.get((room_type, course_dept, section_campus), [])
    # 2. open rooms (no dept restriction) — fallback for classrooms, also valid for labs
    yield from open_typed.get((room_type, section_campus), [])