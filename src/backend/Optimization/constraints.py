# constraints.py
# Single source of truth for ALL scheduling constraints and preprocessing.
# Every algorithm (greedy, genetic, etc.) imports from here


import re
from collections import Counter, defaultdict
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


# 1b. COMBINED / CROSS-LISTED COURSE GROUPS
#
# Some courses are deliberately taught as ONE class under several course codes
# (cross-listing, or a lecture and its lab registered separately). They share a
# room and a timeslot on purpose, so counting them as a double-booking is wrong:
# they are the reason the real manual schedule appears to contain 30 room
# collisions despite being in active use.
#
# There is NO flag for this in the data -- it has to be inferred. Mined from
# schedule_id=10 on 2026-08-13: distinct courses co-located in the same room at
# the same timeslot, and corroborated by a shared instructor or a near-identical
# course name. Co-location ALONE is deliberately not enough, because that rule
# would be circular -- it would excuse any double-booking our own algorithms
# produced, and quietly destroy the hard tier.
#
# One entry (405311/405312) is NOT co-located: the pair sits in two different
# rooms at the same 75-minute slot with the same instructor and near-identical
# names, i.e. one class whose room record disagrees with itself. The already
# co-located Ergonomics pair (405341/405342) shows exactly the same split on
# other days, which is what makes this reading safe rather than a guess.
#
# Deliberately NOT included: thesis, internship, senior-project and clinical
# "Advanced EP" placements. One supervisor legitimately covers many of those at
# one nominal time, so they look combined but are not one class -- they are
# supervision, and they are handled by classification (NEEDS_TIME_ONLY /
# NEEDS_NOTHING) plus the zero-span placeholder rule below, not by grouping.
COMBINED_COURSE_GROUPS = [
    frozenset((401314, 401315)),    # Reinforced Concrete Design 1 (x2)
    frozenset((401701, 401706)),    # Directed Studies in CE (x2)
    frozenset((405103, 405105)),    # Introduction to IE&EM / Introduction to IEEM
    frozenset((405311, 405312)),    # Operations Research-1 / Operations Research I

    frozenset((405341, 405342)),    # Erg. Work & Process Improvt. / Fundamentals of Ergonomics
    frozenset((500105, 500150)),    # Biology Laboratory / Biology
    frozenset((602531, 602560)),    # Public Int. Law (E)(In-depth) / Public International Law (E)
    frozenset((1102113, 1102230)),  # Pathophysiology 1 / Pathophysiology
    frozenset((1103221, 1103599)),  # Cosmetics & Para-pharm. / Cosmetics & Parapharmaceutical
    frozenset((1103411, 1103470)),  # Pharmaceutics II A / Biopharmaceutics & Pharmacokin.
    frozenset((1103511, 1103590)),  # Pharmaceutics3 / Drug Delivery Systems
    frozenset((1104359, 1104452)),  # Self Care & OTC Therapy / Principle of OTC Therapy
    frozenset((1104453, 1104479)),  # Law and Ethics / Pharmacy Ethics and Law
    frozenset((1426106, 1426155)),  # General Chemistry Lab for HS / General Chemistry for HS
    frozenset((1426216, 1426217)),  # Organic Chemistry Lab for HS / Organic Chemistry for HS
    frozenset((1502230, 1502231, 1502232)),  # Micro. & Assembly Language (+ Lab)
    frozenset((1502334, 1502336)),  # Embedded Systems Design / Microcontroller Based Design
]

# Co-located course pairs that could NOT be corroborated and are therefore still
# counted as real conflicts. Recorded for transparency, not used by any rule.
# Most are same-department graduate clusters whose instructor field is simply
# empty, so they are probably combined too -- but "probably" is not evidence,
# and a loose rule here would let our own algorithms hide double-bookings.
# Resolving these needs a source outside the schedule table (a catalog or
# registrar export), so the oracle keeps ~11 unexplained collisions until then.
UNCORROBORATED_COLOCATIONS = [
    frozenset((306620, 306622)),    # Leadership and Org. Behavior / International Business
    frozenset((203221, 203330)),    # History of the Umayyad State / Islamic Civilization(2)
    frozenset((402202, 402240)),    # Circuit Analysis I / Signals and Systems
    frozenset((404513, 404514, 404516)),  # Cultural Heritage graduate cluster
    frozenset((405102, 405262)),    # Engineering Graphics / Database Mang. & Ind. Inf. Systems
    frozenset((807511, 807513, 807515)),  # Digital Media graduate cluster
    frozenset((1502461, 1502463)),  # S. T. in Cyber Security / Special Topics in SCA
]


def _build_combined_index():
    """course_id -> group key, for the whitelisted combined groups."""
    index = {}
    for group in COMBINED_COURSE_GROUPS:
        key = min(group)  # stable, readable representative
        for course_id in group:
            index[course_id] = key
    return index


COMBINED_COURSE_INDEX = _build_combined_index()


def combined_group_key(course_id):
    """Return the shared key for a cross-listed course, else None.

    Two sections whose courses return the same non-None key are the same class
    taught once, so they may share a room and a timeslot without conflict.
    """
    return COMBINED_COURSE_INDEX.get(course_id)


def occupancy_key(course_id, section_no):
    """Identity of the *class* occupying a room/instructor slot.

    Cross-listed courses collapse onto one key so their shared booking counts
    once; everything else stays distinct per (course, section).
    """
    group = COMBINED_COURSE_INDEX.get(course_id)
    if group is not None:
        return ("combined", group)
    return ("section", course_id, str(section_no))


# 1c. SUPERVISION vs TEACHING
#
# The instructor column mixes two unrelated relationships:
#
#   TEACHING     -- this person stands in this room at this hour. Two of these
#                   at once is physically impossible.
#   SUPERVISION  -- this person is nominally attached to a student's thesis or
#                   project. Twenty-two supervisors on one senior project are
#                   not twenty-two people in a room; the row is a ROSTER ENTRY.
#
# Counting supervision as teaching invents double-bookings nobody experiences
# and blocks supervisors out of the hours they actually teach. 56% of rows have
# no instructor at all, and of those that do, the widest rosters are all
# supervision (see HANDOVER.md 4.3/4.4 and tests/test_data_invariants.py).
#
# Detection deliberately requires CORROBORATION, the same principle as
# COMBINED_COURSE_GROUPS above -- a single signal is not enough:
#
#   * course_type is the primary signal, and is right for all but a handful of
#     courses.
#   * course NAME alone is NOT sufficient. 'Project Management',
#     'Engineering Project Management' and 'Research Methods' are ordinary
#     taught courses whose names contain supervision keywords. Mined 2026-08-18:
#     a name keyword only counts when the source data ALSO never gives the
#     course a room -- which is what separates 'Senior Project in Bioinfor.'
#     (22 staff, 0 rooms, 0 meeting times) from 'Project Management' (1 room,
#     1 meeting time).
#   * Roster width is NOT used. It looked promising and is wrong: five
#     instructors on 'Intensive English Foundation 1' is genuine team teaching
#     in a real room, and five on an M.Sc. thesis is a roster.
SUPERVISION_COURSE_TYPES = frozenset({
    "Thesis",
    "Thesis / Dissertation Master",
    "Thesis / Dissertation Doctorat",
    "Senior Project Supervision",
    "Project",
    "Independent Study",
    "Proposal Doctorate",
    "Comprehensive Exam",
    "Qualifying Exam",
})

# Courses whose course_type does NOT say supervision but which demonstrably are.
# Mined from schedule_id=1 on 2026-08-18: name matches project/thesis/
# dissertation/graduation AND the course is never given a room in the source.
# Pinned by tests/test_data_invariants.py::
# test_supervision_courses_are_miscategorised_by_type.
SUPERVISION_COURSE_IDS = frozenset({
    103490,    # 'Graduation Project'            typed 'Office Hours'
    401493,    # 'Environmental Outreach Project' typed 'Office Hours'
    703415,    # 'VC Graduation Project II'      typed 'Office Hours'
    1002704,   # 'Investi. Leading to Thesis I'  typed 'Seminar Graduate'
    1501497,   # 'Senior Project in Bioinfor.'   typed 'Lecture Undergraduate'
})


def _course_identity(obj):
    """(course_id, course_type) from either a Section or a ScheduleItem."""
    course = getattr(obj, "course", None)
    if course is not None:                       # Section
        return course.id, course.type
    return obj.course_id, obj.course_type        # ScheduleItem


def is_supervision(obj) -> bool:
    """True when this section/item is supervision rather than taught contact.

    Supervision still needs a slot in the schedule -- what it does NOT do is
    consume its instructors' teaching time. See instructor_occupancy_id().
    """
    course_id, course_type = _course_identity(obj)
    if course_id in SUPERVISION_COURSE_IDS:
        return True
    return course_type in SUPERVISION_COURSE_TYPES


def instructor_occupancy_ids(obj):
    """Every instructor identity that occupies a timeslot. Possibly empty.

    Returns () in two cases, which every occupancy structure must skip:
      * no instructor recorded (the majority of rows)
      * supervision -- a supervisor attached to a thesis is not in a room

    Returns ALL of them otherwise: a team-taught class commits every one of its
    instructors for that hour. Reading a single `.instructor_id` discarded 598
    real assignments -- 'Intensive English Foundation 1' is taught by five
    people, and four of them were invisible to the conflict counter.

    Funnelling every occupancy site through one function is deliberate: the
    instructor rules were previously repeated at ~20 call sites across five
    algorithms and three different structures (sets, Counters, owner-maps), and
    they drifted. Anything that builds instructor occupancy should call this
    rather than reading .instructor_id / .instructor_ids directly.

    Known trade-off: a handful of supervision activities are real group sessions
    with a room and an hour (e.g. 1100599 'Graduation Project', 20 staff at
    T 19:00). Those stop being checked for instructor clashes. That is the right
    side to err on -- the false constraints removed vastly outnumber the genuine
    clashes lost, and the ROOM check still applies to them unchanged.
    """
    if is_supervision(obj):
        return ()
    instructor_ids = getattr(obj, "instructor_ids", None)
    if instructor_ids:
        return tuple(instructor_ids)
    instructor_id = getattr(obj, "instructor_id", None)
    return () if instructor_id is None else (instructor_id,)


def instructor_occupancy_id(obj):
    """First occupying instructor, or None.

    Convenience for the few places that genuinely need one value (a sort key, a
    'who is blocking this slot' lookup). Anything COUNTING or RESERVING must use
    instructor_occupancy_ids() -- using this instead silently ignores every
    instructor after the first.
    """
    ids = instructor_occupancy_ids(obj)
    return ids[0] if ids else None


# ── Sibling meeting patterns ─────────────────────────────────────────────────
# A section that meets in several blocks becomes several Sections sharing one
# sibling_key. They are the same students, so:
#
#   * two siblings at the same timeslot is a HARD conflict (nobody can attend
#     their own class twice at once), and
#   * siblings that need a room should share ONE room -- which is what the
#     source data does in every case: the six Studio sections of 404421 repeat
#     the same room across both their blocks, and the clinical sections carry no
#     room at all.


def sibling_key(obj):
    """Identity shared by every meeting pattern of one section."""
    key = getattr(obj, "sibling_key", None)
    if key is not None:
        return key
    course = getattr(obj, "course", None)
    if course is not None:
        return (course.id, str(obj.no))
    return (obj.course_id, str(obj.section))


def group_siblings(items):
    """sibling_key -> [items], keeping only sections that really have siblings."""
    groups = defaultdict(list)
    for item in items:
        groups[sibling_key(item)].append(item)
    return {key: group for key, group in groups.items() if len(group) > 1}


def count_sibling_room_splits(schedule) -> int:
    """Sibling groups whose blocks sit in more than one room.

    Blocks with no room are ignored -- most multi-pattern sections are clinical,
    which classify as time-only and are given no room at all, so this is vacuous
    for them. In practice it applies to the six Studio sections of course
    404421, which in the source data already repeat one room across both blocks.
    """
    splits = 0
    for group in group_siblings(schedule).values():
        rooms = {item.room_id for item in group if item.room_id is not None}
        if len(rooms) > 1:
            splits += 1
    return splits


def _occupancy_of(schedule):
    rooms = set()
    instructors = set()
    for item in schedule:
        if item.timeslot_id is None:
            continue
        if item.room_id is not None:
            rooms.add((item.room_id, item.timeslot_id))
        for instructor_id in instructor_occupancy_ids(item):
            instructors.add((instructor_id, item.timeslot_id))
    return rooms, instructors


def resolve_sibling_time_conflicts(schedule, sections, timeslots,
                                   valid_timeslot_cache=None):
    """Give each meeting block of one section its own hour.

    The five algorithms place blocks independently and have no cross-block
    state, so a section that meets twice happily gets both blocks at the same
    time -- which is impossible for the students, and which nothing else in the
    model catches: the blocks may carry no room and no instructor.

    Threading a coupling constraint through five different placement loops for
    the eleven sections this affects would be disproportionate, so it is
    repaired here instead. A block only moves to a slot that is valid for its
    course type and free for its room and every one of its instructors, so this
    can never introduce a conflict; blocks with nowhere to go are left alone and
    still register in count_sibling_conflicts().
    """
    groups = group_siblings(schedule)
    if not groups:
        return schedule

    if valid_timeslot_cache is None:
        # Only the handful of sections that actually have siblings, rather than
        # rebuilding the whole cache.
        valid_timeslot_cache = {
            sibling_key(section): {
                ts.id for ts in get_valid_timeslots_for_section(section, timeslots)
            }
            for section in sections
            if sibling_key(section) in groups
        }

    occupied_rooms, occupied_instructors = _occupancy_of(schedule)

    for key, group in groups.items():
        candidates = sorted(valid_timeslot_cache.get(key) or ())
        taken = set()

        for item in group:
            current = item.timeslot_id
            if current is None:
                continue
            if current not in taken:
                taken.add(current)
                continue

            instructors = instructor_occupancy_ids(item)
            replacement = next(
                (
                    slot for slot in candidates
                    if slot not in taken
                    and (item.room_id is None
                         or (item.room_id, slot) not in occupied_rooms)
                    and all((instructor_id, slot) not in occupied_instructors
                            for instructor_id in instructors)
                ),
                None,
            )
            if replacement is None:
                continue

            if item.room_id is not None:
                occupied_rooms.discard((item.room_id, current))
                occupied_rooms.add((item.room_id, replacement))
            for instructor_id in instructors:
                occupied_instructors.discard((instructor_id, current))
                occupied_instructors.add((instructor_id, replacement))

            item.timeslot_id = replacement
            taken.add(replacement)

    return schedule


def finalize_schedule(schedule, sections, timeslots, valid_timeslot_cache=None):
    """Repairs every algorithm applies to the schedule it returns.

    Order matters: separate the sibling blocks in TIME first, because two blocks
    sharing an hour can never also share a room -- unifying rooms before the
    times are fixed would silently do nothing.
    """
    schedule = resolve_sibling_time_conflicts(
        schedule, sections, timeslots, valid_timeslot_cache
    )
    return unify_sibling_rooms(schedule)


def unify_sibling_rooms(schedule):
    """Move every meeting block of one section into a single shared room.

    A section that meets twice a week meets in the SAME place both times; the
    source data does exactly that. The algorithms place each block independently
    and have no cross-block state, so rather than thread a coupling constraint
    through five different placement loops, this repairs it afterwards -- the
    affected population is six sections, so a repair pass is the proportionate
    tool.

    Never creates a conflict: a block only moves if the target room is actually
    free at that block's timeslot. Blocks that cannot move are left where they
    are, so this can improve the schedule but never break it. Mutates in place
    and returns the schedule.
    """
    occupied = {
        (item.room_id, item.timeslot_id)
        for item in schedule
        if item.room_id is not None and item.timeslot_id is not None
    }

    for group in group_siblings(schedule).values():
        roomed = [item for item in group
                  if item.room_id is not None and item.timeslot_id is not None]
        if len(roomed) < 2:
            continue
        if len({item.room_id for item in roomed}) == 1:
            continue  # already unified

        # Free the group's own bookings first, so a room is not judged busy on
        # account of the very blocks being moved.
        for item in roomed:
            occupied.discard((item.room_id, item.timeslot_id))

        # Try the rooms these blocks already sit in -- all are viable for this
        # section by construction. Most-used first, ties on lowest id so repeated
        # runs converge rather than oscillate. Trying every candidate rather than
        # only the most common one matters: the popular room is often busy at one
        # sibling's hour while a less popular one is free at all of them.
        counts = Counter(item.room_id for item in roomed)
        candidates = sorted(counts, key=lambda room_id: (-counts[room_id], room_id))

        target = next(
            (
                room_id for room_id in candidates
                if all((room_id, item.timeslot_id) not in occupied
                       for item in roomed)
            ),
            None,
        )

        if target is not None:
            for item in roomed:
                item.room_id = target

        for item in roomed:
            occupied.add((item.room_id, item.timeslot_id))

    return schedule


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
        if item.timeslot_id != timeslot_id:
            continue
        # Membership, not equality: a team-taught item commits all its
        # instructors, and equality only ever saw the first one.
        if instructor_id in instructor_occupancy_ids(item):
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
    # Supervision does not consume its instructors' teaching time, so it neither
    # creates occupancy nor is blocked by it -- instructor_occupancy_ids returns
    # () and the check short-circuits to free. See section 1c.
    # EVERY instructor must be free, not just the first: a team-taught class
    # cannot run if any one of its teachers is already committed.
    timeslot_id = timeslot.id if timeslot else None
    instructor_ids = instructor_occupancy_ids(section)

    if occupied_instructors is not None:
        instructor_free = all(
            is_instructor_free_map(occupied_instructors, instructor_id, timeslot_id)
            for instructor_id in instructor_ids
        )
    else:
        instructor_free = all(
            is_instructor_free(schedule, instructor_id, timeslot_id)
            for instructor_id in instructor_ids
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
        if item.timeslot_id is None:
            continue
        for instructor_id in instructor_occupancy_ids(item):
            key = (instructor_id, item.timeslot_id)
            if key in seen:
                return False
            seen.add(key)
    return True


def check_siblings(schedule: list, data=None) -> bool:
    """No section may hold two of its own meeting blocks at one timeslot."""
    for group in group_siblings(schedule).values():
        times = [item.timeslot_id for item in group if item.timeslot_id is not None]
        if len(times) != len(set(times)):
            return False
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