"""What a section IS: what it needs, which class it belongs to, whether it is
supervision, and which meeting blocks are siblings. No room or time logic."""

from collections import defaultdict
from typing import NamedTuple

from . import institution_data as _institution
from .institution_data import COMBINED_COURSE_GROUPS, SUPERVISION_COURSE_IDS

# Registrar CAMP codes of branch campuses in other towns. None of their rooms
# are in the room table, so their sections are scheduled for a time only.
# getattr() so an older institution_data.py still imports (rule then off).
BRANCH_CAMPUS_CODES = getattr(_institution, "BRANCH_CAMPUS_CODES", frozenset())


def is_branch_campus(obj) -> bool:
    """A Section or ScheduleItem taught at a branch campus."""
    return getattr(obj, "campus", None) in BRANCH_CAMPUS_CODES


# COURSE-TYPE / LEVEL CLASSIFICATION
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


def needs_for(course_type, level, campus=None) -> Needs:
    """Core classifier: what does a (course_type, level) section need?

    Looks up the mined (level, course_type) table; falls back to the safe
    room+time default. This is the single place both the section path
    (section_needs) and the item path (item_needs) share.

    *campus* is the registrar's CAMP code. A branch-campus section keeps its
    time but never needs a room: we hold none of that campus's rooms, and
    placing it in one of ours would put the class in the wrong town.
    """
    needs = MINED_NEEDS.get((level, course_type), DEFAULT_NEEDS)
    if campus in BRANCH_CAMPUS_CODES and needs.room:
        needs = Needs(room=False, time=needs.time)
    return needs


def section_needs(section) -> Needs:
    """Canonical classifier for a Section object -> Needs(room, time)."""
    level = getattr(section.course, "level", None)
    return needs_for(section.course.type, level, getattr(section, "campus", None))


def item_needs(item) -> Needs:
    """Same as section_needs(), for a ScheduleItem."""
    return needs_for(item.course_type, getattr(item, "level", None),
                     getattr(item, "campus", None))


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


# SUPERVISION vs TEACHING
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
# COMBINED_COURSE_GROUPS (institution_data.py) -- a single signal is not enough:
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
    with a room and an hour (e.g. one graduation project with 20 staff at
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
#     source data does in every case: the six sections of one Studio course repeat
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
    for them. In practice it applies to the six sections of one Studio course,
    which in the source data already repeat one room across both blocks.
    """
    splits = 0
    for group in group_siblings(schedule).values():
        rooms = {item.room_id for item in group if item.room_id is not None}
        if len(rooms) > 1:
            splits += 1
    return splits
