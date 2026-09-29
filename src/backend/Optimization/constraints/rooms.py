"""Which rooms suit a section, and how badly each one bends the soft rules.

Nothing here refuses a room: get_viable_rooms() ranks, it does not filter."""

import re


# CONSTRAINT TIERS
#
# THE HARD/SOFT SPLIT (2026-08-13). Validated against the real manual schedule
# (schedule_id=10, 4698 sections, 0 unscheduled): that schedule is in active use,
# yet it breaks the rules this package used to enforce as HARD —
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


# COURSE-TYPE → ROOM-TYPE MAPPING

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


# CAMPUS HELPERS
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


def rank_rooms(obj, rooms, limit=DEFAULT_ROOM_CANDIDATES):
    """
    Rank every room for *obj* (a Section or a ScheduleItem) cheapest-first by
    soft penalty, and return the best ones. This REPLACES the old filter-based
    viable-room logic — see the tier note at the top of this module.

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
