# constraints.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for ALL scheduling constraints and preprocessing.
# Every algorithm (greedy, genetic, etc.) imports from here 
# ─────────────────────────────────────────────────────────────────────────────

import re


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


def get_valid_timeslots(section, timeslots: list) -> list:
    """
    Filter the master timeslot list down to slots appropriate for this
    section's course type.

      Lectures / seminars → MW or TR slots only (appears twice a week)
      Labs / Lecture-Lab  → single-day slots
      Everything else     → single-day slots
    """
    ct = section.course.type

    if ct in {
        "Lecture Undergraduate",
        "Lecture Graduate",
        "Lecutre / Studio Undergraduate",
        "Seminar Graduate",
    }:
        return [ts for ts in timeslots if ts.day in {"MW", "TR"}]

    if ct in {"Laboratory", "Lecture/Lab"}:
        return [ts for ts in timeslots if ts.day in SINGLE_DAY]

    return [ts for ts in timeslots if ts.day in SINGLE_DAY]


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
        suffix = sno[len(base_no) :]  # whatever was stripped

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


# 8. CONFLICT COUNTING  (for schedule scoring / comparison)


def count_instructor_conflicts(schedule: list) -> int:
    seen, conflicts = set(), 0
    for item in schedule:
        if item.instructor_id is None or item.timeslot_id is None:
            continue
        key = (item.instructor_id, item.timeslot_id)
        if key in seen:
            conflicts += 1
        else:
            seen.add(key)
    return conflicts


def count_room_conflicts(schedule: list) -> int:
    seen, conflicts = set(), 0
    for item in schedule:
        if item.room_id is None or item.timeslot_id is None:
            continue
        key = (item.room_id, item.timeslot_id)
        if key in seen:
            conflicts += 1
        else:
            seen.add(key)
    return conflicts


def count_campus_conflicts(schedule: list, rooms: list) -> int:
    room_map = {room.id: room for room in rooms}
    conflicts = 0
    for item in schedule:
        room = room_map.get(item.room_id)
        if not room:
            continue
        if get_section_campus(item.section) != get_building_campus(room.building):
            conflicts += 1
    return conflicts


def count_conflicts(schedule: list, rooms: list) -> int:
    """Total hard-constraint violation score (lower = better)."""
    return (
        count_instructor_conflicts(schedule)
        + count_room_conflicts(schedule)
        + count_campus_conflicts(schedule, rooms)
    )


def calculate_fitness(
    schedule: list,
    rooms: list,
    scheduled_count: int,
    total_sections: int,
) -> float:
    """Compute a higher-is-better fitness score for the schedule."""
    conflict_penalty = count_conflicts(schedule, rooms) * 10
    unscheduled_penalty = max(0, total_sections - scheduled_count) * 20

    # Base fitness gives strong preference to a complete, conflict-free schedule.
    fitness = 1000.0 - conflict_penalty - unscheduled_penalty
    return max(0.0, fitness)


# 9. FULL-SCHEDULE VALIDATORS  (used for final verification / testing)


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
