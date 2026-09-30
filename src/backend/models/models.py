# models.py


def pattern_key(row):
    """Identity of one meeting block, ignoring which instructor filed the row."""
    return (row["room_id"], row["timeslot_id"], row["sec_capacity"])


def group_rows_by_pattern(rows):
    """Raw schedule rows -> {(course_id, section): {pattern_key: [rows]}}.

    The source table stores one row per INSTRUCTOR, so 4,698 rows describe
    3,270 real meetings. Grouping by (course, section, room, timeslot, capacity)
    collapses the 777 exact repeats and 651 roster rows while keeping every
    genuine meeting block. Grouping by (course, section) alone would merge the
    clinical sections that meet five times a week and delete four meetings.

    Source room/timeslot are used ONLY to count blocks; the scheduler assigns
    fresh ones. Order is first-seen throughout, so runs are reproducible.

    Lives here rather than in loader.py because loader.py opens a Supabase
    client at import time, and this needs to be testable offline.
    """
    grouped = {}
    for row in rows:
        section_key = (row["course_id"], row["section"])
        grouped.setdefault(section_key, {}).setdefault(pattern_key(row), []).append(row)
    return grouped


def instructors_in(rows):
    """Distinct non-null instructor ids across rows, first-seen order."""
    return list(dict.fromkeys(
        row["instructor_id"] for row in rows if row["instructor_id"] is not None
    ))


def section_info_index(rows):
    """section_info rows -> {(course_id, section): {"campus", "enrolment"}}.

    Keyed exactly as group_rows_by_pattern() keys schedule rows, so the two
    join without any normalisation. Lives here, not in loader.py, so tests can
    build it offline from a fixture.
    """
    return {
        (row["course_id"], row["section"]): {
            "campus": row.get("campus"),
            "enrolment": row.get("enrolment"),
        }
        for row in rows
    }


def build_sections(rows, section_info=None):
    """Raw schedule rows -> one Section per (course, section, meeting pattern).

    *section_info* maps (course_id, section) to the registrar's section-level
    facts (the section_info table: campus code, enrolment). Optional, so a
    database without that table still loads -- the rules then fall back to the
    section-number campus and the planned capacity.
    """
    section_info = section_info or {}
    sections = []
    for key, patterns in group_rows_by_pattern(rows).items():
        pattern_count = len(patterns)
        for pattern_index, pattern_rows in enumerate(patterns.values()):
            sections.append(Section(
                pattern_rows[0],
                instructor_ids=instructors_in(pattern_rows),
                pattern_index=pattern_index,
                pattern_count=pattern_count,
                info=section_info.get(key),
            ))
    return sections


class Course:
    def __init__(self, id, name, type, dept_id, level=None, course_class=None):
        self.id = id
        self.name = name
        self.type = type
        self.dept = dept_id
        # level ('Undergraduate' / 'Master' / 'Doctorate' / ...) and course_class
        # ('UG Year 1', 'Postgraduate', ...) are needed by classify_section:
        # the same course_type (e.g. 'Laboratory') needs a room at undergrad
        # level but not at grad level. Optional so older callers still work.
        self.level = level
        self.course_class = course_class


class Room:
    def __init__(self, data):
        self.id = data["room_id"]
        self.capacity = data["capacity"]
        self.type = data["room_type"]
        self.building = data["building"]
        self.dept_id = data["dept_id"]
        self.no=data["room_num"]

class Timeslot:
    def __init__(self, data):
        self.id = data["timeslot_id"]
        self.day = data["day"]
        self.start = data["start_time"]
        self.end = data["end_time"]


class Section:
    """One thing to be scheduled: a (course, section, meeting pattern).

    NOT one source row. `schedule_detailes` stores one row per INSTRUCTOR, so
    4,698 rows describe 3,270 actual meetings -- 777 exact repeats and 651
    roster rows collapse away. See HANDOVER.md section 4.

    Two structures matter here:

      instructor_ids -- a class may be taught by several people. A single
                        scalar kept only the first and silently discarded 598
                        assignments.
      pattern_index  -- a section may meet in several blocks (clinical practice
                        runs morning and afternoon across different days). Each
                        block is its own Section sharing one `sibling_key`;
                        collapsing them to one entity would delete real
                        meetings.
    """

    def __init__(self, data, instructor_ids=None, pattern_index=0, pattern_count=1,
                 info=None):
        self.course = Course(
            id=data["courses"]["course_id"],
            name=data["courses"]["name"],
            type=data["courses"]["course_type"],
            dept_id=data["courses"]["dept_id"],
            level=data["courses"].get("level"),
            course_class=data["courses"].get("course_class"),
        )
        self.no = data["section"]
        self.capacity = data["sec_capacity"]

        # Registrar facts from section_info, None when unknown. `campus` is the
        # raw CAMP code (MAM, MAW, UOS, MDM, ...); `enrolment` is ENROL, the
        # students actually registered -- which exceeds `capacity` (the
        # planned maximum) on 691 sections.
        info = info or {}
        self.campus = info.get("campus")
        self.enrolment = info.get("enrolment")

        # Callers that still build a Section from one raw row (tests, older
        # code) get the single-instructor behaviour for free.
        if instructor_ids is None:
            single = data.get("instructor_id")
            instructor_ids = (single,) if single is not None else ()
        self.instructor_ids = tuple(instructor_ids)

        self.pattern_index = pattern_index
        self.pattern_count = pattern_count

        self.tutorial_parent_id = None
        self.lab_parent_id = None

    @property
    def instructor_id(self):
        """First instructor, or None.

        Retained so the many call sites that only need *an* instructor -- and
        the DB write-back, which has one instructor_id column -- keep working.
        Never use it to decide occupancy: use instructor_occupancy_ids().
        """
        return self.instructor_ids[0] if self.instructor_ids else None

    @property
    def sibling_key(self):
        """Shared by every meeting pattern of one section.

        Two Sections with the same sibling_key are the same students, so they
        must not be scheduled at the same time.
        """
        return (self.course.id, str(self.no))

    @property
    def id(self):
        """Stable unique identifier for use in graphs and lookups."""
        return (self.course.id, self.no, self.pattern_index)


class ScheduleItem:
    def __init__(
        self,
        course_id,
        course_name,
        course_type,
        course_dept,
        capacity,
        instructor_id,
        room_id,
        timeslot_id,
        section,
        level=None,
        course_class=None,
        instructor_ids=None,
        pattern_index=0,
        campus=None,
        enrolment=None,
    ):
        self.course_id = course_id
        self.course_name = course_name
        self.course_type = course_type
        self.course_dept = course_dept
        self.capacity = capacity
        self.instructor_id = instructor_id
        self.room_id = room_id
        self.timeslot_id = timeslot_id
        self.section = section
        # Every instructor on this class, not just the reported one. Defaults
        # from instructor_id so existing call sites keep working; anything that
        # decides occupancy must go through instructor_occupancy_ids().
        if instructor_ids is None:
            instructor_ids = (instructor_id,) if instructor_id is not None else ()
        self.instructor_ids = tuple(instructor_ids)
        self.pattern_index = pattern_index
        # Optional: carried so item-level classification (hybrid.py) can be
        # level-aware, same reason as Course.level above. Defaults to None so
        # existing ScheduleItem(...) call sites in the algorithms still work.
        self.level = level
        self.course_class = course_class
        # Registrar facts, as on Section. The hard room rules read them, so an
        # item that loses them is judged by the fallback rules instead.
        self.campus = campus
        self.enrolment = enrolment

    @classmethod
    def from_section(cls, section, room_id=None, timeslot_id=None):
        """The one way an algorithm turns a Section into a ScheduleItem.

        Every field the constraint rules read is copied here, in one place.
        Five algorithms used to spell this constructor out by hand, and a field
        added to one copy but not the others (`level`, three separate times)
        silently changed how the item was classified.
        """
        course = section.course
        return cls(
            course_id=course.id,
            course_name=course.name,
            course_type=course.type,
            course_dept=course.dept,
            capacity=section.capacity,
            instructor_id=section.instructor_id,
            room_id=room_id,
            timeslot_id=timeslot_id,
            section=str(section.no),
            level=getattr(course, "level", None),
            course_class=getattr(course, "course_class", None),
            instructor_ids=getattr(section, "instructor_ids", ()),
            pattern_index=getattr(section, "pattern_index", 0),
            campus=getattr(section, "campus", None),
            enrolment=getattr(section, "enrolment", None),
        )

    @property
    def sibling_key(self):
        """Shared by every meeting pattern of one section. See Section."""
        return (self.course_id, str(self.section))

    def __repr__(self):
        return (
            f"ScheduleItem(course={self.course_id}, "
            f"section={self.section}, "
            f"instructor={self.instructor_id}, "
            f"room={self.room_id}, "
            f"timeslot={self.timeslot_id})"
        )
