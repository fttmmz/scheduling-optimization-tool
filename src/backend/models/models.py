# models.py

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
    def __init__(self, data):
        self.course = Course(
            id=data["courses"]["course_id"],
            name=data["courses"]["name"],
            type=data["courses"]["course_type"],
            dept_id=data["courses"]["dept_id"],
            level=data["courses"].get("level"),
            course_class=data["courses"].get("course_class"),
        )
        self.instructor_id = data["instructor_id"]
        self.no = data["section"]
        self.capacity = data["sec_capacity"]

        self.tutorial_parent_id = None
        self.lab_parent_id = None

    @property
    def id(self):
        """Stable unique identifier for use in graphs and lookups."""
        return (self.course.id, self.no)


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
        # Optional: carried so item-level classification (hybrid.py) can be
        # level-aware, same reason as Course.level above. Defaults to None so
        # existing ScheduleItem(...) call sites in the algorithms still work.
        self.level = level
        self.course_class = course_class

    def __repr__(self):
        return (
            f"ScheduleItem(course={self.course_id}, "
            f"section={self.section}, "
            f"instructor={self.instructor_id}, "
            f"room={self.room_id}, "
            f"timeslot={self.timeslot_id})"
        )
