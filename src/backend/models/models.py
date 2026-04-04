class Course:
    def __init__(self, data):
        self.course_id = data["course_id"]
        self.name = data["name"]
        self.credits = data["credits"]
        self.dept_id = data["dept_id"]


class Room:
    def __init__(self, data):
        self.room_id = data["room_id"]
        self.capacity = data["capacity"]
        self.room_type = data["room_type"]


class TimeSlot:
    def __init__(self, data):
        self.timeslot_id = data["timeslot_id"]
        self.day = data["day"]
        self.start = data["start_time"]
        self.end = data["end_time"]
class Instructor:
    def __init__(self, data):
        self.inst_id = data["inst_id"]
        self.name = data["name"]
        self.dept_id=data["dept_id"]
class ScheduleDetail:
    def __init__(self, data):
        self.course_id = data["course_id"]
        self.instructor_id = data["instructor_id"]
        self.section=data["section"]

class ScheduleItem:
    def __init__(self, course_id, instructor_id, room_id, timeslot_id,section):
        self.course_id = course_id
        self.instructor_id = instructor_id
        self.room_id = room_id
        self.timeslot_id = timeslot_id
        self.section = section