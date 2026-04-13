class Course:
    def __init__(self, id, name, type):
        self.id =id
        self.name =name
        self.type = type
class Room:
    def __init__(self, data):
        self.id = data["room_id"]
        self.capacity = data["capacity"]
        self.type = data["room_type"]
        self.building = data["building"]

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
            )
        self.instructor_id = data["instructor_id"]
        self.no = data["section"]
        self.capacity=data["sec_capacity"]

class scheduled_item:
    def __init__(self, section, room, timeslot):
        self.section = section
        self.room = room
        self.timeslot = timeslot

    def __repr__(self):
        return (
            f"scheduled_item(course={self.section.course.id}, "
            f"section={self.section.no}, "
            f"instructor={self.section.instructor_id}, "
            f"room={self.room.id}, "
            f"timeslot={self.timeslot.id})"
        )