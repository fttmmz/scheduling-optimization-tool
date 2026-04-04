from models.models import Course, Instructor, Room, TimeSlot, ScheduleDetail
from database.db import supabase 

class DataLoader:

    def load_courses(self):
        res = supabase.table("courses").select("*").execute()
        return [Course(row) for row in res.data]

    def load_rooms(self):
        res = supabase.table("room").select("*").execute()
        return [Room(row) for row in res.data]

    def load_timeslots(self):
        res = supabase.table("timeslot").select("*").execute()
        return [TimeSlot(row) for row in res.data]
    
    def load_instructors(self):
        res=supabase.table("instructor").select("*").execute()
        return [Instructor(row) for row in res.data]
    
    def load_schedule_details(self):  
        res = supabase.table("schedule_details").select("*").execute()
        return [ScheduleDetail(row) for row in res.data]
    
    def build_course_instructor_map(self, schedule_details):
        mapping = {}

        for detail in schedule_details:
            c_id = detail.course_id
            i_id = detail.instructor_id

            if c_id not in mapping:
                mapping[c_id] = set()

            if i_id is not None:
                mapping[c_id].add(i_id)

        return mapping

    def load_all(self):
        courses = self.load_courses()
        instructors = self.load_instructors()
        rooms = self.load_rooms()
        timeslots = self.load_timeslots()
        schedule_details = self.load_schedule_details()

        course_instructor_map = self.build_course_instructor_map(schedule_details)

        return {
            "courses": courses,
            "instructors": instructors,
            "rooms": rooms,
            "timeslots": timeslots,
            "schedule_details": schedule_details,
            "course_instructor_map": course_instructor_map
    }


