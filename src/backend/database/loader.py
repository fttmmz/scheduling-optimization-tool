import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.models import Timeslot, Room, Section
from database.db import supabase 

def load_rooms():
    res = supabase.table("room").select("*").execute()
    return [Room(row) for row in res.data]

def load_timeslots():
    res = supabase.table("timeslot").select("*").execute()
    return [Timeslot(row) for row in res.data]
    
    
    
def load_section_details():  
    sections_data = []
    page_size = 1000
    offset = 0

    while True:
        res = supabase.table("schedule_detailes").select("*, courses(*)").range(offset, offset + page_size - 1).execute()
        sections_data.extend(res.data)
        
        if len(res.data) < page_size:  # no more rows left
            break
        offset += page_size

    return [Section(row) for row in sections_data]     

def get_scheduling_data():
    rooms = load_rooms()
    timeslots = load_timeslots()
    sections = load_section_details()
    return rooms, timeslots, sections




