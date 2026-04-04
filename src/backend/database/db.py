import os
from supabase import create_client
from dotenv import load_dotenv



#connect to database via supabase
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Connected to Supabase!")

#list of existing tables
tables = [
    "college", "course_req", "courses", "department",
    "instructor", "preference", "room", "rule_set",
    "schedule", "schedule_detailes", "scheduling_rule",
    "student", "timeslot", "user"
]

#loop thorough all tables 
for table in tables:
    print(f"\nTable: {table}")
    data = supabase.table(table).select("*").execute()
print("\ndone")