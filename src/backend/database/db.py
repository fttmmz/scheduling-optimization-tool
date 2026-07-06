import os
from supabase import create_client
from supabase.lib.client_options import SyncClientOptions
from dotenv import load_dotenv

# connect to database via supabase
load_dotenv()

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")

# This client is a single instance shared across every request from every
# user (FastAPI handlers all import the same module-level `supabase`).
# auto_refresh_token/persist_session default to True, which is meant for a
# client that owns one user's session — here it would silently rotate
# whichever user's refresh token it last touched via a background timer,
# invalidating that user's frontend-held refresh token out from under them.
# The backend is stateless (tokens live in the frontend's localStorage), so
# both must stay off.
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
    options=SyncClientOptions(auto_refresh_token=False, persist_session=False),
)

print("Connected to Supabase!")

# list of existing tables
tables = [
    "college",
    "course_req",
    "courses",
    "department",
    "instructor",
    "preference",
    "room",
    "rule_set",
    "schedule",
    "schedule_detailes",
    "scheduling_rule",
    "student",
    "timeslot",
    "user",
]

# loop thorough all tables
# for table in tables:
#     print(f"\nTable: {table}")
#     data = supabase.table(table).select("*").execute()
# print("\ndone")
