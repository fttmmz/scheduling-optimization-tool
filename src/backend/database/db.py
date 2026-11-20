import os
from supabase import create_client
from supabase.lib.client_options import SyncClientOptions
from dotenv import load_dotenv

# connect to database via supabase
load_dotenv()

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")

def make_auth_client():
    # Sign-in and refresh can change the SDK client's Authorization header.
    # Give each Auth operation its own client so that session stays isolated.
    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
        options=SyncClientOptions(auto_refresh_token=False, persist_session=False),
    )


# Render can roll out the server-only key before the policy migration.
# Keep the publishable fallback for existing development environments.
supabase = create_client(
    SUPABASE_URL,
    os.getenv("SUPABASE_SERVER_KEY") or SUPABASE_KEY,
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
