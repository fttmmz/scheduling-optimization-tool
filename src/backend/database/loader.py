from backend.models.models import Timeslot, Room, Section
from backend.database.db import supabase


def load_rooms():
    res = supabase.table("room").select("*").execute()
    return [Room(row) for row in res.data]


def load_timeslots():
    res = supabase.table("timeslot").select("*").execute()
    return [Timeslot(row) for row in res.data if row["day"] is not None]


def load_section_details():
    sections_data = []
    page_size = 1000
    offset = 0

    while True:
        res = (
            supabase.table("schedule_detailes")
            .select("*, courses(*)")
            .eq("schedule_id", 1)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        sections_data.extend(res.data)

        if len(res.data) < page_size:  # no more rows left
            break
        offset += page_size

    return [Section(row) for row in sections_data]


def get_scheduling_data():
    rooms = load_rooms()
    timeslots = load_timeslots()
    sections = load_section_details()

    return {
        "rooms": rooms,
        "timeslots": timeslots,
        "sections": sections,
    }


def save_schedule(schedule_items, schedule_detailes):
    """Save a generated schedule to the database.

    The schedule metadata is stored in `schedule`, and each scheduled row
    is stored in `schedule_detailes` linked by `schedule_id`.
    """

    metadata = {
        "name": schedule_detailes.get("sch_name"),
        "alg_name": schedule_detailes.get("alg"),
        "semester": schedule_detailes.get("semester"),
        "fitness_score": schedule_detailes.get("fitness_score"),
        "conflicts_count": schedule_detailes.get("conflicts"),
        "exec_time": schedule_detailes.get("exec_time"),
        "user_id": schedule_detailes.get("user_id"),
        "rule_set_id": schedule_detailes.get("rule_set"),
        "Scheduled": schedule_detailes.get("Scheduled"),
        "unscheduled":schedule_detailes.get("unscheduled")
    }

    schedule_res = supabase.table("schedule").insert(metadata).execute()
    if not getattr(schedule_res, "data", None):
        raise RuntimeError("Failed to save schedule metadata: no data returned")

    saved_schedule = schedule_res.data[0]
    schedule_id = saved_schedule.get("schedule_id")
    if schedule_id is None:
        raise RuntimeError("Saved schedule metadata did not return an ID.")

    detail_records = []
    for item in schedule_items:
        detail_records.append(
            {
                "schedule_id": schedule_id,
                "course_id": item.course_id,
                "section": item.section,
                "instructor_id": item.instructor_id,
                "room_id": item.room_id,
                "timeslot_id": item.timeslot_id,
                "sec_capacity": item.capacity,
            }
        )

    detail_res = supabase.table("schedule_detailes").insert(detail_records).execute()
    if not getattr(detail_res, "data", None):
        raise RuntimeError("Failed to save schedule rows: no data returned")

    return {
        "schedule": saved_schedule,
        "details_saved": len(detail_records),
    }


def load_schedule(schedule_id):
    """Load a saved schedule from the database."""
    from backend.models.models import ScheduleItem

    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        res = (
            supabase.table("schedule_detailes")
            .select("*, courses(*)")
            .eq("schedule_id", schedule_id)
            .range(offset, offset + page_size - 1)
            .execute()
        )

        if not res.data:
            break

        all_rows.extend(res.data)

        # if we got less than a full page, we're done
        if len(res.data) < page_size:
            break

        offset += page_size

    schedule_items = []
    for row in all_rows:
        course = row["courses"]
        item = ScheduleItem(
            course_id=row["course_id"],
            course_name=course["name"],
            course_type=course["course_type"],
            course_dept=course["dept_id"],
            capacity=row["sec_capacity"],
            instructor_id=row["instructor_id"],
            room_id=row["room_id"],
            timeslot_id=row["timeslot_id"],
            section=row["section"],
        )
        schedule_items.append(item)

    return schedule_items