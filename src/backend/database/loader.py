from backend.models.models import (
    Room, Timeslot, build_sections, group_rows_by_pattern, section_info_index,
)
from backend.database.db import supabase
from backend.Optimization.constraints import is_placeholder_timeslot


def load_rooms():
    res = supabase.table("room").select("*").execute()
    return [Room(row) for row in res.data]


def load_timeslots():
    """Real, bookable timeslots only.

    Excludes both placeholder encodings -- day IS NULL (the TBA bucket) and
    zero-duration slots -- so no algorithm can assign one. They are not
    schedulable times; they mean "this activity has no fixed meeting time".
    """
    res = supabase.table("timeslot").select("*").execute()
    return [
        timeslot
        for timeslot in (Timeslot(row) for row in res.data)
        if not is_placeholder_timeslot(timeslot)
    ]


def load_section_details():
    """The list of things to schedule -- one Section per (course, section, pattern).

    The source table stores one row per INSTRUCTOR, not one per class, so the
    4,698 rows describe 3,270 real meetings. Building one Section per row (what
    this used to do) inflated demand by ~44%: it scheduled the same class into
    several rooms at once, and made "0 unscheduled out of 4,698" mean far less
    than it appeared to. See HANDOVER.md section 4.

    Grouping is by (course, section, room, timeslot, capacity) -- the source
    room/timeslot are used ONLY to count how many distinct meeting blocks a
    section has, then discarded; the scheduler assigns fresh ones. Grouping by
    (course, section) alone would be wrong: it merges the clinical sections that
    genuinely meet five times a week into one entity and deletes four meetings.
    """
    rows = _fetch_schedule_rows(1)
    return build_sections(rows, load_section_info())


def load_section_info():
    """(course_id, section) -> registrar facts {campus, enrolment}.

    From the section_info table (scripts/data_import/section_info_import.py).
    The campus code drives the medical and branch-campus rules and enrolment
    the room size, so an empty table silently turns those off -- hence the
    warning rather than a quiet {}.
    """
    rows, offset, page = [], 0, 1000
    while True:
        res = (
            supabase.table("section_info")
            .select("course_id,section,campus,enrolment")
            .range(offset, offset + page - 1)
            .execute()
        )
        rows.extend(res.data)
        if len(res.data) < page:
            break
        offset += page
    if not rows:
        print("[loader] WARNING: section_info is empty -- campus codes and "
              "enrolment unknown; medical/branch rules and ENROL sizing are off")
    return section_info_index(rows)


def _fetch_schedule_rows(schedule_id):
    rows = []
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
        rows.extend(res.data)

        if len(res.data) < page_size:  # no more rows left
            break
        offset += page_size

    return rows




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
    """Load a saved schedule from the database.

    Collapsed the same way as load_section_details(): one ScheduleItem per
    (course, section, room, timeslot, capacity), carrying every instructor on
    that block. Without this the manual schedule would be read at row
    granularity while generated schedules are read at meeting granularity, and
    the two would not be comparable -- the manual one would appear to contain
    hundreds of double-bookings that are really just repeated rows.
    """
    from backend.models.models import ScheduleItem

    from backend.models.models import instructors_in

    # Same registrar facts as the sections get, so a saved (or the manual)
    # schedule is judged by the same room rules as a freshly generated one.
    section_info = load_section_info()

    schedule_items = []
    for key, patterns in group_rows_by_pattern(_fetch_schedule_rows(schedule_id)).items():
        info = section_info.get(key, {})
        for pattern_index, pattern_rows in enumerate(patterns.values()):
            row = pattern_rows[0]
            course = row["courses"]
            instructor_ids = instructors_in(pattern_rows)
            schedule_items.append(ScheduleItem(
                course_id=row["course_id"],
                course_name=course["name"],
                course_type=course["course_type"],
                course_dept=course["dept_id"],
                capacity=row["sec_capacity"],
                instructor_id=instructor_ids[0] if instructor_ids else None,
                room_id=row["room_id"],
                timeslot_id=row["timeslot_id"],
                section=row["section"],
                level=course.get("level"),
                course_class=course.get("course_class"),
                instructor_ids=instructor_ids,
                pattern_index=pattern_index,
                campus=info.get("campus"),
                enrolment=info.get("enrolment"),
            ))

    return schedule_items