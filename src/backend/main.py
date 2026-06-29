# main.py
import time
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.Optimization.constraints import tag_intro_it_pairs, tag_section_links
from backend.Optimization.engine import SchedulingEngine
from backend.Optimization.evaluation import (
    calculate_fitness,
    count_conflicts,
    count_instructor_conflicts,
    count_room_conflicts,
    count_campus_conflicts,
    count_room_type_conflicts,
    count_department_conflicts,
    count_capacity_conflicts,
    count_timeslot_guideline_conflicts,
    build_timeslot_guideline_cache,
)
from backend.database.db import supabase
from backend.database.loader import get_scheduling_data, load_schedule, save_schedule

app = FastAPI()

# In-memory job store for async optimization runs.
# Keys are job_id strings; values are dicts with "status" and optionally "result"/"error".
_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)


class TimetableRequest(BaseModel):
    algorithm: str = Field(
        "greedy",
        description="The scheduling algorithm to run. Available values: greedy, genetic, hybrid.",
    )
    num_runs: int = Field(
        1,
        ge=1,
        le=30,
        description="Number of independent runs for genetic/hybrid algorithms (1–30). Higher values improve quality but take longer.",
    )


class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    email: str
    password: str


class ScheduleSaveRequest(BaseModel):
    name: str
    algorithm: str
    semester: str = "202510"
    schedule: List[Dict[str, Any]]
    fitness: Optional[float] = None
    conflicts: Optional[int] = None
    exec_time: Optional[float] = None
    scheduled: Optional[int] = None
    unscheduled: Optional[int] = None


class CourseIn(BaseModel):
    course_id: int
    name: str
    credits: Optional[int] = None
    course_type: Optional[str] = None
    course_class: Optional[str] = None
    level: Optional[str] = None
    dept_id: Optional[int] = None


class InstructorIn(BaseModel):
    inst_id: int
    name: str
    dept_id: Optional[int] = None


class RoomIn(BaseModel):
    room_id: Optional[int] = None
    building: str
    room_num: str
    capacity: int
    room_type: str
    dept_id: Optional[int] = None


class TimeslotIn(BaseModel):
    timeslot_id: Optional[int] = None
    day: str
    start_time: str
    end_time: str


class PrerequisiteIn(BaseModel):
    course_id: int
    req_course_id: int
    req_type: str


class DeptIn(BaseModel):
    dept_id: Optional[int] = None
    name: str


def get_schedule_for_user(schedule_id: int, user_id: int):
    res = (
        supabase.table("schedule")
        .select("schedule_id, user_id")
        .eq("schedule_id", schedule_id)
        .maybe_single()
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Schedule not found")
    if res.data["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized for this schedule")
    return res.data


async def get_current_user_id(
    authorization: Optional[str] = Header(default=None),
) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    token = authorization.removeprefix("Bearer ").strip()

    try:
        user_response = supabase.auth.get_user(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    auth_id = str(user_response.user.id)
    profile_res = (
        supabase.table("user")
        .select("user_id")
        .eq("auth_id", auth_id)
        .maybe_single()
        .execute()
    )

    if not profile_res.data:
        raise HTTPException(status_code=404, detail="User profile not found")

    return profile_res.data["user_id"]


def run_optimization(algorithm: str, num_runs: int = 1) -> Dict[str, Any]:
    scheduling_data = get_scheduling_data()
    if scheduling_data is None:
        raise RuntimeError("Unable to load scheduling data from the database.")

    tag_section_links(scheduling_data["sections"])
    tag_intro_it_pairs(scheduling_data["sections"])

    try:
        engine = SchedulingEngine(algorithm, num_runs=num_runs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    t0 = time.perf_counter()
    schedule_items, scheduled, unscheduled_count = engine.run(scheduling_data)
    exec_time = time.perf_counter() - t0

    rooms = scheduling_data["rooms"]
    timeslots = scheduling_data["timeslots"]
    sections = scheduling_data["sections"]

    room_map = {r.id: r for r in rooms}
    timeslot_map = {ts.id: ts for ts in timeslots}

    enriched = []
    for item in schedule_items:
        room = room_map.get(item.room_id)
        ts = timeslot_map.get(item.timeslot_id)
        enriched.append(
            {
                "course_id": item.course_id,
                "course_name": item.course_name,
                "course_type": item.course_type,
                "course_dept": item.course_dept,
                "section": item.section,
                "instructor_id": item.instructor_id,
                "capacity": item.capacity,
                "room_id": item.room_id,
                "room_no": room.no if room else None,
                "room_type": room.type if room else None,
                "room_building": room.building if room else None,
                "timeslot_id": item.timeslot_id,
                "day": ts.day if ts else None,
                "start": ts.start if ts else None,
                "end": ts.end if ts else None,
            }
        )

    ts_cache = build_timeslot_guideline_cache(sections, timeslots)
    conflict_count = count_conflicts(schedule_items, rooms, sections, timeslots, valid_timeslot_cache=ts_cache)
    conflict_detail = {
        "instructor": count_instructor_conflicts(schedule_items),
        "room": count_room_conflicts(schedule_items),
        "campus": count_campus_conflicts(schedule_items, rooms),
        "room_type": count_room_type_conflicts(schedule_items, rooms),
        "department": count_department_conflicts(schedule_items, rooms),
        "capacity": count_capacity_conflicts(schedule_items, rooms),
        "timeslot_guidelines": count_timeslot_guideline_conflicts(schedule_items, sections, timeslots, ts_cache),
    }
    fitness_score = calculate_fitness(
        schedule_items, rooms, sections=sections, timeslots=timeslots, valid_timeslot_cache=ts_cache,
    )

    return {
        "schedule": enriched,
        "scheduled": scheduled,
        "unscheduled": unscheduled_count,
        "conflicts": conflict_count,
        "conflict_detail": conflict_detail,
        "fitness": fitness_score,
        "exec_time": exec_time,
        "total_sections": len(sections),
    }


def crud_list(table: str, order_by: Optional[str] = None):
    query = supabase.table(table).select("*")
    if order_by:
        query = query.order(order_by)
    res = query.execute()
    return res.data


def crud_insert(table: str, data: Dict[str, Any]):
    data = {k: v for k, v in data.items() if v is not None}
    res = supabase.table(table).insert(data).execute()
    if not res.data:
        raise HTTPException(status_code=400, detail=f"Failed to insert into {table}")
    return res.data[0]


def crud_update(table: str, id_col: str, id_val: Any, data: Dict[str, Any]):
    data = {k: v for k, v in data.items() if k != id_col}
    res = supabase.table(table).update(data).eq(id_col, id_val).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail=f"{table} row not found")
    return res.data[0]


def crud_delete(table: str, id_col: str, id_val: Any):
    res = supabase.table(table).delete().eq(id_col, id_val).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail=f"{table} row not found")
    return {"status": "deleted"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    # Matches any Vercel deployment of this project regardless of preview suffix
    allow_origin_regex=r"https://v0-university-timetable-tool-2[^.]*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    try:
        auth_response = supabase.auth.sign_in_with_password(
            {"email": request.email, "password": request.password}
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not auth_response.user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    auth_id = str(auth_response.user.id)
    profile_res = (
        supabase.table("user")
        .select("user_id, email, role")
        .eq("auth_id", auth_id)
        .maybe_single()
        .execute()
    )

    if not profile_res.data:
        raise HTTPException(status_code=404, detail="User profile not found")

    return {
        "user_id": profile_res.data["user_id"],
        "email": profile_res.data["email"],
        "role": profile_res.data["role"],
        "access_token": auth_response.session.access_token,
    }


@app.post("/api/auth/signup")
async def signup(request: SignupRequest):
    try:
        auth_response = supabase.auth.sign_up(
            {"email": request.email, "password": request.password}
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not auth_response.user:
        raise HTTPException(status_code=400, detail="Could not create account")

    auth_id = str(auth_response.user.id)
    profile_res = (
        supabase.table("user")
        .insert({"auth_id": auth_id, "email": request.email, "role": "student"})
        .execute()
    )

    if not profile_res.data:
        raise HTTPException(status_code=500, detail="Account created but profile setup failed")

    profile = profile_res.data[0]

    if not auth_response.session:
        return {
            "user_id": profile["user_id"],
            "email": profile["email"],
            "role": profile["role"],
            "access_token": None,
            "message": "Account created. Please check your email to confirm before logging in.",
        }

    return {
        "user_id": profile["user_id"],
        "email": profile["email"],
        "role": profile["role"],
        "access_token": auth_response.session.access_token,
    }


@app.post("/api/optimize")
async def optimize(
    request: TimetableRequest,
    user_id: int = Depends(get_current_user_id),
):
    """Synchronous optimize — fine for greedy and quick single-run genetic/hybrid."""
    result = run_optimization(request.algorithm, num_runs=request.num_runs)
    return {"status": "success", "timetable": result}


def _run_job(job_id: str, algorithm: str, num_runs: int) -> None:
    """Worker function executed in a thread pool for async optimization jobs."""
    try:
        result = run_optimization(algorithm, num_runs=num_runs)
        _jobs[job_id] = {"status": "done", "result": result}
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "error": str(exc)}


@app.post("/api/optimize/async")
async def optimize_async(
    request: TimetableRequest,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(get_current_user_id),
):
    """
    Async optimize — starts the job in a background thread and returns a job_id
    immediately. Use GET /api/jobs/{job_id} to poll for the result.
    Intended for multi-run genetic/hybrid experiments (num_runs up to 30).
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running"}
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_job, job_id, request.algorithm, request.num_runs)
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, user_id: int = Depends(get_current_user_id)):
    """Poll the result of an async optimization job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "running":
        return {"status": "running"}
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"])
    result = job["result"]
    del _jobs[job_id]
    return {"status": "done", "timetable": result}


@app.post("/api/schedules")
async def create_schedule(
    request: ScheduleSaveRequest,
    user_id: int = Depends(get_current_user_id),
):
    schedule_detailes = {
        "sch_name": request.name,
        "alg": request.algorithm,
        "semester": request.semester,
        "fitness_score": request.fitness,
        "conflicts": request.conflicts,
        "exec_time": request.exec_time,
        "user_id": user_id,
        "rule_set": None,
        "Scheduled": request.scheduled,
        "unscheduled": request.unscheduled,
    }

    class _Item:
        def __init__(self, d):
            self.course_id = d.get("course_id")
            self.section = d.get("section")
            self.instructor_id = d.get("instructor_id")
            self.room_id = d.get("room_id")
            self.timeslot_id = d.get("timeslot_id")
            self.capacity = d.get("capacity")

    schedule_items = [_Item(item) for item in request.schedule]

    try:
        result = save_schedule(schedule_items, schedule_detailes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "success", **result}


@app.get("/api/schedules")
async def list_schedules(user_id: int = Depends(get_current_user_id)):
    res = (
        supabase.table("schedule")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data


@app.get("/api/schedules/{schedule_id}")
async def get_schedule(schedule_id: int, user_id: int = Depends(get_current_user_id)):
    metadata = get_schedule_for_user(schedule_id, user_id)
    schedule_items = load_schedule(schedule_id)

    scheduling_data = get_scheduling_data()
    room_map = {r.id: r for r in scheduling_data["rooms"]}
    timeslot_map = {ts.id: ts for ts in scheduling_data["timeslots"]}

    enriched = []
    for item in schedule_items:
        room = room_map.get(item.room_id)
        ts = timeslot_map.get(item.timeslot_id)
        enriched.append(
            {
                "course_id": item.course_id,
                "course_name": item.course_name,
                "course_type": item.course_type,
                "course_dept": item.course_dept,
                "section": item.section,
                "instructor_id": item.instructor_id,
                "capacity": item.capacity,
                "room_id": item.room_id,
                "room_no": room.no if room else None,
                "room_type": room.type if room else None,
                "room_building": room.building if room else None,
                "timeslot_id": item.timeslot_id,
                "day": ts.day if ts else None,
                "start": ts.start if ts else None,
                "end": ts.end if ts else None,
            }
        )

    return {"metadata": metadata, "schedule": enriched}


@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: int, user_id: int = Depends(get_current_user_id)):
    get_schedule_for_user(schedule_id, user_id)
    supabase.table("schedule_detailes").delete().eq("schedule_id", schedule_id).execute()
    supabase.table("schedule").delete().eq("schedule_id", schedule_id).execute()
    return {"status": "deleted"}


# --- Courses ---

@app.get("/api/courses")
async def list_courses(user_id: int = Depends(get_current_user_id)):
    return crud_list("courses", order_by="course_id")


@app.post("/api/courses")
async def create_course(course: CourseIn, user_id: int = Depends(get_current_user_id)):
    return crud_insert("courses", course.model_dump())


@app.put("/api/courses/{course_id}")
async def update_course(course_id: int, course: CourseIn, user_id: int = Depends(get_current_user_id)):
    return crud_update("courses", "course_id", course_id, course.model_dump())


@app.delete("/api/courses/{course_id}")
async def delete_course(course_id: int, user_id: int = Depends(get_current_user_id)):
    return crud_delete("courses", "course_id", course_id)


# --- Instructors ---

@app.get("/api/instructors")
async def list_instructors(user_id: int = Depends(get_current_user_id)):
    return crud_list("instructor", order_by="inst_id")


@app.post("/api/instructors")
async def create_instructor(instructor: InstructorIn, user_id: int = Depends(get_current_user_id)):
    return crud_insert("instructor", instructor.model_dump())


@app.put("/api/instructors/{inst_id}")
async def update_instructor(inst_id: int, instructor: InstructorIn, user_id: int = Depends(get_current_user_id)):
    return crud_update("instructor", "inst_id", inst_id, instructor.model_dump())


@app.delete("/api/instructors/{inst_id}")
async def delete_instructor(inst_id: int, user_id: int = Depends(get_current_user_id)):
    return crud_delete("instructor", "inst_id", inst_id)


# --- Rooms ---

@app.get("/api/rooms")
async def list_rooms(user_id: int = Depends(get_current_user_id)):
    return crud_list("room", order_by="room_id")


@app.post("/api/rooms")
async def create_room(room: RoomIn, user_id: int = Depends(get_current_user_id)):
    return crud_insert("room", room.model_dump())


@app.put("/api/rooms/{room_id}")
async def update_room(room_id: int, room: RoomIn, user_id: int = Depends(get_current_user_id)):
    return crud_update("room", "room_id", room_id, room.model_dump())


@app.delete("/api/rooms/{room_id}")
async def delete_room(room_id: int, user_id: int = Depends(get_current_user_id)):
    return crud_delete("room", "room_id", room_id)


# --- Timeslots ---

@app.get("/api/timeslots")
async def list_timeslots(user_id: int = Depends(get_current_user_id)):
    return crud_list("timeslot", order_by="timeslot_id")


@app.post("/api/timeslots")
async def create_timeslot(timeslot: TimeslotIn, user_id: int = Depends(get_current_user_id)):
    return crud_insert("timeslot", timeslot.model_dump())


@app.put("/api/timeslots/{timeslot_id}")
async def update_timeslot(timeslot_id: int, timeslot: TimeslotIn, user_id: int = Depends(get_current_user_id)):
    return crud_update("timeslot", "timeslot_id", timeslot_id, timeslot.model_dump())


@app.delete("/api/timeslots/{timeslot_id}")
async def delete_timeslot(timeslot_id: int, user_id: int = Depends(get_current_user_id)):
    return crud_delete("timeslot", "timeslot_id", timeslot_id)


# --- Prerequisites (course_req) ---

@app.get("/api/prerequisites")
async def list_prerequisites(user_id: int = Depends(get_current_user_id)):
    return crud_list("course_req")


@app.post("/api/prerequisites")
async def create_prerequisite(prereq: PrerequisiteIn, user_id: int = Depends(get_current_user_id)):
    return crud_insert("course_req", prereq.model_dump())


@app.delete("/api/prerequisites/{course_id}/{req_course_id}")
async def delete_prerequisite(course_id: int, req_course_id: int, user_id: int = Depends(get_current_user_id)):
    res = (
        supabase.table("course_req")
        .delete()
        .eq("course_id", course_id)
        .eq("req_course_id", req_course_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Prerequisite not found")
    return {"status": "deleted"}


@app.get("/api/departments")
async def list_departments(user_id: int = Depends(get_current_user_id)):
    # Table is "department" in Supabase; no assumed order_by since column name may vary
    res = supabase.table("department").select("*").execute()
    return res.data or []


@app.post("/api/departments")
async def create_department(dept: DeptIn, user_id: int = Depends(get_current_user_id)):
    return crud_insert("department", dept.model_dump())


@app.put("/api/departments/{dept_id}")
async def update_department(dept_id: int, dept: DeptIn, user_id: int = Depends(get_current_user_id)):
    return crud_update("department", "dept_id", dept_id, dept.model_dump())


@app.delete("/api/departments/{dept_id}")
async def delete_department(dept_id: int, user_id: int = Depends(get_current_user_id)):
    return crud_delete("department", "dept_id", dept_id)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "University Timetable API", "docs": "/docs", "health": "/api/health"}
