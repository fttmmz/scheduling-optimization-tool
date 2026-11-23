# main.py
import ctypes
import os
import sqlite3
import tempfile
import time
import uuid
import asyncio
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.Optimization.constraints import tag_intro_it_pairs, tag_section_links
from backend.Optimization.engine import SchedulingEngine
from backend.Optimization.evaluation import (
    calculate_fitness,
    count_all_violations_flat,
    count_hard_conflicts,
    count_room_rule_violations,
    count_instructor_conflicts,
    count_room_conflicts,
    count_scheduled_sections,
    count_sibling_conflicts,
    count_timeslot_guideline_conflicts,
    build_timeslot_guideline_cache,
    soft_violation_counts,
    total_soft_penalty,
    total_time_penalty,
)
from backend.Optimization.InstructorPreferences.preference_config import (
    PreferenceRunConfig,
    get_preference_calculator,
    load_all_preference_config,
    preference_penalty,
)
from backend.database.db import make_auth_client, supabase
from backend.database.loader import get_scheduling_data, load_schedule, save_schedule
from backend.fairness_report import build_fairness_report
from backend.login_limit import release_attempt, reserve_attempt
from backend.preference_report import build_preference_report, load_instructor_names
from backend.preferences import (
    PreferenceIn,
    find_instructor,
    load_preference,
    login_email,
    save_preference,
)

from backend.schedule_editing import apply_edits, find_placements, find_problems

app = FastAPI()

# In-memory job store for async optimization runs, keyed by job_id. Each job records its owner and
# what was run, so any page can show its progress and the owner alone can read it. A finished result
# is kept (not deleted on first read) until JOB_RESULT_TTL_S passes or MAX_FINISHED_JOBS newer ones
# push it out: a browser that reloads, changes page or polls from two tabs must still find it.
# A server restart still loses every job; the frontend reports that as "lost".
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)
JOB_RESULT_TTL_S = 6 * 3600
MAX_FINISHED_JOBS = 6          # each result holds a full schedule; Render's free tier has 512 MB


def _prune_jobs() -> None:
    """Drop finished jobs past their TTL, then the oldest beyond MAX_FINISHED_JOBS. Caller holds the lock."""
    now = time.time()
    finished = sorted(
        (j for j in _jobs.values() if j["status"] != "running"),
        key=lambda j: j.get("finished_at") or 0,
        reverse=True,
    )
    for n, job in enumerate(finished):
        if n >= MAX_FINISHED_JOBS or now - (job.get("finished_at") or now) > JOB_RESULT_TTL_S:
            _jobs.pop(job["job_id"], None)


def job_summary(job: Dict[str, Any]) -> Dict[str, Any]:
    """A job without its (large) result: what the navbar indicator polls."""
    summary = {k: job.get(k) for k in ("job_id", "status", "algorithm", "num_runs", "started_at", "finished_at")}
    # Waiting for a free worker (two runs at once at most): the page says so instead of a clock
    # that seems to run for nothing.
    summary["queued"] = job["status"] == "running" and not job.get("running_since")
    summary["running_since"] = job.get("running_since")
    result = job.get("result")
    if result is not None:
        summary["fitness"] = result.get("fitness")
        summary["conflicts"] = result.get("conflicts")
        summary["unscheduled"] = result.get("unscheduled")
        summary["fitness_with_preferences"] = bool((result.get("preferences") or {}).get("included"))
    if job["status"] == "error":
        summary["error"] = job.get("error")
    return summary


# Rooms, timeslots and sections, loaded and tagged once and reused for SCORING only (opening, saving
# and editing a schedule). Reloading them cost ~3 s per request. Any write through the data-management
# endpoints clears it, and it expires after SCORING_DATA_TTL_S in case the tables were edited directly
# in Supabase. run_optimization() never uses it: an algorithm gets a fresh copy it may modify.
SCORING_DATA_TTL_S = 15 * 60
_scoring_cache: Dict[str, Any] = {}
_scoring_lock = threading.Lock()

# A saved schedule never changes after it is saved (there is no update endpoint), so its loaded items
# are reused; deleting one evicts it. Editing works on copies (schedule_editing.apply_edits).
_items_cache: "Dict[int, list]" = {}
_items_lock = threading.Lock()
MAX_CACHED_SCHEDULES = 4


def cached_scoring_data() -> Dict[str, Any]:
    with _scoring_lock:
        loaded_at = _scoring_cache.get("loaded_at")
        if loaded_at is None or time.time() - loaded_at > SCORING_DATA_TTL_S:
            data = load_scoring_data()
            data["ts_cache"] = build_timeslot_guideline_cache(data["sections"], data["timeslots"])
            _scoring_cache["data"] = data
            _scoring_cache["loaded_at"] = time.time()
        return _scoring_cache["data"]


def invalidate_scoring_data() -> None:
    with _scoring_lock:
        _scoring_cache.clear()


def cached_schedule_items(schedule_id: int) -> list:
    with _items_lock:
        if schedule_id in _items_cache:
            _items_cache[schedule_id] = _items_cache.pop(schedule_id)     # most recent last
            return _items_cache[schedule_id]
    items = load_schedule(schedule_id)
    with _items_lock:
        _items_cache[schedule_id] = items
        while len(_items_cache) > MAX_CACHED_SCHEDULES:
            _items_cache.pop(next(iter(_items_cache)))
    return items


class TimetableRequest(BaseModel):
    algorithm: str = Field(
        "greedy",
        description="The scheduling algorithm to run. Available values: greedy, genetic, hybrid, grasp, pso, ml_greedy, rl.",
    )
    num_runs: int = Field(
        1,
        ge=1,
        le=30,
        description="Number of independent runs for genetic/hybrid/grasp/pso/ml_greedy/rl algorithms (1–30). Higher values improve quality but take longer.",
    )
    use_preferences: bool = Field(
        False,
        description="Ask the optimizer to try to meet instructors' saved preferences. Soft only: never at the cost of a hard rule or an unscheduled section.",
    )


class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


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
    # The run had instructor preferences ON (its preferences_applied), or the schedule
    # being re-saved from the editor was scored with them. The saved fitness then
    # includes the preference penalty, and schedule.evaluation.preferences says so.
    preferences_applied: bool = False


class CourseIn(BaseModel):
    course_id: int
    name: str
    credits: Optional[float] = None     # some courses really carry 0.3, 0.5 or 0.9 credits
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
    """The schedule's full metadata row, after checking it belongs to this user.

    The whole row, not just the ids: GET /api/schedules/{id} returns it as the
    schedule's metadata, and the Schedules page reads the name, algorithm,
    fitness and counts from it.
    """
    res = (
        supabase.table("schedule")
        .select("*")
        .eq("schedule_id", schedule_id)
        .maybe_single()
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Schedule not found")
    if res.data["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized for this schedule")
    return res.data


async def get_current_profile(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    token = authorization.removeprefix("Bearer ").strip()

    try:
        user_response = make_auth_client().auth.get_user(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    auth_id = str(user_response.user.id)
    profile_res = (
        supabase.table("user")
        .select("user_id, role")
        .eq("auth_id", auth_id)
        .maybe_single()
        .execute()
    )

    if not profile_res.data:
        raise HTTPException(status_code=404, detail="User profile not found")

    return profile_res.data


async def get_current_user_id(profile: Dict[str, Any] = Depends(get_current_profile)) -> int:
    return profile["user_id"]


async def get_current_admin_id(profile: Dict[str, Any] = Depends(get_current_profile)) -> int:
    if profile["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return profile["user_id"]


async def get_current_instructor(user_id: int = Depends(get_current_user_id)) -> Dict[str, Any]:
    instructor = find_instructor(supabase, user_id)
    if not instructor:
        raise HTTPException(status_code=403, detail="This account is not linked to an instructor")
    return instructor


def enrich_schedule_items(schedule_items, rooms, timeslots) -> List[Dict[str, Any]]:
    """ScheduleItems -> the JSON rows the frontend draws, with room and time details filled in.

    `instructor_ids` carries every instructor on the class. `instructor_id` is only the
    first, kept for older clients; a save that sent only it would drop the co-teachers.
    """
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
                "instructor_ids": list(item.instructor_ids),
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
    return enriched


def load_preference_inputs():
    """(saved preferences by inst_id, instructor names), or (None, None) when they can't be read.

    A run must not fail because the preference table is unreadable: it goes ahead without
    preferences and says so through preference_status.
    """
    try:
        return load_all_preference_config(supabase), load_instructor_names(supabase)
    except Exception as exc:
        print(f"[preferences] could not load saved preferences: {exc}")
        return None, None


def saved_preference_config() -> Optional[PreferenceRunConfig]:
    """Today's saved preferences as a scoring config, or None when none are saved or they can't be read.

    Used to score a schedule whose run had preferences ON, so the number shown matches what
    the run optimized. Same weight (1.0) as run_optimization() gives the engine.
    """
    try:
        preferences = load_all_preference_config(supabase)
    except Exception as exc:
        print(f"[preferences] could not load saved preferences: {exc}")
        return None
    config = PreferenceRunConfig(requested=True, weight=1.0, by_instructor=preferences,
                                 calculator=get_preference_calculator())
    return config if config.active else None


def run_optimization(algorithm: str, num_runs: int = 1, use_preferences: bool = False) -> Dict[str, Any]:
    scheduling_data = load_scoring_data()
    preferences, instructor_names = load_preference_inputs()

    # OFF passes no config at all, so a run without preferences is exactly the run it was before.
    preference_config = None
    if use_preferences and preferences is not None:
        preference_config = PreferenceRunConfig(
            requested=True, weight=1.0, by_instructor=preferences,
            calculator=get_preference_calculator(),
        )
    if not use_preferences:
        preference_status = "off"
    elif preference_config is None:
        preference_status = "load_failed"
    else:
        preference_status = preference_config.status

    try:
        engine = SchedulingEngine(algorithm, num_runs=num_runs, preference_config=preference_config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    t0 = time.perf_counter()
    schedule_items, scheduled, unscheduled_count = engine.run(scheduling_data)
    exec_time = time.perf_counter() - t0

    # A run that used preferences is scored with them, the same as it will be once saved.
    evaluation = evaluate_schedule(
        schedule_items, scheduling_data,
        preference_config=preference_config if preference_config and preference_config.active else None,
    )
    # The engine counts with the same count_scheduled_sections(); its figures are kept
    # so this response cannot drift from what the engine printed for the run.
    evaluation["scheduled"] = scheduled
    evaluation["unscheduled"] = unscheduled_count

    return {
        "schedule": enrich_schedule_items(schedule_items, scheduling_data["rooms"], scheduling_data["timeslots"]),
        **evaluation,
        "exec_time": exec_time,
        "preferences_requested": use_preferences,
        # True only when the preferences could change the search: requested, loaded, and not empty.
        "preferences_applied": bool(preference_config and preference_config.active),
        "preference_status": preference_status,
        # How well the result meets the saved preferences, ON or OFF, so the two can be compared.
        # Measured on the run's own items; hand edits made afterwards are measured once saved.
        "preference_report": (
            build_preference_report(schedule_items, scheduling_data["timeslots"], preferences, instructor_names)
            if preferences is not None else None
        ),
        "fairness": measure_fairness(schedule_items, scheduling_data["timeslots"], preferences, instructor_names),
    }


def measure_fairness(schedule_items, timeslots, preferences, names):
    """Workload and preference fairness, or None if measuring fails: an extra never fails a run."""
    try:
        return build_fairness_report(schedule_items, timeslots, preferences, names)
    except Exception as exc:
        print(f"[fairness] could not measure fairness: {exc}")
        return None


def load_scoring_data() -> Dict[str, Any]:
    """Rooms, timeslots and sections, prepared exactly as run_optimization() prepares them."""
    scheduling_data = get_scheduling_data()
    if scheduling_data is None:
        raise RuntimeError("Unable to load scheduling data from the database.")
    tag_section_links(scheduling_data["sections"])
    tag_intro_it_pairs(scheduling_data["sections"])
    return scheduling_data


def evaluate_schedule(schedule_items, scheduling_data: Dict[str, Any],
                      preference_config: Optional[PreferenceRunConfig] = None) -> Dict[str, Any]:
    """Every number the frontend shows about a schedule, measured by the backend's evaluator.

    The one place a schedule is scored, whether it was just generated, saved after
    hand edits, or is the manual base schedule. *schedule_items* must be full
    ScheduleItems (level, course_class, campus, enrolment, instructor_ids): items built
    from what a browser posts lack those and would be judged by the fallback rules,
    which is why a save is scored on the rows reloaded with load_schedule().

    *preference_config* (active, from a run with preferences ON) adds the instructor
    preference penalty to the soft tier of `fitness`, as the run's own fitness did.
    `preferences` in the result says whether it was included, so the number is never
    read as comparable to a schedule scored without them.
    """
    rooms = scheduling_data["rooms"]
    timeslots = scheduling_data["timeslots"]
    sections = scheduling_data["sections"]

    # Depends only on sections and timeslots (~3 s to build), so cached_scoring_data() keeps one.
    ts_cache = scheduling_data.get("ts_cache") or build_timeslot_guideline_cache(sections, timeslots)

    # Reported in TIERS, mirroring the tier note in constraints/rooms.py and evaluation.py.
    # This used to be one flat dict in which `instructor` and `room` sat as
    # equals beside `campus` and `capacity`, and `conflicts` was the flat
    # flat total -- so the UI presented a double-booking as no
    # worse than a campus mismatch, which is the thing the hard/soft split
    # exists to stop the ENGINE doing. The engine was rebuilt in August 2026;
    # this layer had not caught up, and had since fallen a further three rules
    # behind (siblings, and the two time rules).
    #
    #   HARD  physically impossible -- a room, an instructor or a group of
    #         students required in two places at once. Never acceptable, and
    #         the only number that answers "can this timetable be run".
    #   SOFT  bent preferences. Real, priced into the objective, and never a
    #         reason to leave a section unscheduled.
    hard_detail = {
        "instructor": count_instructor_conflicts(schedule_items),
        "room": count_room_conflicts(schedule_items),
        "sibling": count_sibling_conflicts(schedule_items),
        # Classes in a forbidden room (lecture in a lab, lab in another
        # department's lab, wrong side of the medical campus line).
        "room_rules": count_room_rule_violations(schedule_items, rooms),
    }
    soft_detail = soft_violation_counts(schedule_items, rooms, timeslots)
    soft_detail["timeslot_guidelines"] = count_timeslot_guideline_conflicts(
        schedule_items, sections, timeslots, ts_cache
    )

    hard_total = count_hard_conflicts(schedule_items, rooms)
    room_penalty = total_soft_penalty(schedule_items, rooms)
    time_penalty = total_time_penalty(schedule_items, timeslots)

    conflict_detail = {
        "hard": hard_detail,
        "soft": soft_detail,
        # Flat copy so an older client keeps rendering. Do not add to it; new
        # rules belong in the tier they actually sit in.
        **hard_detail,
        **soft_detail,
    }

    fitness_score = calculate_fitness(
        schedule_items, rooms, sections=sections, timeslots=timeslots, valid_timeslot_cache=ts_cache,
        preference_config=preference_config,
    )
    if preference_config is not None and preference_config.active:
        preference_note = {
            "included": True,
            "penalty": round(preference_penalty(schedule_items, timeslots, preference_config), 2),
            # Measured against the preferences saved at this moment; they can change later.
            "measured_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        preference_note = {"included": False}

    # Same formula as SchedulingEngine.run().
    scheduled = count_scheduled_sections(schedule_items, sections)

    return {
        "scheduled": scheduled,
        "unscheduled": len(schedule_items) - scheduled,
        # `conflicts` now means HARD conflicts -- the count that decides whether
        # the timetable can be run at all. It used to be the flat total, which
        # mixed in soft violations and so never reached zero, leaving the UI
        # permanently red for schedules that were in fact perfectly runnable.
        "conflicts": hard_total,
        "conflicts_flat": count_all_violations_flat(
            schedule_items, rooms, sections, timeslots, valid_timeslot_cache=ts_cache
        ),
        "conflict_detail": conflict_detail,
        "soft_penalty": {
            "room": round(room_penalty, 3),
            "time": round(time_penalty, 3),
            "total": round(room_penalty + time_penalty, 3),
        },
        "fitness": fitness_score,
        "preferences": preference_note,
        "total_sections": len(sections),
    }


def crud_list(table: str, order_by: Optional[str] = None):
    """Every row of a table. One select returns at most 1,000 rows, so this reads page by page.

    order_by is one column or several separated by commas. It must identify a row uniquely, or a row can
    land on two pages or on none.
    """
    rows, offset, page_size = [], 0, 1000
    while True:
        query = supabase.table(table).select("*")
        for column in (order_by or "").split(","):
            if column:
                query = query.order(column)
        page = query.range(offset, offset + page_size - 1).execute().data
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def crud_insert(table: str, data: Dict[str, Any]):
    data = {k: v for k, v in data.items() if v is not None}
    res = supabase.table(table).insert(data).execute()
    invalidate_scoring_data()
    if not res.data:
        raise HTTPException(status_code=400, detail=f"Failed to insert into {table}")
    return res.data[0]


def crud_update(table: str, id_col: str, id_val: Any, data: Dict[str, Any]):
    data = {k: v for k, v in data.items() if k != id_col}
    res = supabase.table(table).update(data).eq(id_col, id_val).execute()
    invalidate_scoring_data()
    if not res.data:
        raise HTTPException(status_code=404, detail=f"{table} row not found")
    return res.data[0]


def crud_delete(table: str, id_col: str, id_val: Any):
    res = supabase.table(table).delete().eq(id_col, id_val).execute()
    invalidate_scoring_data()
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


LOGIN_LIMIT_DB = os.environ.get(
    "LOGIN_LIMIT_DB", os.path.join(tempfile.gettempdir(), "timetable-login-limit.sqlite3")
)
LOGIN_LIMIT_MESSAGE = "Too many login attempts. Try again in a few minutes."


@app.post("/api/auth/login")
def login(request: LoginRequest, http_request: Request):
    email = login_email(request.email)
    client_ip = http_request.client.host if http_request.client else ""
    try:
        reservation, retry_after = reserve_attempt(LOGIN_LIMIT_DB, email, client_ip)
    except sqlite3.Error:
        raise HTTPException(status_code=503, detail="Login temporarily unavailable") from None
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=LOGIN_LIMIT_MESSAGE,
            headers={"Retry-After": str(retry_after)},
        )

    keep_attempt = False
    try:
        auth_response = make_auth_client().auth.sign_in_with_password(
            {"email": email, "password": request.password}
        )
        if not auth_response.user:
            keep_attempt = True
            raise HTTPException(status_code=401, detail="Invalid login or password")
    except HTTPException:
        raise
    except Exception as error:
        code = getattr(error, "code", None)
        status = getattr(error, "status", None)
        if status == 429 or code == "over_request_rate_limit":
            raise HTTPException(status_code=429, detail=LOGIN_LIMIT_MESSAGE) from None
        keep_attempt = code == "invalid_credentials"
        raise HTTPException(status_code=401, detail="Invalid login or password") from None
    finally:
        if not keep_attempt:
            try:
                release_attempt(LOGIN_LIMIT_DB, reservation)
            except sqlite3.Error:
                raise HTTPException(status_code=503, detail="Login temporarily unavailable") from None

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
        "refresh_token": auth_response.session.refresh_token,
    }


@app.post("/api/auth/signup")
async def signup(request: SignupRequest):
    try:
        auth_response = make_auth_client().auth.sign_up(
            {"email": request.email, "password": request.password}
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not auth_response.user:
        raise HTTPException(status_code=400, detail="Could not create account")

    auth_id = str(auth_response.user.id)
    # The app has two roles, admin and instructor. Signup used to give "student",
    # which the app does not have, and the frontend sent such accounts to the admin
    # pages. A new account is an instructor: it sees only the Preferences page, which
    # works once an admin links the account to an instructor record (instructor.user_id).
    profile_res = (
        supabase.table("user")
        .insert({"auth_id": auth_id, "email": request.email, "role": "instructor"})
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
            "refresh_token": None,
            "message": "Account created. Please check your email to confirm before logging in.",
        }

    return {
        "user_id": profile["user_id"],
        "email": profile["email"],
        "role": profile["role"],
        "access_token": auth_response.session.access_token,
        "refresh_token": auth_response.session.refresh_token,
    }


@app.post("/api/auth/refresh")
async def refresh_token(request: RefreshRequest):
    """Exchange a refresh token for a new access/refresh token pair.

    Access tokens expire (commonly after 1 hour); long-running optimization
    jobs poll for longer than that, so the frontend calls this to stay
    logged in instead of being kicked out mid-run.
    """
    try:
        auth_response = make_auth_client().auth.refresh_session(request.refresh_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    if not auth_response or not auth_response.session:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    return {
        "access_token": auth_response.session.access_token,
        "refresh_token": auth_response.session.refresh_token,
    }


@app.post("/api/optimize")
async def optimize(
    request: TimetableRequest,
    user_id: int = Depends(get_current_admin_id),
):
    """Synchronous optimize — fine for greedy and quick single-run genetic/hybrid."""
    result = run_optimization(
        request.algorithm, num_runs=request.num_runs, use_preferences=request.use_preferences
    )
    return {"status": "success", "timetable": result}


class JobCancelled(BaseException):
    """Raised inside a job's worker thread to stop it (see cancel_job).

    A BaseException, not an Exception, so the algorithms' own `except Exception` blocks can't
    swallow it and carry on.
    """


def _interrupt_thread(thread_id: int) -> bool:
    """Make *thread_id* raise JobCancelled at its next Python instruction (CPython only).

    Python has no way to kill a thread. This is the standard workaround: the exception is
    delivered the next time the thread runs Python code, so an algorithm stops within a moment,
    and a long C call (a database read, an XGBoost prediction) is allowed to finish first.
    `with` blocks and `finally` clauses still run on the way out, so locks are released.
    """
    hit = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), ctypes.py_object(JobCancelled))
    if hit > 1:     # never expected: undo rather than interrupt threads we didn't mean to
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(thread_id), None)
        return False
    return hit == 1


def _run_job(job_id: str, algorithm: str, num_runs: int, use_preferences: bool = False) -> None:
    """Worker function executed in a thread pool for async optimization jobs.

    Cancellation safety: cancel_job() interrupts this thread only while job["thread_id"] is set,
    and sets and reads it under job["lock"]. So an interrupt can only be pending while this
    function is still running, and the loop at the end of the try block makes sure a pending
    one is raised here, not later in the pool's own code where it would kill the worker.
    """
    job = _jobs[job_id]
    try:
        with job["lock"]:
            if job.get("cancel_requested"):
                raise JobCancelled
            job["thread_id"] = threading.get_ident()
            job["running_since"] = time.time()
        try:
            result = run_optimization(algorithm, num_runs=num_runs, use_preferences=use_preferences)
        finally:
            with job["lock"]:
                job["thread_id"] = None
        update = {"status": "done", "result": result}
        for _ in range(64):     # backward jumps: CPython delivers any pending interrupt here
            pass
    except JobCancelled:
        update = {"status": "cancelled"}
    except HTTPException as exc:
        update = {"status": "error", "error": str(exc.detail)}
    except Exception as exc:
        update = {"status": "error", "error": str(exc)}
    if job.get("cancel_requested"):
        # Cancelled while it was finishing: the person asked not to have it.
        update = {"status": "cancelled"}
    with _jobs_lock:
        job.update(update, finished_at=time.time())
        _prune_jobs()


@app.post("/api/optimize/async")
async def optimize_async(
    request: TimetableRequest,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(get_current_admin_id),
):
    """
    Async optimize — starts the job in a background thread and returns a job_id
    immediately. Use GET /api/jobs/{job_id} to poll for the result, and
    POST /api/jobs/{job_id}/cancel to stop it.
    """
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id, "status": "running", "user_id": user_id,
            "algorithm": request.algorithm, "num_runs": request.num_runs,
            "started_at": time.time(), "finished_at": None,
            "lock": threading.Lock(), "thread_id": None, "running_since": None,
        }
        # Stored so a job still waiting for a free worker can be cancelled before it starts.
        _jobs[job_id]["future"] = _executor.submit(
            _run_job, job_id, request.algorithm, request.num_runs, request.use_preferences
        )
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs")
async def list_jobs(user_id: int = Depends(get_current_user_id)):
    """This user's jobs, newest first, without their results: what the navbar indicator polls."""
    with _jobs_lock:
        _prune_jobs()
        mine = [job_summary(j) for j in _jobs.values() if j.get("user_id") == user_id]
    return sorted(mine, key=lambda j: j["started_at"] or 0, reverse=True)


def _own_job(job_id: str, user_id: int) -> Dict[str, Any]:
    job = _jobs.get(job_id)
    # Someone else's job answers exactly like a missing one, so job ids can't be probed.
    if job is None or job.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, user_id: int = Depends(get_current_user_id)):
    """Poll an async optimization job. The result stays readable until it expires (see _jobs)."""
    job = _own_job(job_id, user_id)
    if job["status"] in ("running", "cancelled"):
        return job_summary(job)
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"])
    return {**job_summary(job), "timetable": job["result"]}


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, user_id: int = Depends(get_current_admin_id)):
    """Stop a run. A queued one never starts; a running one is interrupted (see _interrupt_thread).

    The job reads "cancelled" straight away. Its worker is free again as soon as the algorithm
    reaches its next Python instruction, normally well under a second.
    """
    job = _own_job(job_id, user_id)
    if job["status"] != "running":
        raise HTTPException(status_code=409, detail=f"This run has already {job['status'] if job['status'] != 'done' else 'finished'}.")
    with job["lock"]:
        first_request = not job.get("cancel_requested")
        job["cancel_requested"] = True
        future = job.get("future")
        if future is not None and future.cancel():
            pass                                       # still queued: it will never run
        elif first_request and job.get("thread_id") is not None:
            _interrupt_thread(job["thread_id"])        # at most once per job
    with _jobs_lock:
        job.update(status="cancelled", finished_at=time.time())
        job.pop("result", None)
    return job_summary(job)


def schedule_metrics_row(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """The evaluation's numbers under the `schedule` table's column names."""
    return {
        "fitness_score": evaluation["fitness"],
        "conflicts_count": evaluation["conflicts"],
        "Scheduled": evaluation["scheduled"],
        "unscheduled": evaluation["unscheduled"],
    }


def evaluation_breakdown(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """The per-rule detail Compare charts, stored in `schedule.evaluation` (jsonb).

    Only the tiered counts: the flat copies in conflict_detail exist for old
    clients of /api/optimize and would just duplicate them here.
    """
    detail = evaluation.get("conflict_detail") or {}
    return {
        "hard": detail.get("hard", {}),
        "soft": detail.get("soft", {}),
        "soft_penalty": evaluation.get("soft_penalty"),
        "total_sections": evaluation.get("total_sections"),
        # {"included": bool, ...}: whether fitness_score carries the preference penalty.
        "preferences": evaluation.get("preferences") or {"included": False},
    }


# A plain `def`, not `async def`: FastAPI runs it in a worker thread, so the few
# seconds of reloading and scoring don't freeze every other request.
@app.post("/api/schedules")
def create_schedule(
    request: ScheduleSaveRequest,
    user_id: int = Depends(get_current_admin_id),
):
    # fitness / conflicts / scheduled / unscheduled from the request are ignored:
    # after a hand edit they are the browser's guess, not a measurement. The
    # backend scores the saved rows itself below. exec_time is kept; only the
    # client saw how long the run took end to end.
    schedule_detailes = {
        "sch_name": request.name,
        "alg": request.algorithm,
        "semester": request.semester,
        "fitness_score": None,
        "conflicts": None,
        "exec_time": request.exec_time,
        "user_id": user_id,
        "rule_set": None,
        "Scheduled": None,
        "unscheduled": None,
    }

    class _Item:
        def __init__(self, d):
            self.course_id = d.get("course_id")
            self.section = d.get("section")
            self.instructor_id = d.get("instructor_id")
            # Every instructor on the class. An older client sends only instructor_id.
            ids = d.get("instructor_ids")
            if ids is None:
                ids = [self.instructor_id] if self.instructor_id is not None else []
            self.instructor_ids = list(ids)
            self.room_id = d.get("room_id")
            self.timeslot_id = d.get("timeslot_id")
            self.capacity = d.get("capacity")

    schedule_items = [_Item(item) for item in request.schedule]

    try:
        result = save_schedule(schedule_items, schedule_detailes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Score what was actually stored: reload it the way any saved schedule is
    # read, so each item regains level, course_class, campus and enrolment from
    # the database. The numbers saved are then the numbers a later reload gives.
    schedule_id = result["schedule"]["schedule_id"]
    # Preferences that can't be read (or none saved) score it without them; the
    # stored `preferences.included: false` then says what the number means.
    preference_config = saved_preference_config() if request.preferences_applied else None
    try:
        evaluation = evaluate_schedule(cached_schedule_items(schedule_id), cached_scoring_data(),
                                       preference_config=preference_config)
        metrics = schedule_metrics_row(evaluation)
    except Exception as exc:
        # The rows are saved; only the measurement failed. Say so rather than
        # store a guess. The schedule shows "—" until it is measured.
        return {"status": "success", **result, "measured": False, "measure_error": str(exc)}

    # The per-rule breakdown goes in the `evaluation` jsonb column, added by
    # hand in Supabase (alter table public.schedule add column evaluation jsonb).
    # Without the column the update is rejected; the headline numbers are then
    # saved on their own, so a missing column never loses the measurement.
    breakdown_saved = True
    try:
        supabase.table("schedule").update(
            {**metrics, "evaluation": evaluation_breakdown(evaluation)}
        ).eq("schedule_id", schedule_id).execute()
    except Exception:
        breakdown_saved = False
        supabase.table("schedule").update(metrics).eq("schedule_id", schedule_id).execute()

    result["schedule"].update(metrics)
    if breakdown_saved:
        result["schedule"]["evaluation"] = evaluation_breakdown(evaluation)
    return {"status": "success", **result, "measured": True, "breakdown_saved": breakdown_saved,
            "evaluation": evaluation}


# The manual schedule the registrar actually ran: the feasibility reference and
# the baseline every algorithm is compared with (see CLAUDE.md, "Project scope").
BASE_SCHEDULE_ID = 10

# Schedule 10 never changes, so it is loaded and scored once per server process.
# Compare asks for its metrics and its rows at the same moment; the lock makes the
# second request wait for the first instead of doing the same 20 seconds of work.
_baseline_cache: Dict[str, Any] = {}
_baseline_lock = threading.Lock()


def baseline() -> Dict[str, Any]:
    """{"row": schedule-row-shaped metrics, "items": enriched rows} for the manual schedule."""
    with _baseline_lock:
        if "row" not in _baseline_cache:
            res = (
                supabase.table("schedule")
                .select("schedule_id, name, alg_name, created_at")
                .eq("schedule_id", BASE_SCHEDULE_ID)
                .maybe_single()
                .execute()
            )
            if not res or not res.data:
                raise HTTPException(status_code=404, detail="Base schedule not found")
            items = cached_schedule_items(BASE_SCHEDULE_ID)
            data = cached_scoring_data()
            evaluation = evaluate_schedule(items, data)
            _baseline_cache["items"] = enrich_schedule_items(items, data["rooms"], data["timeslots"])
            _baseline_cache["problems"] = find_problems(items, data["rooms"], data["sections"])
            _baseline_cache["row"] = {
                **res.data,
                **schedule_metrics_row(evaluation),
                "evaluation": evaluation_breakdown(evaluation),
                "exec_time": None,
                "baseline": True,
            }
        return _baseline_cache


# Declared before /api/schedules/{schedule_id}, which would otherwise try to read
# "baseline" as an id. A plain `def` for the same reason as create_schedule().
@app.get("/api/schedules/baseline")
def get_baseline_schedule(user_id: int = Depends(get_current_user_id)):
    """The manual base schedule's metrics, measured by the same evaluator as every saved schedule.

    Shaped like a `schedule` row so the frontend reads it with the same code. It
    belongs to another account, so it is read here without the owner check; it is
    never modified. Its rows come from GET /api/schedules/{BASE_SCHEDULE_ID}.
    """
    return baseline()["row"]


# The stored `schedule` row for the manual schedule holds numbers from an older evaluator
# (and may hold none at all). Its numbers always come from baseline() instead, so the list,
# the Dashboard and Compare agree with what opening the schedule shows.
BASELINE_ROW_FIELDS = ("fitness_score", "conflicts_count", "Scheduled", "unscheduled", "evaluation", "exec_time")


def with_current_baseline_numbers(rows: list) -> list:
    if not any(row.get("schedule_id") == BASE_SCHEDULE_ID for row in rows):
        return rows
    fresh = baseline()["row"]
    return [{**row, **{k: fresh.get(k) for k in BASELINE_ROW_FIELDS}, "baseline": True}
            if row.get("schedule_id") == BASE_SCHEDULE_ID else row
            for row in rows]


# A plain `def`: when the manual schedule is in the list and not yet scored, baseline()
# takes several seconds and must not block the event loop.
@app.get("/api/schedules")
def list_schedules(user_id: int = Depends(get_current_user_id)):
    res = (
        supabase.table("schedule")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return with_current_baseline_numbers(res.data or [])


@app.on_event("startup")
def warm_baseline() -> None:
    """Score the manual schedule in the background, so the first list or Compare after a
    restart doesn't wait for it. A failure here only means the first request scores it."""
    def run():
        try:
            baseline()
        except Exception as exc:  # noqa: BLE001 - logged, the request path retries
            print(f"[baseline] warm-up failed: {exc}")
    threading.Thread(target=run, name="baseline-warmup", daemon=True).start()


@app.get("/api/schedules/{schedule_id}")
def get_schedule(schedule_id: int, user_id: int = Depends(get_current_user_id)):
    # The manual schedule is readable by every signed-in user (Compare's side A)
    # though it belongs to another account; it is served from the cache, read-only.
    # `problems` names the items behind the hard-conflict and unscheduled counts (by index into
    # `schedule`), so the page can say which classes they are.
    if schedule_id == BASE_SCHEDULE_ID:
        cached = baseline()
        return {"metadata": cached["row"], "schedule": cached["items"], "problems": cached["problems"]}

    metadata = get_schedule_for_user(schedule_id, user_id)
    schedule_items = cached_schedule_items(schedule_id)

    data = cached_scoring_data()
    enriched = enrich_schedule_items(schedule_items, data["rooms"], data["timeslots"])

    return {"metadata": metadata, "schedule": enriched,
            "problems": find_problems(schedule_items, data["rooms"], data["sections"])}


def readable_schedule_items(schedule_id: int, user_id: int) -> list:
    """The saved items of a schedule this user may read: their own, or the manual base schedule."""
    if schedule_id != BASE_SCHEDULE_ID:
        get_schedule_for_user(schedule_id, user_id)
    return cached_schedule_items(schedule_id)


class ScheduleEdit(BaseModel):
    idx: int = Field(..., ge=0)
    room_id: Optional[int] = None
    timeslot_id: Optional[int] = None


class ReviewRequest(BaseModel):
    edits: List[ScheduleEdit] = Field(default_factory=list, max_length=5000)


class PlacementRequest(ReviewRequest):
    idx: int = Field(..., ge=0)


def scored_with_preferences(schedule_id: int) -> bool:
    """True when the saved schedule's fitness includes instructor preferences (never the manual one)."""
    if schedule_id == BASE_SCHEDULE_ID:
        return False
    res = supabase.table("schedule").select("evaluation").eq("schedule_id", schedule_id).maybe_single().execute()
    evaluation = (res.data or {}).get("evaluation") if res else None
    return bool(((evaluation or {}).get("preferences") or {}).get("included"))


def _edited_items(schedule_id: int, user_id: int, edits: List[ScheduleEdit]) -> list:
    try:
        return apply_edits(readable_schedule_items(schedule_id, user_id), [e.model_dump() for e in edits])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# Plain `def`s: scoring takes a second or two and must not block the event loop.
@app.post("/api/schedules/{schedule_id}/review")
def review_schedule(schedule_id: int, request: ReviewRequest, user_id: int = Depends(get_current_admin_id)):
    """Score a saved schedule with unsaved hand edits applied, and list what is still wrong.

    Nothing is written: the Edit page calls this after every change so its numbers are the
    evaluator's, not the browser's guess. Saving still goes through POST /api/schedules.
    """
    items = _edited_items(schedule_id, user_id, request.edits)
    data = cached_scoring_data()
    # Scored the way the saved schedule was, so a hand edit doesn't jump between the two scales.
    preference_config = saved_preference_config() if scored_with_preferences(schedule_id) else None
    evaluation = evaluate_schedule(items, data, preference_config=preference_config)
    return {
        "fitness": evaluation["fitness"],
        "conflicts": evaluation["conflicts"],
        "unscheduled": evaluation["unscheduled"],
        "scheduled": evaluation["scheduled"],
        "total_sections": evaluation["total_sections"],
        "evaluation": evaluation_breakdown(evaluation),
        "problems": find_problems(items, data["rooms"], data["sections"]),
    }


@app.get("/api/schedules/{schedule_id}/preferences")
def get_schedule_preference_report(schedule_id: int, user_id: int = Depends(get_current_user_id)):
    """How well a saved schedule meets each instructor's preferences, as they are saved now.

    Measured on request rather than stored with the schedule, because instructors can change
    their preferences after it was made. A plain `def`: the preference and name reads block.
    """
    items = readable_schedule_items(schedule_id, user_id)
    preferences, names = load_preference_inputs()
    if preferences is None:
        raise HTTPException(status_code=503, detail="Saved preferences could not be read")
    return build_preference_report(items, cached_scoring_data()["timeslots"], preferences, names)


@app.get("/api/schedules/{schedule_id}/fairness")
def get_schedule_fairness(schedule_id: int, user_id: int = Depends(get_current_user_id)):
    """How evenly a saved schedule spreads teaching load and preference results across instructors.

    Readable for the user's own schedules and the manual one, like GET /api/schedules/{id}.
    `preferences` is null when saved preferences can't be read or none exist; workload is
    measured either way.
    """
    items = readable_schedule_items(schedule_id, user_id)
    preferences, names = load_preference_inputs()
    return build_fairness_report(items, cached_scoring_data()["timeslots"], preferences, names)


@app.post("/api/schedules/{schedule_id}/placements")
def schedule_placements(schedule_id: int, request: PlacementRequest, user_id: int = Depends(get_current_admin_id)):
    """Every room and time where one class could go without a hard conflict, cheapest soft cost first.

    Checked against the schedule as edited so far (request.edits), with the class itself left out,
    so it works both for placing an unscheduled class and for moving a clashing one.
    """
    items = _edited_items(schedule_id, user_id, request.edits)
    if request.idx >= len(items):
        raise HTTPException(status_code=422, detail=f"No item {request.idx} in this schedule")
    data = cached_scoring_data()
    result = find_placements(items, request.idx, data["rooms"], data["timeslots"], data["sections"])
    rooms = {r.id: r for r in data["rooms"]}
    slots = {t.id: t for t in data["timeslots"]}
    for time_option in result["candidates"]:
        ts = slots[time_option["timeslot_id"]]
        time_option.update(day=ts.day, start=ts.start, end=ts.end)
        for option in time_option["rooms"]:
            room = rooms.get(option["room_id"])
            if room is not None:
                option.update(room_no=room.no, room_building=room.building,
                              room_type=room.type, room_capacity=room.capacity)
    return result


@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: int, user_id: int = Depends(get_current_admin_id)):
    get_schedule_for_user(schedule_id, user_id)
    supabase.table("schedule_detailes").delete().eq("schedule_id", schedule_id).execute()
    supabase.table("schedule").delete().eq("schedule_id", schedule_id).execute()
    with _items_lock:
        _items_cache.pop(schedule_id, None)
    return {"status": "deleted"}


# --- The schedule instructors see ---
# At most one schedule is published at a time (a partial unique index on is_published
# enforces it in the database). Publishing another one takes the flag off the old one
# first. Only the schedule's owner can publish or withdraw it; instructors own none.

@app.put("/api/schedules/{schedule_id}/publish")
def publish_schedule(schedule_id: int, user_id: int = Depends(get_current_admin_id)):
    get_schedule_for_user(schedule_id, user_id)
    supabase.table("schedule").update({"is_published": False, "published_at": None}).eq("is_published", True).execute()
    res = (
        supabase.table("schedule")
        .update({"is_published": True, "published_at": datetime.now(timezone.utc).isoformat()})
        .eq("schedule_id", schedule_id)
        .execute()
    )
    return res.data[0] if res.data else {"schedule_id": schedule_id, "is_published": True}


@app.delete("/api/schedules/{schedule_id}/publish")
def unpublish_schedule(schedule_id: int, user_id: int = Depends(get_current_admin_id)):
    get_schedule_for_user(schedule_id, user_id)
    supabase.table("schedule").update({"is_published": False, "published_at": None}).eq("schedule_id", schedule_id).execute()
    return {"schedule_id": schedule_id, "is_published": False}


# --- Courses ---

@app.get("/api/courses")
async def list_courses(user_id: int = Depends(get_current_user_id)):
    return crud_list("courses", order_by="course_id")


@app.post("/api/courses")
async def create_course(course: CourseIn, user_id: int = Depends(get_current_admin_id)):
    return crud_insert("courses", course.model_dump())


@app.put("/api/courses/{course_id}")
async def update_course(course_id: int, course: CourseIn, user_id: int = Depends(get_current_admin_id)):
    return crud_update("courses", "course_id", course_id, course.model_dump())


@app.delete("/api/courses/{course_id}")
async def delete_course(course_id: int, user_id: int = Depends(get_current_admin_id)):
    return crud_delete("courses", "course_id", course_id)


# --- Instructors ---

@app.get("/api/instructors")
async def list_instructors(user_id: int = Depends(get_current_user_id)):
    return crud_list("instructor", order_by="inst_id")


@app.post("/api/instructors")
async def create_instructor(instructor: InstructorIn, user_id: int = Depends(get_current_admin_id)):
    return crud_insert("instructor", instructor.model_dump())


@app.put("/api/instructors/{inst_id}")
async def update_instructor(inst_id: int, instructor: InstructorIn, user_id: int = Depends(get_current_admin_id)):
    return crud_update("instructor", "inst_id", inst_id, instructor.model_dump())


@app.delete("/api/instructors/{inst_id}")
async def delete_instructor(inst_id: int, user_id: int = Depends(get_current_admin_id)):
    return crud_delete("instructor", "inst_id", inst_id)


# --- An instructor's own preferences ---

@app.get("/api/my/preferences")
async def get_my_preferences(instructor: Dict[str, Any] = Depends(get_current_instructor)):
    return {
        "instructor": instructor,
        "preferences": load_preference(supabase, instructor["inst_id"]),
    }


@app.put("/api/my/preferences")
async def save_my_preferences(
    preference: PreferenceIn,
    instructor: Dict[str, Any] = Depends(get_current_instructor),
):
    return {
        "instructor": instructor,
        "preferences": save_preference(supabase, instructor["inst_id"], preference),
    }


def my_preference_result(items, timeslots, instructor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """This instructor's row of the preference report, or None if they saved none or it can't be read."""
    try:
        row = load_preference(supabase, instructor["inst_id"])
        if not row:
            return None
        key = str(instructor["inst_id"])
        pref = PreferenceIn.model_validate(row).model_dump()
        report = build_preference_report(items, timeslots, {key: pref}, {key: instructor.get("name")})
        return report["instructors"][0]
    except Exception as exc:
        print(f"[my schedule] could not judge preferences for {instructor.get('inst_id')}: {exc}")
        return None


@app.get("/api/my/schedule")
def get_my_schedule(instructor: Dict[str, Any] = Depends(get_current_instructor)):
    """The signed-in instructor's classes in the published schedule, and how their preferences fared.

    Only their own classes: every item that lists them, co-taught ones included. `schedule`
    is null while nothing is published. A plain `def`: the schedule and preference reads block.
    """
    res = (
        supabase.table("schedule")
        .select("schedule_id, name, published_at")
        .eq("is_published", True)
        .limit(1)
        .execute()
    )
    published = res.data[0] if res.data else None
    out = {"instructor": instructor, "schedule": published, "classes": [], "preferences": None}
    if not published:
        return out

    me = str(instructor["inst_id"])
    items = cached_schedule_items(published["schedule_id"])
    mine = [item for item in items if me in {str(i) for i in (item.instructor_ids or ())}]
    data = cached_scoring_data()
    out["classes"] = enrich_schedule_items(mine, data["rooms"], data["timeslots"])
    out["preferences"] = my_preference_result(mine, data["timeslots"], instructor)
    return out


# --- Rooms ---

@app.get("/api/rooms")
async def list_rooms(user_id: int = Depends(get_current_user_id)):
    return crud_list("room", order_by="room_id")


@app.post("/api/rooms")
async def create_room(room: RoomIn, user_id: int = Depends(get_current_admin_id)):
    return crud_insert("room", room.model_dump())


@app.put("/api/rooms/{room_id}")
async def update_room(room_id: int, room: RoomIn, user_id: int = Depends(get_current_admin_id)):
    return crud_update("room", "room_id", room_id, room.model_dump())


@app.delete("/api/rooms/{room_id}")
async def delete_room(room_id: int, user_id: int = Depends(get_current_admin_id)):
    return crud_delete("room", "room_id", room_id)


# --- Timeslots ---

@app.get("/api/timeslots")
async def list_timeslots(user_id: int = Depends(get_current_user_id)):
    return crud_list("timeslot", order_by="timeslot_id")


@app.post("/api/timeslots")
async def create_timeslot(timeslot: TimeslotIn, user_id: int = Depends(get_current_admin_id)):
    return crud_insert("timeslot", timeslot.model_dump())


@app.put("/api/timeslots/{timeslot_id}")
async def update_timeslot(timeslot_id: int, timeslot: TimeslotIn, user_id: int = Depends(get_current_admin_id)):
    return crud_update("timeslot", "timeslot_id", timeslot_id, timeslot.model_dump())


@app.delete("/api/timeslots/{timeslot_id}")
async def delete_timeslot(timeslot_id: int, user_id: int = Depends(get_current_admin_id)):
    return crud_delete("timeslot", "timeslot_id", timeslot_id)


# --- Prerequisites (course_req) ---

@app.get("/api/prerequisites")
async def list_prerequisites(user_id: int = Depends(get_current_user_id)):
    return crud_list("course_req", order_by="course_id,req_course_id")


@app.post("/api/prerequisites")
async def create_prerequisite(prereq: PrerequisiteIn, user_id: int = Depends(get_current_admin_id)):
    return crud_insert("course_req", prereq.model_dump())


@app.delete("/api/prerequisites/{course_id}/{req_course_id}")
async def delete_prerequisite(course_id: int, req_course_id: int, user_id: int = Depends(get_current_admin_id)):
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


@app.get("/api/colleges")
async def list_colleges(user_id: int = Depends(get_current_user_id)):
    """college_id -> name. department.college_id points here; the timetable colours classes by college."""
    return crud_list("college", order_by="college_id")


@app.get("/api/departments")
async def list_departments(user_id: int = Depends(get_current_user_id)):
    # Table is "department" in Supabase; no assumed order_by since column name may vary
    res = supabase.table("department").select("*").execute()
    return res.data or []


@app.post("/api/departments")
async def create_department(dept: DeptIn, user_id: int = Depends(get_current_admin_id)):
    return crud_insert("department", dept.model_dump())


@app.put("/api/departments/{dept_id}")
async def update_department(dept_id: int, dept: DeptIn, user_id: int = Depends(get_current_admin_id)):
    return crud_update("department", "dept_id", dept_id, dept.model_dump())


@app.delete("/api/departments/{dept_id}")
async def delete_department(dept_id: int, user_id: int = Depends(get_current_admin_id)):
    return crud_delete("department", "dept_id", dept_id)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "University Timetable API", "docs": "/docs", "health": "/api/health"}
