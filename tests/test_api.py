"""API integration and security tests for backend.main (FastAPI).

OFFLINE, like the rest of the suite (see conftest.py). The real app runs end to end,
routing, auth dependency, validation, the engine and the evaluator, but Supabase is
replaced by FakeSupabase, an in-memory stand-in, and the scheduling data by the small
synthetic dataset from test_algorithms.py. Nothing here can reach or write to the real
database: the fake records every write so tests can assert on it.

Tests marked `xfail(strict=True)` are KNOWN GAPS. They assert the behaviour we want
and currently fail. When someone fixes the gap the test starts passing, strict xfail
turns that into a failure, and the marker should then be removed.
"""
import os
import re
import time
import types

import pytest

# backend.database.db builds a Supabase client on import; give it harmless values
# when no .env is present. The client it builds is never used: every test swaps
# backend.main.supabase for FakeSupabase.
os.environ.setdefault("NEXT_PUBLIC_SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

import backend.main as main  # noqa: E402
from test_algorithms import make_rooms, make_sections, make_timeslots  # noqa: E402


# ── In-memory Supabase ────────────────────────────────────────────────────────

class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    """Enough of postgrest's builder for the calls backend.main makes."""

    def __init__(self, db, table):
        self.db, self.table, self.op, self.payload = db, table, "select", None
        self.filters, self.single, self.limit_n = [], False, None

    def select(self, *_a, **_k):
        return self

    def insert(self, payload):
        self.op, self.payload = "insert", payload
        return self

    def upsert(self, payload, **_k):
        self.op, self.payload = "upsert", payload
        return self

    def update(self, payload):
        self.op, self.payload = "update", payload
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, col, val):
        self.filters.append((col, val))
        return self

    def maybe_single(self):
        self.single = True
        return self

    def limit(self, n):
        self.limit_n = n
        return self

    def order(self, *_a, **_k):
        return self

    def range(self, start, end):
        self.limit_n = (start, end)
        return self

    def _rows(self):
        rows = self.db.tables.setdefault(self.table, [])
        return [r for r in rows if all(r.get(c) == v for c, v in self.filters)]

    def execute(self):
        if self.op != "select":
            self.db.writes.append((self.op, self.table, self.payload, list(self.filters)))
            if self.op in ("insert", "upsert"):
                row = dict(self.payload, **{f"{self.table}_id": 999})
                return _Result([row])
            hit = self._rows()
            return _Result([dict(r, **(self.payload or {})) for r in hit])
        rows = self._rows()
        if isinstance(self.limit_n, tuple):
            rows = rows[self.limit_n[0]:self.limit_n[1] + 1]
        elif self.limit_n is not None:
            rows = rows[:self.limit_n]
        if self.single:
            return _Result(rows[0] if rows else None)
        return _Result(rows)


class _Auth:
    TOKENS = {"tok-admin": "auth-admin", "tok-inst": "auth-inst", "tok-orphan": "auth-orphan"}
    PASSWORDS = {"admin@uni.test": "right-password"}

    def __init__(self, db):
        self.db = db

    def get_user(self, token):
        if token not in self.TOKENS:
            raise RuntimeError("invalid JWT")
        return types.SimpleNamespace(user=types.SimpleNamespace(id=self.TOKENS[token]))

    def _session(self, auth_id):
        return types.SimpleNamespace(
            user=types.SimpleNamespace(id=auth_id),
            session=types.SimpleNamespace(access_token="new-access", refresh_token="new-refresh"),
        )

    def sign_in_with_password(self, creds):
        if self.PASSWORDS.get(creds["email"]) != creds["password"]:
            raise RuntimeError("Invalid login credentials")
        return self._session("auth-admin")

    def sign_up(self, creds):
        return self._session("auth-new")

    def refresh_session(self, token):
        if token != "good-refresh":
            raise RuntimeError("Invalid Refresh Token")
        return self._session("auth-admin")


class FakeSupabase:
    def __init__(self):
        self.writes = []
        self.tables = {
            "user": [
                {"user_id": 1, "auth_id": "auth-admin", "email": "admin@uni.test", "role": "admin"},
                {"user_id": 2, "auth_id": "auth-inst", "email": "inst@uni.test", "role": "instructor"},
            ],
            "schedule": [
                {"schedule_id": 10, "user_id": 3, "name": "Base Schedule", "alg_name": "base",
                 "created_at": "2026-04-24"},
                {"schedule_id": 20, "user_id": 1, "name": "admin run", "alg_name": "greedy"},
                {"schedule_id": 21, "user_id": 2, "name": "someone else's run", "alg_name": "pso"},
            ],
            # One row each, so an update/delete by an instructor actually reaches its row
            # (a 404 "row not found" would hide whether the role was checked).
            "room": [{"room_id": 1, "building": "M8"}],
            "courses": [{"course_id": 1, "name": "x"}],
            "timeslot": [{"timeslot_id": 1, "day": "M"}],
            "department": [{"dept_id": 1, "name": "x", "college_id": 5}],
            "college": [{"college_id": 5, "name": "Engineering"}],
        }
        self.auth = _Auth(self)

    def table(self, name):
        return _Query(self, name)


# ── Fixtures ──────────────────────────────────────────────────────────────────

ADMIN = {"Authorization": "Bearer tok-admin"}
INSTRUCTOR = {"Authorization": "Bearer tok-inst"}


@pytest.fixture(scope="module")
def scheduling_data():
    return {"sections": make_sections(), "timeslots": make_timeslots(), "rooms": make_rooms()}


@pytest.fixture(scope="module")
def greedy_items(scheduling_data):
    from backend.Optimization.Algorithims.greedy import greedy_schedule
    return greedy_schedule(scheduling_data["sections"], scheduling_data["timeslots"],
                           scheduling_data["rooms"])


@pytest.fixture
def fake(monkeypatch, scheduling_data, greedy_items):
    db = FakeSupabase()
    monkeypatch.setattr(main, "supabase", db)
    monkeypatch.setattr(main, "load_scoring_data", lambda: scheduling_data)
    monkeypatch.setattr(main, "load_schedule", lambda schedule_id: list(greedy_items))

    def fake_save(items, details):
        db.writes.append(("save_schedule", "schedule", details, len(items)))
        return {"schedule": {"schedule_id": 999, "name": details["sch_name"]}, "rows_saved": len(items)}

    monkeypatch.setattr(main, "save_schedule", fake_save)
    for cache in (main._baseline_cache, main._jobs, main._scoring_cache, main._items_cache):
        cache.clear()
    yield db
    for cache in (main._baseline_cache, main._scoring_cache, main._items_cache):
        cache.clear()


@pytest.fixture
def client(fake):
    return TestClient(main.app)


# ── Authentication ────────────────────────────────────────────────────────────

PUBLIC = {"/", "/api/health", "/api/auth/login", "/api/auth/signup", "/api/auth/refresh"}


def _protected_routes():
    out = []
    for route in main.app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/") or path in PUBLIC:
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            concrete = re.sub(r"\{[^}]+\}", "1", path)
            out.append((method, concrete))
    return out


@pytest.mark.parametrize("method,path", _protected_routes())
def test_every_protected_route_rejects_a_request_without_a_token(client, method, path):
    assert client.request(method, path).status_code == 401


@pytest.mark.parametrize("header", ["tok-admin", "Basic tok-admin", "Bearer ", "Bearer not-a-real-token"])
def test_malformed_or_invalid_tokens_are_rejected(client, header):
    r = client.get("/api/schedules", headers={"Authorization": header})
    assert r.status_code == 401


def test_valid_token_without_a_profile_is_refused(client):
    r = client.get("/api/schedules", headers={"Authorization": "Bearer tok-orphan"})
    assert r.status_code == 404


def test_public_endpoints_need_no_token(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200


def test_login_with_wrong_password_is_401_and_does_not_say_which_part_was_wrong(client):
    wrong_pw = client.post("/api/auth/login", json={"email": "admin@uni.test", "password": "x"})
    no_user = client.post("/api/auth/login", json={"email": "nobody@uni.test", "password": "x"})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


def test_login_returns_role_and_both_tokens(client):
    r = client.post("/api/auth/login", json={"email": "admin@uni.test", "password": "right-password"})
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "admin" and body["access_token"] and body["refresh_token"]


def test_refresh_with_a_bad_refresh_token_is_401(client):
    assert client.post("/api/auth/refresh", json={"refresh_token": "stale"}).status_code == 401
    ok = client.post("/api/auth/refresh", json={"refresh_token": "good-refresh"})
    assert ok.status_code == 200 and ok.json()["access_token"] == "new-access"


def test_signup_always_creates_an_instructor_even_if_the_client_asks_for_admin(client, fake):
    r = client.post("/api/auth/signup",
                    json={"email": "new@uni.test", "password": "pw123456", "role": "admin"})
    assert r.status_code == 200
    inserts = [w for w in fake.writes if w[0] == "insert" and w[1] == "user"]
    assert len(inserts) == 1 and inserts[0][2]["role"] == "instructor"


# ── Optimize (synchronous) ────────────────────────────────────────────────────

def test_sync_greedy_returns_a_runnable_measured_timetable(client, scheduling_data):
    r = client.post("/api/optimize", headers=ADMIN, json={"algorithm": "greedy"})
    assert r.status_code == 200
    t = r.json()["timetable"]
    assert t["conflicts"] == 0                                   # HARD conflicts
    assert t["conflict_detail"]["hard"] == {"instructor": 0, "room": 0, "sibling": 0, "room_rules": 0}
    assert t["scheduled"] + t["unscheduled"] == len(t["schedule"])
    assert 0 < t["fitness"] <= 1
    assert t["total_sections"] == len(scheduling_data["sections"])
    assert t["preferences_applied"] is False
    row = t["schedule"][0]
    for key in ("course_id", "section", "instructor_ids", "room_id", "timeslot_id", "day", "start", "end"):
        assert key in row


def test_unknown_algorithm_is_a_400(client):
    r = client.post("/api/optimize", headers=ADMIN, json={"algorithm": "quantum"})
    assert r.status_code == 400 and "Available" in r.json()["detail"]


@pytest.mark.parametrize("runs", [0, 31, -1])
def test_num_runs_outside_1_to_30_is_rejected(client, runs):
    r = client.post("/api/optimize", headers=ADMIN, json={"algorithm": "greedy", "num_runs": runs})
    assert r.status_code == 422


# ── Optimize (background job) ─────────────────────────────────────────────────

def _poll(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}", headers=ADMIN)
        if r.status_code != 200 or r.json()["status"] != "running":
            return r
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_async_job_runs_to_completion_and_returns_the_timetable(client):
    start = client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "greedy"})
    assert start.status_code == 200 and start.json()["status"] == "running"
    done = _poll(client, start.json()["job_id"])
    assert done.status_code == 200 and done.json()["status"] == "done"
    assert done.json()["timetable"]["conflicts"] == 0


def test_a_finished_job_can_be_read_again_after_a_refresh(client):
    """The result is kept after the first read, so a reload, another page or a second tab still gets it."""
    job_id = client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "greedy"}).json()["job_id"]
    first = _poll(client, job_id)
    second = client.get(f"/api/jobs/{job_id}", headers=ADMIN)
    assert first.status_code == second.status_code == 200
    assert second.json()["timetable"]["fitness"] == first.json()["timetable"]["fitness"]


def test_job_list_shows_only_the_callers_jobs_without_their_results(client):
    mine = client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "greedy", "num_runs": 2}).json()["job_id"]
    theirs = client.post("/api/optimize/async", headers=INSTRUCTOR, json={"algorithm": "greedy"}).json()["job_id"]
    _poll(client, mine)
    listed = client.get("/api/jobs", headers=ADMIN).json()
    assert [j["job_id"] for j in listed] == [mine]
    job = listed[0]
    assert job["status"] == "done" and job["algorithm"] == "greedy" and job["num_runs"] == 2
    assert job["finished_at"] >= job["started_at"] and job["conflicts"] == 0
    assert "timetable" not in job and "result" not in job
    assert theirs not in [j["job_id"] for j in client.get("/api/jobs", headers=ADMIN).json()]


def test_finished_jobs_are_pruned_beyond_the_cap(client):
    now = time.time()
    for n in range(main.MAX_FINISHED_JOBS + 2):
        main._jobs[f"old-{n}"] = {"job_id": f"old-{n}", "status": "done", "user_id": 1, "result": {},
                                  "algorithm": "greedy", "num_runs": 1, "started_at": now - 100 + n,
                                  "finished_at": now - 100 + n}
    main._jobs["expired"] = {"job_id": "expired", "status": "done", "user_id": 1, "result": {},
                             "algorithm": "greedy", "num_runs": 1, "started_at": 0,
                             "finished_at": now - main.JOB_RESULT_TTL_S - 1}
    listed = client.get("/api/jobs", headers=ADMIN).json()
    assert len(listed) == main.MAX_FINISHED_JOBS
    assert not {"old-0", "old-1", "expired"} & set(main._jobs)       # the two oldest and the expired one


def test_unknown_job_id_is_404(client):
    assert client.get("/api/jobs/does-not-exist", headers=ADMIN).status_code == 404


def test_async_job_with_a_bad_algorithm_reports_the_error_on_poll(client):
    start = client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "quantum"})
    assert start.status_code == 200                      # accepted, fails later in the worker
    r = _poll(client, start.json()["job_id"])
    assert r.status_code == 500 and "Unknown algorithm" in r.json()["detail"]


def test_a_job_is_readable_only_by_the_user_who_started_it(client):
    job_id = client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "greedy"}).json()["job_id"]
    _poll(client, job_id)
    r = client.get(f"/api/jobs/{job_id}", headers=INSTRUCTOR)
    assert r.status_code == 404                     # same answer as a missing job: ids can't be probed


def _slow_run(stop_after=30.0):
    """A stand-in for a long algorithm: pure Python, so it can be interrupted, and done after stop_after s."""
    def run(algorithm, num_runs=1, use_preferences=False):
        end = time.time() + stop_after
        while time.time() < end:
            sum(range(1000))
        return {"fitness": 1.0, "conflicts": 0, "unscheduled": 0, "schedule": []}
    return run


def _wait(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_cancelling_a_running_job_stops_it_and_frees_the_worker(client, monkeypatch):
    monkeypatch.setattr(main, "run_optimization", _slow_run())
    job_id = client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "ga"}).json()["job_id"]
    assert _wait(lambda: main._jobs[job_id].get("running_since"))
    assert client.get(f"/api/jobs/{job_id}", headers=ADMIN).json()["queued"] is False
    started = time.time()
    r = client.post(f"/api/jobs/{job_id}/cancel", headers=ADMIN)
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    # The worker thread really stopped (it would otherwise run for 30 s), well within a second or two.
    assert _wait(lambda: main._jobs[job_id]["future"].done(), timeout=5)
    assert time.time() - started < 5
    assert main._jobs[job_id]["status"] == "cancelled" and "result" not in main._jobs[job_id]
    polled = client.get(f"/api/jobs/{job_id}", headers=ADMIN)
    assert polled.status_code == 200 and polled.json()["status"] == "cancelled" and "timetable" not in polled.json()
    # The pool still works afterwards.
    monkeypatch.setattr(main, "run_optimization", _slow_run(0.1))
    again = client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "ga"}).json()["job_id"]
    assert _poll(client, again).json()["status"] == "done"


def test_a_queued_job_can_be_cancelled_before_it_starts(client, monkeypatch):
    monkeypatch.setattr(main, "run_optimization", _slow_run(30))
    busy = [client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "ga"}).json()["job_id"] for _ in range(2)]
    queued = client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "ga"}).json()["job_id"]
    assert _wait(lambda: all(main._jobs[j].get("running_since") for j in busy))
    assert client.get(f"/api/jobs/{queued}", headers=ADMIN).json()["queued"] is True
    assert client.post(f"/api/jobs/{queued}/cancel", headers=ADMIN).json()["status"] == "cancelled"
    assert main._jobs[queued]["future"].cancelled() and not main._jobs[queued].get("running_since")
    for j in busy:                                   # tidy up: stop the two long ones
        client.post(f"/api/jobs/{j}/cancel", headers=ADMIN)
    assert _wait(lambda: all(main._jobs[j]["future"].done() for j in busy), timeout=5)


def test_only_the_owner_can_cancel_and_only_while_it_runs(client, monkeypatch):
    monkeypatch.setattr(main, "run_optimization", _slow_run(0.2))
    job_id = client.post("/api/optimize/async", headers=ADMIN, json={"algorithm": "ga"}).json()["job_id"]
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=INSTRUCTOR).status_code == 404
    _poll(client, job_id)
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=ADMIN).status_code == 409
    assert client.post("/api/jobs/nope/cancel", headers=ADMIN).status_code == 404


# ── Saved schedules ───────────────────────────────────────────────────────────

def test_list_shows_only_the_callers_own_schedules(client):
    ids = {row["schedule_id"] for row in client.get("/api/schedules", headers=ADMIN).json()}
    assert ids == {20}


def test_reading_another_users_schedule_is_403_and_a_missing_one_404(client):
    assert client.get("/api/schedules/21", headers=ADMIN).status_code == 403
    assert client.get("/api/schedules/12345", headers=ADMIN).status_code == 404


def test_deleting_another_users_schedule_is_403_and_deletes_nothing(client, fake):
    assert client.delete("/api/schedules/21", headers=ADMIN).status_code == 403
    assert not [w for w in fake.writes if w[0] == "delete"]


def test_deleting_your_own_schedule_removes_its_rows_and_its_header(client, fake):
    assert client.delete("/api/schedules/20", headers=ADMIN).status_code == 200
    deleted = [(w[1], w[3]) for w in fake.writes if w[0] == "delete"]
    assert deleted == [("schedule_detailes", [("schedule_id", 20)]), ("schedule", [("schedule_id", 20)])]


def test_the_manual_baseline_is_readable_by_every_signed_in_user(client):
    for headers in (ADMIN, INSTRUCTOR):
        row = client.get("/api/schedules/baseline", headers=headers).json()
        assert row["baseline"] is True and row["schedule_id"] == 10
        full = client.get("/api/schedules/10", headers=headers)
        assert full.status_code == 200 and full.json()["schedule"]


def test_save_ignores_the_browsers_metrics_and_stores_the_evaluators(client, fake):
    rows = client.post("/api/optimize", headers=ADMIN, json={"algorithm": "greedy"}).json()["timetable"]["schedule"]
    r = client.post("/api/schedules", headers=ADMIN, json={
        "name": "t", "algorithm": "greedy", "schedule": rows,
        "fitness": 0.123, "conflicts": 777, "scheduled": 1, "unscheduled": 1, "exec_time": 4.2,
    })
    assert r.status_code == 200 and r.json()["measured"] is True
    update = [w for w in fake.writes if w[0] == "update" and w[1] == "schedule"][-1][2]
    assert update["conflicts_count"] == 0 and update["fitness_score"] != 0.123
    assert "hard" in update["evaluation"] and "soft" in update["evaluation"]
    saved = [w for w in fake.writes if w[0] == "save_schedule"][0]
    assert saved[2]["user_id"] == 1 and saved[2]["exec_time"] == 4.2


# ── Problems and hand editing ─────────────────────────────────────────────────

def _room_clash_edit(rows):
    """An edit that moves one roomed class onto another's room and time: a certain room clash."""
    roomed = [i for i, r in enumerate(rows) if r["room_id"] is not None and r["timeslot_id"] is not None]
    a, b = roomed[0], next(i for i in roomed[1:] if rows[i]["room_id"] != rows[roomed[0]]["room_id"])
    return a, b, {"idx": b, "room_id": rows[a]["room_id"], "timeslot_id": rows[a]["timeslot_id"]}


def test_opening_a_schedule_lists_its_problems(client):
    body = client.get("/api/schedules/20", headers=ADMIN).json()
    problems = body["problems"]
    assert problems["hard"] == [] and problems["hard_pairs"] == 0      # greedy's schedule is clean
    assert all(0 <= i < len(body["schedule"]) for i in problems["unscheduled"])


def test_review_scores_unsaved_edits_and_names_the_clashing_pair(client, fake):
    rows = client.get("/api/schedules/20", headers=ADMIN).json()["schedule"]
    clean = client.post("/api/schedules/20/review", headers=ADMIN, json={"edits": []}).json()
    assert clean["conflicts"] == 0
    a, b, edit = _room_clash_edit(rows)
    clashed = client.post("/api/schedules/20/review", headers=ADMIN, json={"edits": [edit]}).json()
    assert clashed["conflicts"] >= 1 and clashed["fitness"] < clean["fitness"]
    room_pairs = [p for p in clashed["problems"]["hard"] if p["kind"] == "room"]
    assert any(set(p["items"]) == {a, b} for p in room_pairs)
    # Nothing written, and the saved copy is untouched: a fresh review is clean again.
    assert not [w for w in fake.writes if w[0] in ("insert", "update", "delete", "save_schedule")]
    assert client.post("/api/schedules/20/review", headers=ADMIN, json={"edits": []}).json()["conflicts"] == 0


def test_every_suggested_placement_keeps_the_schedule_clash_free(client):
    rows = client.get("/api/schedules/20", headers=ADMIN).json()["schedule"]
    idx = next(i for i, r in enumerate(rows) if r["room_id"] is not None and r["timeslot_id"] is not None)
    unplace = {"idx": idx, "room_id": None, "timeslot_id": None}
    result = client.post("/api/schedules/20/placements", headers=ADMIN,
                         json={"idx": idx, "edits": [unplace]}).json()
    assert result["needs"] == {"room": True, "time": True} and result["candidates"]
    costs = [t["cost"] for t in result["candidates"]]
    assert costs == sorted(costs)
    for time_option in result["candidates"][:5]:
        assert {"day", "start", "end"} <= set(time_option)
        for option in time_option["rooms"][:3]:
            place = {"idx": idx, "room_id": option["room_id"], "timeslot_id": time_option["timeslot_id"]}
            review = client.post("/api/schedules/20/review", headers=ADMIN, json={"edits": [place]}).json()
            assert review["conflicts"] == 0, (time_option, option)


def test_placements_never_suggest_a_taken_room_and_time(client):
    rows = client.get("/api/schedules/20", headers=ADMIN).json()["schedule"]
    idx = next(i for i, r in enumerate(rows) if r["room_id"] is not None and r["timeslot_id"] is not None)
    taken = {(r["room_id"], r["timeslot_id"]) for i, r in enumerate(rows) if i != idx}
    result = client.post("/api/schedules/20/placements", headers=ADMIN, json={"idx": idx}).json()
    offered = {(o["room_id"], t["timeslot_id"]) for t in result["candidates"] for o in t["rooms"]}
    assert offered and not offered & taken


def test_editing_endpoints_respect_ownership_and_validate_indexes(client):
    for path in ("/api/schedules/21/review", "/api/schedules/21/placements"):
        assert client.post(path, headers=ADMIN, json={"idx": 0, "edits": []}).status_code == 403
    assert client.post("/api/schedules/10/review", headers=INSTRUCTOR, json={"edits": []}).status_code == 200
    assert client.post("/api/schedules/20/placements", headers=ADMIN, json={"idx": 10**6}).status_code == 422
    bad = {"edits": [{"idx": 10**6, "room_id": 1, "timeslot_id": 1}]}
    assert client.post("/api/schedules/20/review", headers=ADMIN, json=bad).status_code == 422
    assert client.post("/api/schedules/20/review", headers=ADMIN,
                       json={"edits": [{"idx": -1}]}).status_code == 422


def test_colleges_are_listed_for_colouring_the_timetable(client):
    assert client.get("/api/colleges", headers=ADMIN).json() == [{"college_id": 5, "name": "Engineering"}]


def test_non_numeric_ids_are_rejected_by_validation(client):
    assert client.get("/api/schedules/1;drop table schedule", headers=ADMIN).status_code == 422


# ── Roles / authorisation (security) ──────────────────────────────────────────

ADMIN_WRITES = [
    ("POST", "/api/rooms", {"building": "M8", "room_num": "1", "capacity": 30, "room_type": "classroom"}),
    ("DELETE", "/api/rooms/1", None),
    ("POST", "/api/courses", {"course_id": 1, "name": "x"}),
    ("DELETE", "/api/courses/1", None),
    ("PUT", "/api/timeslots/1", {"day": "M", "start_time": "08:00", "end_time": "09:00"}),
    ("DELETE", "/api/departments/1", None),
]


@pytest.mark.xfail(strict=True, reason="KNOWN GAP: data-management endpoints never check the role")
@pytest.mark.parametrize("method,path,body", ADMIN_WRITES)
def test_an_instructor_cannot_change_university_data(client, method, path, body):
    r = client.request(method, path, headers=INSTRUCTOR, json=body)
    assert r.status_code == 403


@pytest.mark.xfail(strict=True, reason="KNOWN GAP: an instructor can trigger optimisation runs")
def test_an_instructor_cannot_start_an_optimisation_run(client):
    r = client.post("/api/optimize/async", headers=INSTRUCTOR, json={"algorithm": "greedy"})
    assert r.status_code == 403


def test_an_account_not_linked_to_an_instructor_has_no_preferences_page(client):
    assert client.get("/api/my/preferences", headers=ADMIN).status_code == 403


# ── CORS ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("origin,allowed", [
    ("http://localhost:3000", True),
    ("https://v0-university-timetable-tool-2.vercel.app", True),
    ("https://v0-university-timetable-tool-2-git-main-logiphic.vercel.app", True),
    ("https://evil.example.com", False),
    ("https://v0-university-timetable-tool-2.vercel.app.evil.com", False),
])
def test_cors_allows_only_the_frontend(client, origin, allowed):
    r = client.options("/api/health", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed
