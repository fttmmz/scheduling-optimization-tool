# Project Layout

- **Backend** (this repo, GA / hybrid scheduling optimization algorithm): `D:\uniDB\scheduling-optimization-tool`
  - Deployed via **Render**, under the **fttmz** account.
- **Frontend**: `D:\UCT frontend\v0-university-timetable-tool-2`
  - Deployed via **Vercel**, under the **logiphic** account.

The frontend calls this backend's `/api/optimize` endpoint (see the frontend's `NEXT_PUBLIC_API_URL` and `lib/api.ts`).

## Long-running jobs (GA / hybrid)

GA and hybrid runs can take 20+ minutes to over an hour (30 runs), which is why the optimize flow is split:
- `POST /api/optimize` — synchronous, only fine for `greedy` or a quick single genetic/hybrid run.
- `POST /api/optimize/async` + `GET /api/jobs/{job_id}` — starts the run in a background thread and returns a `job_id` immediately; the frontend polls every 3s. Any algorithm other than `greedy` should go through this path (see the frontend's `runGeneration`).
- A GitHub Actions workflow (`.github/workflows/keep-alive.yml`) pings `/api/health` every 10 minutes so Render's free tier doesn't spin the service down from inactivity during idle periods.

## Known gotcha: shared Supabase client + auth session expiry

`supabase` in `database/db.py` is a single module-level client reused across every request from every user. Its `ClientOptions` are explicitly set to `auto_refresh_token=False, persist_session=False` — **do not remove this**. With the defaults (both `True`), any login or token operation silently registers a background auto-refresh timer on that shared client, which rotates a user's Supabase refresh token on the server without the frontend knowing. Since the frontend (not the backend) owns token lifecycle via `localStorage`, that silent rotation invalidates the frontend's copy of the refresh token — surfacing as a forced logout roughly an hour after login, which was especially painful mid-way through a long GA/hybrid run. `GET/POST` auth endpoints (login, signup, `/api/auth/refresh`) all go through this same client, so this setting must stay off for all of them.
