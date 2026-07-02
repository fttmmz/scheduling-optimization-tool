# Project Layout

- **Backend** (this repo, GA / hybrid scheduling optimization algorithm): `D:\uniDB\scheduling-optimization-tool`
  - Deployed via **Render**, under the **fttmz** account.
- **Frontend**: `D:\UCT frontend\v0-university-timetable-tool-2`
  - Deployed via **Vercel**, under the **logiphic** account.

The frontend calls this backend's `/api/optimize` endpoint (see the frontend's `NEXT_PUBLIC_API_URL` and `lib/api.ts`).
