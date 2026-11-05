"""Instructor preferences: the saved shape, its rules, storage, and login by instructor id.

The rules repeat the check constraints in scripts/data_import/instructor_preferences.sql, so a bad save gets
a clear answer here instead of a database error. Functions that touch the database take the client as an
argument, so this module opens no connection and the offline tests can import it.
"""
from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Day = Literal["M", "T", "W", "R"]       # the days the form offers, as the timeslot table writes them: R = Thursday
TimeBand = Literal["morning", "afternoon", "evening"]
Priority = Literal["low", "medium", "high"]

# Supabase logins need an email, but instructors sign in with their id. Each account gets this hidden address.
# It is confirmed when the account is created, so no mail is ever sent to it.
INSTRUCTOR_EMAIL_DOMAIN = "instructors.v0-university-timetable-tool-2.vercel.app"


def instructor_email(inst_id) -> str:
    return f"{inst_id}@{INSTRUCTOR_EMAIL_DOMAIN}"


def login_email(identifier: str) -> str:
    """The email to sign in with: an all-digit identifier is an instructor id, anything else is an email."""
    identifier = identifier.strip()
    return instructor_email(identifier) if identifier.isdigit() else identifier


class PreferenceIn(BaseModel):
    """One instructor's preferences. An empty priority means no preference on that item."""

    preferred_days: List[Day] = []
    avoid_days: List[Day] = []
    days_priority: Optional[Priority] = None
    preferred_times: List[TimeBand] = []
    time_priority: Optional[Priority] = None
    gaps_priority: Optional[Priority] = None
    fewer_days_priority: Optional[Priority] = None
    max_classes_per_day: Optional[int] = Field(None, ge=1, le=10)
    max_classes_priority: Optional[Priority] = None

    @model_validator(mode="after")
    def _check_rules(self):
        if set(self.preferred_days) & set(self.avoid_days):
            raise ValueError("A day cannot be both preferred and avoided")
        if (self.days_priority is None) != (not self.preferred_days and not self.avoid_days):
            raise ValueError("Days and their priority must be set together")
        if (self.time_priority is None) != (not self.preferred_times):
            raise ValueError("Preferred times and their priority must be set together")
        if (self.max_classes_priority is None) != (self.max_classes_per_day is None):
            raise ValueError("Max classes per day and its priority must be set together")
        return self


def find_instructor(client, user_id):
    """The instructor linked to this login, or None."""
    res = (
        client.table("instructor")
        .select("inst_id, name, dept_id")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def load_preference(client, inst_id):
    """This instructor's saved preferences, or None if they have none."""
    res = client.table("preference").select("*").eq("inst_id", inst_id).limit(1).execute()
    return res.data[0] if res.data else None


def save_preference(client, inst_id, preference: PreferenceIn):
    """Create or replace this instructor's row. It becomes source='manual', replacing any synthetic values."""
    row = {
        **preference.model_dump(),
        "inst_id": inst_id,
        "source": "manual",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    res = client.table("preference").upsert(row, on_conflict="inst_id").execute()
    if not res.data:
        raise RuntimeError("Failed to save preferences: no data returned")
    return res.data[0]
