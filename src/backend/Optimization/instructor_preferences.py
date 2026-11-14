"""Manage instructor preference settings for individual scheduling runs.

Saved preferences are loaded from the database and passed to the shared
fitness calculation. A configurable penalty calculator supplies the instructor preference cost
without changing the original fitness when preferences are disabled.

A context variable keeps each run's settings isolated and is reset when the
run finishes. Algorithm-specific candidate scoring is handled separately.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
from typing import Callable, Mapping, Optional

from backend.preferences import PreferenceIn

# A penalty calculator receives the timetable, timeslots, and saved preferences.
# It returns a nonnegative soft penalty before the global weight is applied.
PreferencePenaltyCalculator = Callable[[list, list, Mapping[str, dict]], float]


@dataclass(frozen=True)
class PreferenceRunConfig:
    requested: bool = False
    weight: float = 1.0
    by_instructor: Mapping[str, dict] = field(default_factory=dict)
    calculator: Optional[PreferencePenaltyCalculator] = None

    def __post_init__(self):
        if not math.isfinite(self.weight) or self.weight not in (0.0, 1.0):
            raise ValueError("preference lambda must be 0 (OFF) or 1 (ON)")

    @property
    def active(self) -> bool:
        """True only when the preferences can actually affect fitness."""
        return bool(
            self.requested and self.weight > 0 and self.by_instructor
            and self.calculator is not None
        )

    @property
    def status(self) -> str:
        if not self.requested:
            return "off"
        if self.weight == 0:
            return "zero_weight"
        if not self.by_instructor:
            return "no_saved_preferences"
        if self.calculator is None:
            return "waiting_for_preference_logic"
        return "fitness_hook_active"


_CURRENT_CONFIG: ContextVar[Optional[PreferenceRunConfig]] = ContextVar(
    "instructor_preference_run_config", default=None
)


def current_preference_config() -> Optional[PreferenceRunConfig]:
    return _CURRENT_CONFIG.get()


@contextmanager
def preference_run_context(config: Optional[PreferenceRunConfig]):
    token = _CURRENT_CONFIG.set(config)
    try:
        yield
    finally:
        _CURRENT_CONFIG.reset(token)


def preference_penalty(schedule, timeslots, config: Optional[PreferenceRunConfig]) -> float:
    """Return the weighted preference penalty, or exactly zero when inactive."""
    if config is None or not config.active:
        return 0.0
    if timeslots is None:
        raise ValueError("timeslots are required when instructor preferences are active")
    value = float(config.calculator(schedule, timeslots, config.by_instructor))
    if not math.isfinite(value) or value < 0:
        raise ValueError("instructor preference penalty must be finite and nonnegative")
    return config.weight * value


def load_all_instructor_preferences(client, page_size: int = 1000) -> dict:
    """Load and validate all saved instructor preferences from Supabase.

    Read rows in pages and normalize instructor IDs to strings so they match
    schedule records. Return only validated preference fields.
    """
    if page_size < 1:
        raise ValueError("page_size must be positive")
    result = {}
    offset = 0
    while True:
        response = (
            client.table("preference")
            .select("*")
            .order("inst_id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = response.data or []
        for row in rows:
            if row.get("inst_id") is None:
                raise ValueError("preference row is missing inst_id")
            # Validate the preference fields and their associated priority settings.
            validated = PreferenceIn.model_validate(row)
            result[str(row["inst_id"])] = validated.model_dump()
        if len(rows) < page_size:
            break
        offset += page_size
    return result


def get_preference_calculator() -> PreferencePenaltyCalculator:
    """Return the instructor preference penalty calculator used by the optimizer."""
    from backend.Optimization.provisional_preference_penalty import (
        calculate_instructor_preference_penalty,
    )
    return calculate_instructor_preference_penalty
