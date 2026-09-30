"""Scheduling constraints: the single source of truth every algorithm imports.

Import from the package, never from a submodule, so the layout can change
without touching callers:

    from backend.Optimization.constraints import get_viable_rooms

    classification.py    what a section IS: Needs, supervision, combined
                         classes, sibling meeting blocks
    timeslots.py         which timeslots fit a section, preferred first
    rooms.py             which rooms suit a section, cheapest soft penalty first
    hard.py              the hard tier: double-booked room or instructor
    repair.py            finalize_schedule(), run by every algorithm on its result
    preprocessing.py     section tagging done once by main.py
    institution_data.py  the real course ids the rules above depend on

Every algorithm needs classification, timeslots, rooms and repair. Only genetic
and GRASP call passes_hard_constraints(); greedy, hybrid and PSO track occupancy
themselves. evaluation.py uses the *_parts() penalty breakdowns for reporting.
"""

from .classification import (
    DEFAULT_NEEDS,
    MINED_NEEDS,
    NEEDS_NOTHING,
    NEEDS_ROOM_AND_TIME,
    NEEDS_ROOM_ONLY,
    NEEDS_TIME_ONLY,
    SUPERVISION_COURSE_TYPES,
    Needs,
    classify_section,
    combined_group_key,
    count_sibling_room_splits,
    group_siblings,
    instructor_occupancy_id,
    instructor_occupancy_ids,
    is_branch_campus,
    is_supervision,
    item_needs,
    needs_for,
    needs_to_label,
    occupancy_key,
    section_needs,
    sibling_key,
)
from .hard import (
    is_instructor_free,
    is_instructor_free_map,
    is_room_free,
    is_room_free_map,
    passes_hard_constraints,
)
from .institution_data import COMBINED_COURSE_GROUPS, SUPERVISION_COURSE_IDS
from .preprocessing import intro_it_days_ok, tag_intro_it_pairs, tag_section_links
from .repair import finalize_schedule, resolve_sibling_time_conflicts, unify_sibling_rooms
from .rooms import (
    COURSE_TYPE_TO_ROOM_TYPE,
    DEFAULT_ROOM_CANDIDATES,
    SOFT_WEIGHT_CAMPUS,
    SOFT_WEIGHT_CAPACITY,
    SOFT_WEIGHT_CAPACITY_OVERFLOW,
    SOFT_WEIGHT_DEPARTMENT,
    SOFT_WEIGHT_ROOM_TYPE,
    capacity_penalty,
    get_building_campus,
    get_required_room_type,
    get_section_campus,
    get_viable_rooms,
    get_viable_rooms_for_schedule_item,
    is_campus_match,
    is_room_allowed,
    is_capacity_ok,
    is_department_match,
    is_room_type_match,
    rank_rooms,
    room_soft_penalty,
    room_rule_violations,
    room_soft_penalty_parts,
    seats_needed,
    section_campus,
    sections_without_allowed_room,
)
from .timeslots import (
    DAYTIME_COURSE_TYPES,
    EVENING_CUTOFF_HOUR,
    EVENING_PREFERRED_EXTRA_TYPES,
    SINGLE_DAY,
    SOFT_WEIGHT_SUPERVISION_DAYTIME,
    SOFT_WEIGHT_TEACHING_EVENING,
    TIME_NEUTRAL_COURSE_TYPES,
    _parse_time_to_minutes,
    _section_suffix,
    _timeslot_duration,
    get_valid_timeslots,
    get_valid_timeslots_for_section,
    is_evening_timeslot,
    is_placeholder_timeslot,
    prefers_daytime,
    prefers_evening,
    rank_timeslots,
    time_soft_penalty,
    time_soft_penalty_parts,
)
