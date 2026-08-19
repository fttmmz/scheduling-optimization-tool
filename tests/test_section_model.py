"""The section model: one entity per (course, section, meeting pattern).

The source table stores one row per INSTRUCTOR. Building one Section per row --
what the loader did until 18 Aug 2026 -- inflated demand by ~44%, scheduling the
same class into several rooms at once. See HANDOVER.md section 4.

These tests run against the committed fixture, so they exercise the real
collapse on the real data without touching the database.
"""
from collections import Counter

import pytest

from backend.models.models import build_sections, group_rows_by_pattern
from backend.Optimization.constraints import (
    count_sibling_room_splits,
    group_siblings,
    instructor_occupancy_ids,
    section_needs,
    unify_sibling_rooms,
)
from backend.Optimization.evaluation import count_sibling_conflicts
from backend.models.models import ScheduleItem

EXPECTED_ENTITIES = 3270
EXPECTED_MULTI_PATTERN_SECTIONS = 11
EXPECTED_EXTRA_PATTERNS = 16
EXPECTED_MULTI_INSTRUCTOR = 106
EXPECTED_EXTRA_ASSIGNMENTS = 598
EXPECTED_WITH_INSTRUCTOR = 1486


@pytest.fixture(scope="module")
def sections(raw_rows):
    return build_sections(raw_rows)


# ── The collapse ─────────────────────────────────────────────────────────────

def test_rows_collapse_to_the_right_number_of_entities(raw_rows, sections):
    """4,698 rows -> 3,270 things to schedule.

    Not 4,698 (one per row -- counts duplicates and roster entries as separate
    classes) and not 3,254 (one per (course, section) -- deletes 16 real
    clinical and studio meeting blocks).
    """
    assert len(raw_rows) == 4698
    assert len(sections) == EXPECTED_ENTITIES


def test_no_two_entities_share_an_identity(sections):
    ids = [section.id for section in sections]
    assert len(ids) == len(set(ids))


def test_entity_order_is_deterministic(raw_rows):
    first = [s.id for s in build_sections(raw_rows)]
    second = [s.id for s in build_sections(raw_rows)]
    assert first == second


# ── Instructors ──────────────────────────────────────────────────────────────

def test_every_instructor_survives_the_collapse(sections):
    """598 assignments were previously discarded by keeping only the first."""
    multi = [s for s in sections if len(s.instructor_ids) > 1]
    assert len(multi) == EXPECTED_MULTI_INSTRUCTOR
    assert sum(len(s.instructor_ids) - 1 for s in multi) == EXPECTED_EXTRA_ASSIGNMENTS


def test_sections_with_any_instructor(sections):
    assert sum(1 for s in sections if s.instructor_ids) == EXPECTED_WITH_INSTRUCTOR


def test_instructor_ids_never_contain_none(sections):
    for section in sections:
        assert None not in section.instructor_ids


def test_scalar_instructor_id_is_the_first_of_the_list(sections):
    for section in sections:
        if section.instructor_ids:
            assert section.instructor_id == section.instructor_ids[0]
        else:
            assert section.instructor_id is None


def test_a_team_taught_class_occupies_all_its_instructors(sections):
    """The behaviour the scalar read silently lost.

    'Intensive English Foundation 1' is taught by five people; four of them were
    invisible to the conflict counter.
    """
    team = next(s for s in sections
                if len(s.instructor_ids) > 1 and instructor_occupancy_ids(s))
    occupying = instructor_occupancy_ids(team)
    assert len(occupying) == len(team.instructor_ids) > 1
    assert set(occupying) == set(team.instructor_ids)


# ── Meeting patterns ─────────────────────────────────────────────────────────

def test_multi_pattern_sections_survive(sections):
    groups = group_siblings(sections)
    assert len(groups) == EXPECTED_MULTI_PATTERN_SECTIONS
    assert sum(len(g) - 1 for g in groups.values()) == EXPECTED_EXTRA_PATTERNS


def test_the_five_block_clinical_section_is_five_entities(sections):
    """1002776 sec 71 meets five times a week. Collapsing it would delete four."""
    group = [s for s in sections if s.sibling_key == (1002776, "71")]
    assert len(group) == 5
    assert {s.pattern_index for s in group} == {0, 1, 2, 3, 4}
    assert all(s.pattern_count == 5 for s in group)


def test_siblings_share_a_sibling_key_but_not_an_id(sections):
    for group in group_siblings(sections).values():
        assert len({s.sibling_key for s in group}) == 1
        assert len({s.id for s in group}) == len(group)


def test_clinical_multi_pattern_sections_need_no_room(sections):
    """So 'clinical siblings share a room' is vacuous -- they get none.

    Room sharing only bites on the six Studio sections of 404421. Recorded here
    because it is counter-intuitive and was worth discovering once.
    """
    groups = group_siblings(sections)
    needing_room = [g for g in groups.values() if section_needs(g[0]).room]
    assert len(needing_room) == 6
    assert {g[0].course.id for g in needing_room} == {404421}
    for group in groups.values():
        if group[0].course.type == "Clinical Practice":
            assert section_needs(group[0]).room is False


# ── Sibling rules ────────────────────────────────────────────────────────────

def item(course_id=1, section="61", room_id=None, timeslot_id=100, pattern_index=0):
    return ScheduleItem(
        course_id=course_id, course_name="Test", course_type="Studio Undergraduate",
        course_dept=1, capacity=30, instructor_id=None, room_id=room_id,
        timeslot_id=timeslot_id, section=section, level="Undergraduate",
        pattern_index=pattern_index,
    )


def test_two_blocks_of_one_section_at_one_hour_is_a_conflict():
    """The students cannot attend their own class twice at once.

    Nothing else would catch this: the blocks may carry no room and no
    instructor, so neither the room nor the instructor check applies.
    """
    schedule = [item(timeslot_id=100, pattern_index=0),
                item(timeslot_id=100, pattern_index=1)]
    assert count_sibling_conflicts(schedule) == 1


def test_blocks_of_one_section_at_different_hours_are_fine():
    schedule = [item(timeslot_id=100, pattern_index=0),
                item(timeslot_id=101, pattern_index=1)]
    assert count_sibling_conflicts(schedule) == 0


def test_different_sections_at_one_hour_are_not_sibling_conflicts():
    schedule = [item(section="61"), item(section="62")]
    assert count_sibling_conflicts(schedule) == 0


def test_unify_sibling_rooms_moves_blocks_into_one_room():
    schedule = [
        item(room_id=10, timeslot_id=100, pattern_index=0),
        item(room_id=99, timeslot_id=101, pattern_index=1),
    ]
    assert count_sibling_room_splits(schedule) == 1

    unify_sibling_rooms(schedule)
    assert count_sibling_room_splits(schedule) == 0
    assert {i.room_id for i in schedule} == {10}


def test_unify_never_creates_a_room_clash():
    """If the target room is busy at that hour, the block stays put."""
    schedule = [
        item(course_id=1, room_id=10, timeslot_id=100, pattern_index=0),
        item(course_id=1, room_id=99, timeslot_id=101, pattern_index=1),
        item(course_id=2, section="99", room_id=10, timeslot_id=101),  # blocks it
    ]
    unify_sibling_rooms(schedule)

    assert schedule[1].room_id == 99, "must not displace the blocking class"
    rooms_times = [(i.room_id, i.timeslot_id) for i in schedule]
    assert len(rooms_times) == len(set(rooms_times))


def test_unify_is_idempotent():
    schedule = [
        item(room_id=10, timeslot_id=100, pattern_index=0),
        item(room_id=99, timeslot_id=101, pattern_index=1),
    ]
    unify_sibling_rooms(schedule)
    first = [i.room_id for i in schedule]
    unify_sibling_rooms(schedule)
    assert [i.room_id for i in schedule] == first


def test_unify_leaves_roomless_blocks_alone():
    """Clinical blocks need no room; the pass must not invent one."""
    schedule = [item(room_id=None, timeslot_id=100, pattern_index=0),
                item(room_id=None, timeslot_id=101, pattern_index=1)]
    unify_sibling_rooms(schedule)
    assert all(i.room_id is None for i in schedule)


# ── Grouping internals ───────────────────────────────────────────────────────

def test_grouping_is_by_pattern_not_by_section(raw_rows):
    grouped = group_rows_by_pattern(raw_rows)
    assert len(grouped) == 3254, "distinct (course, section)"
    assert sum(len(p) for p in grouped.values()) == EXPECTED_ENTITIES


def test_pattern_counts_match_group_sizes(sections):
    by_key = Counter(s.sibling_key for s in sections)
    for section in sections:
        assert section.pattern_count == by_key[section.sibling_key]
