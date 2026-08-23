"""Pins the dataset facts established in HANDOVER.md section 4.

These are not tests of our code -- they are tests of our UNDERSTANDING of the
data. Every number here was verified against the database and cross-checked
against the live student registration site on 16 Aug 2026.

If one of these fails, the dataset changed. That is a real event: stop, work out
what moved, and update the expectation deliberately. Do not "fix" a failure by
editing the number to match.

The section model rework (HANDOVER.md section 5 item 1) must keep every one of
these passing -- in particular test_meeting_patterns_survive_deduplication,
which is the one the originally-proposed fix would have broken.
"""
from collections import Counter, defaultdict

from conftest import class_key, full_key, is_real_timeslot, section_key

# ── The headline counts (HANDOVER.md 4.1) ────────────────────────────────────

EXPECTED_ROWS = 4698
EXPECTED_FULL_KEYS = 3921       # distinct incl. instructor
EXPECTED_CLASS_KEYS = 3270      # distinct excl. instructor -- the TARGET entity count
EXPECTED_SECTION_KEYS = 3254    # distinct (course, section)
EXPECTED_TRUE_DUPLICATES = 777  # rows - full keys
EXPECTED_ROSTER_ROWS = 651      # full keys - class keys


def test_row_count(raw_rows):
    assert len(raw_rows) == EXPECTED_ROWS


def test_distinct_key_counts(raw_rows):
    assert len(set(map(full_key, raw_rows))) == EXPECTED_FULL_KEYS
    assert len(set(map(class_key, raw_rows))) == EXPECTED_CLASS_KEYS
    assert len(set(map(section_key, raw_rows))) == EXPECTED_SECTION_KEYS


def test_duplicate_arithmetic(raw_rows):
    """rows = distinct classes + true duplicates + roster rows."""
    full = len(set(map(full_key, raw_rows)))
    cls = len(set(map(class_key, raw_rows)))

    assert len(raw_rows) - full == EXPECTED_TRUE_DUPLICATES
    assert full - cls == EXPECTED_ROSTER_ROWS
    assert cls + EXPECTED_ROSTER_ROWS + EXPECTED_TRUE_DUPLICATES == EXPECTED_ROWS


# ── The instructor column is mostly empty (HANDOVER.md 4.3) ──────────────────

EXPECTED_NULL_INSTRUCTOR_ROWS = 2614
EXPECTED_DISTINCT_INSTRUCTORS = 311   # real people; the 312 in early notes counted NULL
EXPECTED_SECTIONS_WITH_INSTRUCTOR = 1486


def test_most_rows_have_no_instructor(raw_rows):
    """56% of rows carry no instructor at all.

    This is why instructor conflicts can never be the whole feasibility story,
    and why every occupancy structure must skip None rather than treating it as
    a person who would then be 'double-booked' 2,614 times over.
    """
    nulls = sum(1 for r in raw_rows if r["instructor_id"] is None)
    assert nulls == EXPECTED_NULL_INSTRUCTOR_ROWS
    assert nulls / len(raw_rows) > 0.5


def test_distinct_instructor_count(raw_rows):
    real = {r["instructor_id"] for r in raw_rows if r["instructor_id"] is not None}
    assert len(real) == EXPECTED_DISTINCT_INSTRUCTORS


def test_fewer_than_half_of_sections_have_an_instructor(raw_rows):
    by_section = defaultdict(set)
    for row in raw_rows:
        by_section[section_key(row)].add(row["instructor_id"])
    with_real = sum(1 for v in by_section.values() if v - {None})
    assert with_real == EXPECTED_SECTIONS_WITH_INSTRUCTOR


# ── "Team teaching" is mostly supervision rosters (HANDOVER.md 4.3) ──────────

EXPECTED_MULTI_INSTRUCTOR_CLASSES = 106
EXPECTED_EXTRA_ASSIGNMENTS = 598
EXPECTED_NULL_MIXED_CLASSES = 53


def _real_instructors_by_class(rows):
    by_class = defaultdict(set)
    for row in rows:
        if row["instructor_id"] is not None:
            by_class[class_key(row)].add(row["instructor_id"])
    return by_class


def test_multi_instructor_classes_excluding_nulls(raw_rows):
    """106 classes / +598 assignments -- NOT the 132 / 651 originally reported.

    The larger figure counted NULL as an instructor value. The two reconcile:
    598 genuine multi-instructor rows + 53 null-mixed classes = 651.
    """
    multi = {k: v for k, v in _real_instructors_by_class(raw_rows).items() if len(v) > 1}
    assert len(multi) == EXPECTED_MULTI_INSTRUCTOR_CLASSES
    assert sum(len(v) - 1 for v in multi.values()) == EXPECTED_EXTRA_ASSIGNMENTS


def test_null_mixed_classes_account_for_the_remainder(raw_rows):
    by_class = defaultdict(set)
    for row in raw_rows:
        by_class[class_key(row)].add(row["instructor_id"])
    mixed = sum(1 for v in by_class.values() if None in v and len(v) > 1)

    assert mixed == EXPECTED_NULL_MIXED_CLASSES
    assert EXPECTED_EXTRA_ASSIGNMENTS + mixed == EXPECTED_ROSTER_ROWS


def test_widest_multi_instructor_classes_are_supervision(raw_rows, timeslots_by_id):
    """The 15+-instructor classes are supervision rosters, not co-taught classes.

    Guards the interpretation, not just the count: if instructor occupancy is
    ever built from these rows it will invent double-bookings for supervisors
    who are not in a room together.

    Tested by SHAPE (no room, no real meeting time) rather than by course_type,
    because course_type is not reliable here -- see
    test_supervision_courses_are_miscategorised_by_type below.
    """
    multi = _real_instructors_by_class(raw_rows)
    widest = [k for k, v in multi.items() if len(v) >= 15]
    assert len(widest) >= 20, "expected many wide supervision rosters"

    for key in widest:
        room_id, timeslot_id = key[2], key[3]
        assert room_id is None, f"{key} occupies a real room with 15+ instructors"
        # 1100599 'Graduation Project' is the sole roster with a real meeting
        # time (T 19:00) -- an evening group session, consistent with 4.7.
        if key[0] != 1100599:
            assert not is_real_timeslot(timeslots_by_id.get(timeslot_id)), key


# Supervision courses whose `course_type` column does not say so. Found by the
# test above on 16 Aug 2026. `course_type` is the ONLY signal the classification
# table (constraints.MINED_NEEDS) keys on, so each of these is currently treated
# as an ordinary class needing a room and a time -- when the source data gives
# it neither, and it carries a 22-person supervisor roster.
MISCATEGORISED_SUPERVISION = {
    1501497: ("Senior Project in Bioinfor.", "Lecture Undergraduate"),
}


def test_supervision_courses_are_miscategorised_by_type(raw_rows, timeslots_by_id):
    """KNOWN DEFECT -- course_type cannot be trusted to identify supervision.

    'Senior Project in Bioinfor.' is typed 'Lecture Undergraduate' but has 22
    supervisors, no room and no meeting time. classify_section() therefore hands
    it the room+time default and the scheduler will book a room it never needed.

    This pins the defect so HANDOVER.md section 5 item 2 (separating teaching
    staff from supervision rosters) cannot be built on course_type alone. Extend
    the dict if more turn up; remove entries only when the data is corrected.
    """
    by_course = defaultdict(list)
    for row in raw_rows:
        by_course[row["course_id"]].append(row)

    for course_id, (name, wrong_type) in MISCATEGORISED_SUPERVISION.items():
        rows = by_course[course_id]
        assert rows, f"course {course_id} vanished from the dataset"

        course = rows[0]["courses"]
        assert course["name"] == name
        assert course["course_type"] == wrong_type
        assert "project" in name.lower(), "name is the corroborating signal"

        # The shape that contradicts the type: no room, no time, many staff.
        assert all(r["room_id"] is None for r in rows)
        assert not any(is_real_timeslot(timeslots_by_id.get(r["timeslot_id"]))
                       for r in rows)
        assert len({r["instructor_id"] for r in rows if r["instructor_id"]}) >= 15


# ── Duplicates are people, not meetings (HANDOVER.md 4.4) ────────────────────

EXPECTED_REDUNDANT_NEEDING_NOTHING = 502
EXPECTED_REDUNDANT_TIME_ONLY = 145
EXPECTED_REDUNDANT_FULLY_SCHEDULABLE = 130


def test_redundant_rows_split_by_what_they_need(raw_rows, timeslots_by_id):
    """Only 130 of the 777 redundant rows are fully schedulable.

    The rest need no room (and usually no time) -- thesis and dissertation
    supervision. Collapsing them cannot cost a real booking.
    """
    counts = Counter()
    for key, n in Counter(map(full_key, raw_rows)).items():
        if n > 1:
            has_room = key[2] is not None
            has_time = is_real_timeslot(timeslots_by_id.get(key[3]))
            counts[(has_room, has_time)] += n - 1

    assert counts[(False, False)] == EXPECTED_REDUNDANT_NEEDING_NOTHING
    assert counts[(False, True)] == EXPECTED_REDUNDANT_TIME_ONLY
    assert counts[(True, True)] == EXPECTED_REDUNDANT_FULLY_SCHEDULABLE
    assert sum(counts.values()) == EXPECTED_TRUE_DUPLICATES


def test_duplicates_concentrate_in_supervision_course_types(raw_rows):
    """Thesis and senior-project supervision dominate, not clinical practice."""
    course_type = {r["course_id"]: (r["courses"] or {}).get("course_type")
                   for r in raw_rows}
    redundant = Counter()
    for key, n in Counter(map(full_key, raw_rows)).items():
        if n > 1:
            redundant[course_type[key[0]]] += n - 1

    top_three = [t for t, _ in redundant.most_common(3)]
    assert top_three == [
        "Thesis / Dissertation Master",
        "Thesis / Dissertation Doctorat",
        "Clinical Practice",
    ]
    supervision = (redundant["Thesis / Dissertation Master"]
                   + redundant["Thesis / Dissertation Doctorat"]
                   + redundant["Senior Project Supervision"])
    assert supervision > redundant["Clinical Practice"] * 3


# ── Meeting patterns are real and must survive (HANDOVER.md 4.5) ─────────────

EXPECTED_MULTI_PATTERN_SECTIONS = 11
EXPECTED_EXTRA_PATTERNS = 16


def _patterns_by_section(rows):
    patterns = defaultdict(set)
    for row in rows:
        patterns[section_key(row)].add((row["room_id"], row["timeslot_id"]))
    return patterns


def test_meeting_patterns_survive_deduplication(raw_rows):
    """THE REGRESSION GUARD for HANDOVER.md section 5 item 1.

    Eleven sections meet in more than one room/time block -- confirmed against
    the registration site. Deduplicating on (course, section) would silently
    discard 16 real meetings. The entity count must be 3,270, not 3,254.
    """
    patterns = _patterns_by_section(raw_rows)
    multi = {k: v for k, v in patterns.items() if len(v) > 1}

    assert len(multi) == EXPECTED_MULTI_PATTERN_SECTIONS
    assert sum(len(v) - 1 for v in multi.values()) == EXPECTED_EXTRA_PATTERNS
    assert len(patterns) + EXPECTED_EXTRA_PATTERNS == EXPECTED_CLASS_KEYS


def test_clinical_section_keeps_all_five_blocks(raw_rows, timeslots_by_id):
    """1002776 sec 71 meets five times a week, morning and afternoon."""
    blocks = _patterns_by_section(raw_rows)[(1002776, "71")]
    assert len(blocks) == 5

    starts = sorted(timeslots_by_id[ts]["start_time"] for _, ts in blocks)
    assert any(s < "12:00:00" for s in starts), "expected morning blocks"
    assert any(s > "12:00:00" for s in starts), "expected afternoon blocks"


def test_multi_pattern_sections_are_clinical_or_studio(raw_rows):
    course_type = {r["course_id"]: (r["courses"] or {}).get("course_type")
                   for r in raw_rows}
    multi = {k for k, v in _patterns_by_section(raw_rows).items() if len(v) > 1}
    assert {course_type[k[0]] for k in multi} == {
        "Clinical Practice", "Studio Undergraduate",
    }


def test_day_column_encodes_multiple_days_in_one_row(raw_timeslots):
    """'MTR' is Mon+Tue+Thu in a SINGLE timeslot row, not three rows.

    Anything that assumes one calendar day per timeslot will miscount overlap.
    """
    days = {t["day"] for t in raw_timeslots if t["day"]}
    multi_day = {d for d in days if len(d) > 1}
    assert {"MTR", "MTWR", "TR", "MW"} <= multi_day


# ── 401498/401499 are prerequisite-linked, NOT cross-listed (4.6) ────────────

def test_401498_401499_are_not_colocated(raw_rows):
    """They are separate courses, one a prerequisite of the other.

    Guards against anyone re-adding them to COMBINED_COURSE_GROUPS on the
    strength of their similar shape -- combining them would excuse a genuine
    double-booking between them. They do not even share a room.
    """
    from backend.Optimization.constraints import combined_group_key

    rooms = {
        cid: {r["room_id"] for r in raw_rows if r["course_id"] == cid}
        for cid in (401498, 401499)
    }
    assert rooms[401498].isdisjoint(rooms[401499])
    assert combined_group_key(401498) is None
    assert combined_group_key(401499) is None


# ── Supervision runs in the evening (HANDOVER.md 4.7) ────────────────────────

def _start_hours_by_type(rows, timeslots_by_id, course_type):
    hours = []
    for row in rows:
        if (row["courses"] or {}).get("course_type") != course_type:
            continue
        ts = timeslots_by_id.get(row["timeslot_id"])
        if is_real_timeslot(ts):
            hours.append(int(str(ts["start_time"])[:2]))
    return hours


def test_supervision_is_scheduled_in_the_evening(raw_rows, timeslots_by_id):
    """98% of timed supervision starts at or after 17:00; lectures 6%.

    A revealed preference: daytime slots belong to regular teaching. This is the
    one Phase 3 preference recoverable from existing data (HANDOVER.md 5 item 3),
    so the pattern is pinned here before anything starts optimising against it.
    """
    supervision = _start_hours_by_type(raw_rows, timeslots_by_id,
                                       "Senior Project Supervision")
    lectures = _start_hours_by_type(raw_rows, timeslots_by_id,
                                    "Lecture Undergraduate")

    assert supervision and lectures
    late_supervision = sum(1 for h in supervision if h >= 17) / len(supervision)
    late_lectures = sum(1 for h in lectures if h >= 17) / len(lectures)

    assert late_supervision > 0.95
    assert late_lectures < 0.10
    assert sorted(supervision)[len(supervision) // 2] >= 17
    assert sorted(lectures)[len(lectures) // 2] <= 12


# -- The evidence the evening preference rests on (HANDOVER.md 4.7, 10) -------
#
# The tests above pin Senior Project Supervision vs Lecture Undergraduate, which
# is the headline. These pin the rest of the per-type evidence that decides
# WHICH types get a preference at all, because two of those verdicts overturn
# what the old low-priority list assumed and would otherwise be one edit away
# from being quietly reverted.

MIN_ROWS_TO_JUDGE = 20      # below this a rate is noise
STRONG = 0.80               # 80/20 one way, or no preference is recorded


def _evening_rate(rows, timeslots_by_id, course_type):
    """(timed rows, fraction starting at or after 17:00) for one course type."""
    hours = _start_hours_by_type(rows, timeslots_by_id, course_type)
    if not hours:
        return 0, 0.0
    return len(hours), sum(1 for h in hours if h >= 17) / len(hours)


def test_daytime_types_really_are_daytime(raw_rows, timeslots_by_id):
    """Every type charged for an evening slot must earn it in the data."""
    from backend.Optimization.constraints import DAYTIME_COURSE_TYPES

    for course_type in DAYTIME_COURSE_TYPES:
        rows, rate = _evening_rate(raw_rows, timeslots_by_id, course_type)
        assert rows, course_type
        if rows >= MIN_ROWS_TO_JUDGE:
            assert rate <= 1 - STRONG, f"{course_type}: {rows} rows, {rate:.0%} evening"


def test_office_hours_has_no_time_preference(raw_rows, timeslots_by_id):
    """47% -- a coin flip, and the reason to measure before encoding.

    Office Hours was in the old low-priority list and was hard-filtered into
    afternoon slots. Carrying that list over without checking would have priced
    a coin flip as a rule.
    """
    rows, rate = _evening_rate(raw_rows, timeslots_by_id, "Office Hours")
    assert rows >= MIN_ROWS_TO_JUDGE
    assert 1 - STRONG < rate < STRONG


def test_project_is_supervision_but_has_no_time_preference(raw_rows, timeslots_by_id):
    """40% evening. Supervision for OCCUPANCY, neutral for TIMING.

    The two questions are separate and here the answers differ, which is what
    constraints.TIME_NEUTRAL_COURSE_TYPES exists for.
    """
    from backend.Optimization.constraints import (
        SUPERVISION_COURSE_TYPES, TIME_NEUTRAL_COURSE_TYPES,
    )

    rows, rate = _evening_rate(raw_rows, timeslots_by_id, "Project")
    assert rows >= MIN_ROWS_TO_JUDGE
    assert 1 - STRONG < rate < STRONG
    assert "Project" in SUPERVISION_COURSE_TYPES
    assert "Project" in TIME_NEUTRAL_COURSE_TYPES


def test_graduate_lectures_have_no_time_preference(raw_rows, timeslots_by_id):
    """52% evening over 477 rows -- graduate teaching genuinely splits the day.

    This is why the daytime rule is not simply "everything that is not
    supervision": half of graduate teaching would be penalised for a pattern it
    demonstrably follows.
    """
    from backend.Optimization.constraints import DAYTIME_COURSE_TYPES

    rows, rate = _evening_rate(raw_rows, timeslots_by_id, "Lecture Graduate")
    assert rows >= MIN_ROWS_TO_JUDGE
    assert 1 - STRONG < rate < STRONG
    assert "Lecture Graduate" not in DAYTIME_COURSE_TYPES


def test_level_is_not_a_usable_key_for_the_daytime_rule(raw_rows):
    """`level` carries eight values, and 'Lecture Undergraduate' spans five.

    An earlier cut of the evening preference keyed the daytime rule on
    level == 'Undergraduate' and silently exempted 356 undergraduate lecture
    rows filed under 'Fine Art', 'Diploma', 'Intensive English' or
    'Foundation Year'. Pinned so nobody reaches for that field again.
    """
    levels = Counter(
        (row["courses"] or {}).get("level")
        for row in raw_rows
        if (row["courses"] or {}).get("course_type") == "Lecture Undergraduate"
    )
    assert len(levels) > 1
    assert "Undergraduate" in levels
    non_undergraduate = sum(v for k, v in levels.items() if k != "Undergraduate")
    assert non_undergraduate > 300, dict(levels)
