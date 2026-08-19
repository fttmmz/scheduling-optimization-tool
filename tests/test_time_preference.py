"""The evening preference (HANDOVER.md section 4.7, section 10).

Daytime hours belong to taught contact; supervision runs around them in the
evening. The rule is SOFT and applies on the time axis, the way the room weights
apply on the room axis.

Two things here are guards against mistakes made while building this, both
caught by measuring rather than by reasoning:

  * the rule must not key on `level` (test_daytime_rule_does_not_key_on_level)
  * Office Hours has no preference at all (test_office_hours_is_neutral)

and one against the behaviour that was replaced:

  * ranking must never DROP a candidate (test_ranking_never_drops_a_candidate)
"""
import pytest

from backend.models.models import Room, ScheduleItem, Section, Timeslot
from backend.Optimization.constraints import (
    DAYTIME_COURSE_TYPES,
    EVENING_CUTOFF_HOUR,
    SOFT_WEIGHT_SUPERVISION_DAYTIME,
    SOFT_WEIGHT_TEACHING_EVENING,
    get_valid_timeslots_for_section,
    is_evening_timeslot,
    prefers_daytime,
    prefers_evening,
    rank_timeslots,
    time_soft_penalty,
    time_soft_penalty_parts,
)


def slot(timeslot_id, start, day="M", end=None):
    return Timeslot({
        "timeslot_id": timeslot_id, "day": day,
        "start_time": start,
        "end_time": end or f"{int(start[:2]) + 1:02d}:15:00",
    })


def section(course_type, level="Undergraduate", course_id=1, section_no="61"):
    return Section({
        "courses": {"course_id": course_id, "name": f"C{course_id}",
                    "course_type": course_type, "dept_id": 1,
                    "level": level, "course_class": None},
        "instructor_id": "X", "section": section_no, "sec_capacity": 25,
    }, instructor_ids=["X"])


def item(course_type, level="Undergraduate", course_id=1, timeslot_id=None):
    return ScheduleItem(
        course_id=course_id, course_name=f"C{course_id}", course_type=course_type,
        course_dept=1, capacity=25, instructor_id="X", room_id=None,
        timeslot_id=timeslot_id, section="61", level=level, instructor_ids=["X"],
    )


MORNING = slot(1, "09:00:00")
EVENING = slot(2, "18:00:00")


# ── The cutoff ───────────────────────────────────────────────────────────────

def test_evening_cutoff_is_seventeen_hundred():
    """17:00, not the 15:00 the old low-priority filter guessed at."""
    assert EVENING_CUTOFF_HOUR == 17
    assert not is_evening_timeslot(slot(1, "16:59:00"))
    assert is_evening_timeslot(slot(2, "17:00:00"))
    assert is_evening_timeslot(slot(3, "23:00:00"))
    assert not is_evening_timeslot(slot(4, "08:00:00"))


def test_a_slot_with_no_start_time_is_not_evening():
    """Placeholder rows must not be silently treated as preferred."""
    assert not is_evening_timeslot(Timeslot({
        "timeslot_id": 9, "day": "M", "start_time": None, "end_time": None,
    }))


# ── Who prefers what ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("course_type", [
    "Senior Project Supervision", "Thesis", "Thesis / Dissertation Master",
    "Thesis / Dissertation Doctorat", "Independent Study",
    "Proposal Doctorate", "Comprehensive Exam", "Qualifying Exam",
])
def test_supervision_prefers_evening(course_type):
    assert prefers_evening(section(course_type))
    assert prefers_evening(item(course_type))


def test_supervision_for_occupancy_is_not_supervision_for_timing():
    """'Project' is supervision and still meets in the daytime 60% of the time.

    Two different questions get asked of the same flag:

        does this person occupy a room at this hour?   -- no, it is supervision
        when is this activity scheduled?               -- 40% evening, no pattern

    is_supervision() answers the first. The evening preference asks the second,
    and for Project the answers diverge, which is what TIME_NEUTRAL_COURSE_TYPES
    exists for. Deriving one from the other without checking would have priced a
    coin flip as a rule -- the same mistake the old low-priority list made with
    Office Hours.
    """
    from backend.Optimization.constraints import (
        TIME_NEUTRAL_COURSE_TYPES, instructor_occupancy_ids, is_supervision,
    )

    project = section("Project")
    assert "Project" in TIME_NEUTRAL_COURSE_TYPES
    assert is_supervision(project)              # still supervision ...
    assert instructor_occupancy_ids(project) == ()   # ... so it holds no hours
    assert not prefers_evening(project)         # ... but has no time preference
    assert not prefers_daytime(project)
    assert time_soft_penalty(project, MORNING) == 0.0
    assert time_soft_penalty(project, EVENING) == 0.0


def test_miscategorised_supervision_courses_prefer_evening():
    """The five SUPERVISION_COURSE_IDS carry a TAUGHT course_type (section 4.8).

    prefers_evening() reuses is_supervision() rather than keeping a second list
    precisely so these are covered. The old hard-coded low-priority list missed
    all five.
    """
    # 1501497 'Senior Project in Bioinfor.' is typed 'Lecture Undergraduate'.
    bioinfo = section("Lecture Undergraduate", course_id=1501497)
    assert prefers_evening(bioinfo)


def test_supervision_never_also_prefers_daytime():
    """The two sets must be mutually exclusive, or a section is charged either way.

    1501497 matches DAYTIME_COURSE_TYPES by its (wrong) course_type AND matches
    supervision by id. Without prefers_daytime() checking prefers_evening()
    first, it would be penalised for a daytime slot as supervision and for an
    evening slot as an undergraduate lecture -- with no slot scoring zero.
    """
    bioinfo = section("Lecture Undergraduate", course_id=1501497)
    assert prefers_evening(bioinfo)
    assert not prefers_daytime(bioinfo)
    assert time_soft_penalty(bioinfo, EVENING) == 0.0


@pytest.mark.parametrize("course_type", sorted(DAYTIME_COURSE_TYPES))
def test_undergraduate_teaching_prefers_daytime(course_type):
    assert prefers_daytime(section(course_type))
    assert not prefers_evening(section(course_type))


@pytest.mark.parametrize("course_type", [
    "Lecture Graduate",     # 52% evening over 406 timed items -- a coin flip
    "Seminar Graduate",     # 26% over 46
    "Office Hours",         # 47% over 34
])
def test_types_without_a_measured_preference_are_neutral(course_type):
    """Charging these would be inventing a rule the data does not contain."""
    sec = section(course_type)
    assert not prefers_evening(sec)
    assert not prefers_daytime(sec)
    assert time_soft_penalty(sec, MORNING) == 0.0
    assert time_soft_penalty(sec, EVENING) == 0.0


def test_office_hours_is_neutral():
    """Regression guard, and the reason to measure before encoding.

    Office Hours sat in the old low-priority list and was hard-filtered into
    afternoon slots. Measured, it runs in the evening 47% of the time over 34
    timed items -- no preference whatsoever. Carrying the old list over without
    checking would have priced a coin flip as a rule.
    """
    oh = section("Office Hours")
    assert not prefers_evening(oh)
    assert not prefers_daytime(oh)


def test_daytime_rule_does_not_key_on_level():
    """Regression guard: `level` has eight values, not two.

    310 of the 1,858 timed 'Lecture Undergraduate' items are filed under
    'Fine Art' (157), 'Diploma' (94), 'Foundation Year' (48) or 'Intensive
    English' (11). An earlier cut of this rule tested
    level == 'Undergraduate' and silently exempted all 310 from a preference
    they obey at 91%.
    """
    for level in ("Undergraduate", "Fine Art", "Diploma",
                  "Foundation Year", "Intensive English"):
        sec = section("Lecture Undergraduate", level=level)
        assert prefers_daytime(sec), level
        assert time_soft_penalty(sec, EVENING) == SOFT_WEIGHT_TEACHING_EVENING, level


def test_a_missing_level_does_not_change_the_time_score():
    """`level` being dropped is the recurring bug in HANDOVER section 6.

    The time rule keys on course_type, so a lost `level` cannot quietly move an
    item between the daytime and neutral populations on top of everything else
    it already breaks.
    """
    with_level = item("Lecture Undergraduate", level="Undergraduate")
    without = item("Lecture Undergraduate", level=None)
    assert time_soft_penalty(with_level, EVENING) == time_soft_penalty(without, EVENING)


# ── Pricing ──────────────────────────────────────────────────────────────────

def test_both_directions_are_priced():
    supervision = section("Senior Project Supervision")
    lecture = section("Lecture Undergraduate")

    assert time_soft_penalty(supervision, MORNING) == SOFT_WEIGHT_SUPERVISION_DAYTIME
    assert time_soft_penalty(supervision, EVENING) == 0.0
    assert time_soft_penalty(lecture, EVENING) == SOFT_WEIGHT_TEACHING_EVENING
    assert time_soft_penalty(lecture, MORNING) == 0.0


def test_supervision_in_the_daytime_is_the_more_expensive_direction():
    """It consumes a scarce daytime hour that taught contact needs; an evening
    lecture is an inconvenience that takes nothing from anyone else."""
    assert SOFT_WEIGHT_SUPERVISION_DAYTIME > SOFT_WEIGHT_TEACHING_EVENING


def test_the_time_rule_stays_below_an_unscheduled_section():
    """A preference must never be worth stranding a class over.

    Unscheduled costs 2.0 normalised by section count; the soft tier is divided
    by SOFT_PENALTY_REFERENCE and weighted 0.5, so even the dearest time penalty
    is a small fraction of it. This is the arithmetic the old hard filter got
    wrong by making the preference unconditional.
    """
    from backend.Optimization.evaluation import (
        SOFT_PENALTY_REFERENCE, SOFT_PENALTY_WEIGHT, UNSCHEDULED_WEIGHT,
    )
    worst_time_cost = (
        max(SOFT_WEIGHT_SUPERVISION_DAYTIME, SOFT_WEIGHT_TEACHING_EVENING)
        / SOFT_PENALTY_REFERENCE * SOFT_PENALTY_WEIGHT
    )
    assert worst_time_cost < UNSCHEDULED_WEIGHT


def test_no_timeslot_costs_nothing():
    """Supervision that needs no time at all must not be charged for lacking one."""
    assert time_soft_penalty(section("Senior Project Supervision"), None) == 0.0
    parts = time_soft_penalty_parts(section("Lecture Undergraduate"), None)
    assert set(parts) == {"supervision_daytime", "teaching_evening"}
    assert sum(parts.values()) == 0.0


def test_penalty_parts_name_the_rule_that_was_bent():
    parts = time_soft_penalty_parts(section("Senior Project Supervision"), MORNING)
    assert parts["supervision_daytime"] > 0
    assert parts["teaching_evening"] == 0


# ── Ranking ──────────────────────────────────────────────────────────────────

def test_ranking_never_drops_a_candidate():
    """The whole point of the change: rank, do not filter.

    get_valid_timeslots_for_section() used to return ONLY slots at/after 15:00
    for supervision. A preference that removes candidates can leave a section
    unscheduled, which costs 2.0 -- far more than the 0.3 of soft penalty it was
    protecting. Ranking keeps every option and simply orders them.
    """
    slots = [slot(1, "08:00:00"), slot(2, "11:00:00"), slot(3, "18:00:00")]
    for course_type in ("Senior Project Supervision", "Lecture Undergraduate",
                        "Lecture Graduate"):
        ranked = rank_timeslots(section(course_type), slots)
        assert len(ranked) == len(slots), course_type
        assert {t.id for t in ranked} == {t.id for t in slots}, course_type


def test_ranking_puts_the_preferred_slots_first():
    slots = [slot(1, "08:00:00"), slot(2, "18:00:00"), slot(3, "11:00:00")]

    supervision = rank_timeslots(section("Senior Project Supervision"), slots)
    assert is_evening_timeslot(supervision[0])

    lecture = rank_timeslots(section("Lecture Undergraduate"), slots)
    assert not is_evening_timeslot(lecture[0])
    assert is_evening_timeslot(lecture[-1])


def test_ranking_is_stable_within_a_preference_band():
    """Equally-preferred slots keep their original order.

    greedy, GRASP and hybrid walk this list and take the first FREE slot, so a
    finer or unstable key would funnel every section of one kind onto the same
    few hours instead of spreading them.
    """
    slots = [slot(1, "08:00:00"), slot(2, "09:00:00"), slot(3, "11:00:00")]
    ranked = rank_timeslots(section("Lecture Undergraduate"), slots)
    assert [t.id for t in ranked] == [1, 2, 3]


def test_neutral_sections_keep_the_original_order():
    slots = [slot(1, "18:00:00"), slot(2, "08:00:00"), slot(3, "19:00:00")]
    ranked = rank_timeslots(section("Lecture Graduate"), slots)
    assert [t.id for t in ranked] == [1, 2, 3]


# ── End to end through the real entry point ──────────────────────────────────

def _mixed_day_slots():
    """Single-day slots spanning morning and evening, which is what supervision
    is offered (the shape rule keeps supervision on single days)."""
    return [
        slot(1, "08:00:00", day="M"),
        slot(2, "11:00:00", day="T"),
        slot(3, "18:00:00", day="W"),
        slot(4, "19:00:00", day="R"),
    ]


def test_supervision_gets_evening_slots_first():
    ranked = get_valid_timeslots_for_section(
        section("Senior Project Supervision"), _mixed_day_slots()
    )
    assert is_evening_timeslot(ranked[0])


def test_supervision_is_still_offered_daytime_slots():
    """Regression guard for the replaced hard filter.

    The old rule 1 returned late slots ONLY whenever any existed, so a
    supervision section could not fall back to a daytime slot even when every
    evening slot was already taken. All four must survive, evening first.
    """
    slots = _mixed_day_slots()
    ranked = get_valid_timeslots_for_section(
        section("Senior Project Supervision"), slots
    )
    assert {t.id for t in ranked} == {t.id for t in slots}
    assert [is_evening_timeslot(t) for t in ranked] == [True, True, False, False]


def test_supervision_survives_when_no_evening_slot_exists():
    daytime_only = [slot(1, "08:00:00", day="M"), slot(2, "11:00:00", day="T")]
    ranked = get_valid_timeslots_for_section(
        section("Senior Project Supervision"), daytime_only
    )
    assert len(ranked) == 2


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_fitness_prefers_the_evening_placement_for_supervision():
    from backend.Optimization.evaluation import calculate_fitness

    timeslots = [MORNING, EVENING]
    rooms = [Room({"room_id": 1, "capacity": 40, "room_type": "classroom",
                   "building": "M8", "dept_id": 1, "room_num": "1"})]

    evening = [item("Senior Project Supervision", timeslot_id=EVENING.id)]
    daytime = [item("Senior Project Supervision", timeslot_id=MORNING.id)]

    assert calculate_fitness(evening, rooms, total_sections=1, timeslots=timeslots) > \
           calculate_fitness(daytime, rooms, total_sections=1, timeslots=timeslots)


def test_fitness_prefers_the_daytime_placement_for_teaching():
    from backend.Optimization.evaluation import calculate_fitness

    timeslots = [MORNING, EVENING]
    rooms = [Room({"room_id": 1, "capacity": 40, "room_type": "classroom",
                   "building": "M8", "dept_id": 1, "room_num": "1"})]

    daytime = [item("Lecture Undergraduate", timeslot_id=MORNING.id)]
    evening = [item("Lecture Undergraduate", timeslot_id=EVENING.id)]

    assert calculate_fitness(daytime, rooms, total_sections=1, timeslots=timeslots) > \
           calculate_fitness(evening, rooms, total_sections=1, timeslots=timeslots)


def test_fitness_without_timeslots_is_unchanged():
    """Callers that pass no timeslot list keep their old score.

    tests/test_evaluation.py and main.py both call it that way; the time rule
    needs the Timeslot objects to read a start time, so it simply does not apply.
    """
    from backend.Optimization.evaluation import calculate_fitness

    rooms = [Room({"room_id": 1, "capacity": 40, "room_type": "classroom",
                   "building": "M8", "dept_id": 1, "room_num": "1"})]
    bad = [item("Senior Project Supervision", timeslot_id=MORNING.id)]
    assert calculate_fitness(bad, rooms, total_sections=1) == \
           calculate_fitness(bad, rooms, total_sections=1, timeslots=None)


def test_total_time_penalty_sums_only_timed_items():
    from backend.Optimization.evaluation import total_time_penalty

    schedule = [
        item("Senior Project Supervision", course_id=1, timeslot_id=MORNING.id),
        item("Senior Project Supervision", course_id=2, timeslot_id=None),
        item("Lecture Undergraduate", course_id=3, timeslot_id=EVENING.id),
    ]
    assert total_time_penalty(schedule, [MORNING, EVENING]) == (
        SOFT_WEIGHT_SUPERVISION_DAYTIME + SOFT_WEIGHT_TEACHING_EVENING
    )


def test_soft_violation_counts_reports_the_time_rules():
    """Counted over TIMED items, not roomed ones.

    Supervision and clinical practice hold an hour and no room at all, and they
    are most of the population this rule is about -- keying the count off the
    room map, the way the room rules are counted, would have measured almost
    none of them.
    """
    from backend.Optimization.evaluation import soft_violation_counts

    roomless = item("Senior Project Supervision", timeslot_id=MORNING.id)
    counts = soft_violation_counts([roomless], rooms=[], timeslots=[MORNING])
    assert counts["supervision_daytime"] == 1
    assert counts["teaching_evening"] == 0


def test_soft_violation_counts_without_timeslots_still_works():
    from backend.Optimization.evaluation import soft_violation_counts

    counts = soft_violation_counts([item("Lecture Undergraduate")], rooms=[])
    assert counts["supervision_daytime"] == 0
    assert counts["teaching_evening"] == 0


def test_grasp_item_cost_includes_the_time_rule():
    """GRASP must optimize the same objective it is judged by.

    Its local search accepts a move by comparing deltas from
    grasp._item_soft_penalty(), then reports a fitness from
    evaluation.calculate_fitness(). A soft term present in one and missing from
    the other makes the two disagree silently: the search would keep "improving"
    a schedule by a measure nobody sees, and the reported score would drift away
    from the one being optimised.

    HANDOVER section 6 lists this class of bug -- anything tracking cost
    incrementally must build, increment and decrement identically.
    """
    from backend.Optimization.Algorithims.grasp import _item_soft_penalty

    room = Room({"room_id": 1, "capacity": 40, "room_type": "classroom",
                 "building": "M8", "dept_id": 1, "room_num": "1"})
    supervision = item("Senior Project Supervision")
    timeslot_map = {MORNING.id: MORNING, EVENING.id: EVENING}
    # Both slots valid, so only the time preference can separate them.
    cache = {(supervision.course_id, supervision.section): {MORNING.id, EVENING.id}}

    daytime = _item_soft_penalty(supervision, room, MORNING.id, cache, timeslot_map)
    evening = _item_soft_penalty(supervision, room, EVENING.id, cache, timeslot_map)

    assert daytime - evening == SOFT_WEIGHT_SUPERVISION_DAYTIME
