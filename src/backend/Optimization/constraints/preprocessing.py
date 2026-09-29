"""Section tagging run once by main.py before any algorithm.

NOTE: no algorithm reads these tags yet -- the Intro-to-IT day gap and the
tutorial/lab parent links are computed but not enforced."""

import re
from collections import defaultdict

from .timeslots import _is_intro_it, _section_suffix


# Day ordering used to enforce the "one day gap" rule for Intro-to-IT pairs
DAY_ORDER = {"M": 0, "T": 1, "W": 2, "R": 3, "F": 4}


def _days_apart(day_a: str, day_b: str) -> int:
    """
    Return the absolute calendar-day distance between two single-day codes.
    Returns 99 if either day is unknown.
    """
    a = DAY_ORDER.get(day_a, 99)
    b = DAY_ORDER.get(day_b, 99)
    return abs(a - b)


# ─────────────────────────────────────────────────────────────────────────────
# INTRO-TO-IT PAIRING HELPER
# Call once before scheduling to mark which Intro-to-IT sections are
# lecture vs. lab, and record their pairing so the algorithm can enforce
# the "≥2 calendar days apart" rule.
# ─────────────────────────────────────────────────────────────────────────────

def tag_intro_it_pairs(all_sections: list) -> None:
    """
    For every Intro-to-IT course, locate the base lecture section (e.g. '04')
    and its matching lab section (e.g. '04 Lab' or suffix 'L'), then annotate
    both in-place:

      section.intro_it_role         — 'lecture' | 'lab' | None
      section.intro_it_pair_id      — Section.id of the counterpart, or None

    The scheduling algorithm must then ensure the two paired sections are
    placed on days that are at least 2 apart (e.g. M and W are fine; M and T
    are not, because |0-1| = 1 < 2).

    Helper:  intro_it_days_ok(day_a, day_b) → bool
    """
    # Group Intro-to-IT sections by course_id and base number
    groups: dict[tuple, dict] = defaultdict(lambda: {"lecture": None, "lab": None})

    for sec in all_sections:
        if not _is_intro_it(sec):
            sec.intro_it_role    = None
            sec.intro_it_pair_id = None
            continue

        base_no = _base_section_no(str(sec.no))
        key     = (sec.course.id, base_no)
        suffix  = _section_suffix(sec.no)

        if suffix == "L" or "lab" in str(sec.no).lower():
            groups[key]["lab"] = sec
        else:
            groups[key]["lecture"] = sec

    for (course_id, base_no), pair in groups.items():
        lec = pair["lecture"]
        lab = pair["lab"]

        if lec:
            lec.intro_it_role    = "lecture"
            lec.intro_it_pair_id = lab.id if lab else None
        if lab:
            lab.intro_it_role    = "lab"
            lab.intro_it_pair_id = lec.id if lec else None


def intro_it_days_ok(day_a: str, day_b: str) -> bool:
    """
    Return True if the two single-day codes are at least 2 calendar days
    apart — enforcing the Intro-to-IT lec/lab separation rule.
    """
    return _days_apart(day_a, day_b) >= 2


# SECTION-LINKING PREPROCESSING  (run once, before any algorithm)

def _base_section_no(section_no: str) -> int:
    """Strip trailing letter(s) to get the base section number as an integer."""
    base_str = re.sub(r"[A-Za-z]+$", "", str(section_no)).strip()
    try:
        return int(base_str) if base_str else 0
    except ValueError:
        return 0


def tag_section_links(all_sections: list) -> None:
    """
    Annotate every Section object in-place with two optional attributes:

      section.tutorial_parent_id  — set on tutorial sections (e.g. '02T')
                                    points to the Section.id of the parent
                                    lecture with the same course + base number

      section.lab_parent_id       — set on lab sub-sections (e.g. '02X','02Y')
                                    points to the Section.id of the parent
                                    lecture with the same course + base number

    All other sections get both attributes set to None.

    SOFT constraints only — algorithms should TRY to honour these links
    but are not required to.
    """
    # Build lookup: (course_id, base_section_no) → Section
    base_lookup: dict[tuple, object] = {}
    for sec in all_sections:
        sno = str(sec.no)
        base_no = _base_section_no(sno)
        key = (sec.course.id, base_no)
        # Extract base string to check if section has no suffix
        base_str = re.sub(r"[A-Za-z]+$", "", sno).strip()
        # Only index sections that are themselves "base" (no trailing letter)
        if sno.strip() == base_str:
            base_lookup[key] = sec

    for sec in all_sections:
        sec.tutorial_parent_id = None
        sec.lab_parent_id = None

        sno = str(sec.no)
        base_no = _base_section_no(sno)
        base_str = re.sub(r"[A-Za-z]+$", "", sno).strip()
        suffix = sno[len(base_str):]  # whatever was stripped

        if not suffix:
            continue  # base section — no link needed

        parent = base_lookup.get((sec.course.id, base_no))
        parent_id = parent.id if parent else None  # None if parent missing

        if suffix.upper() == "T":
            sec.tutorial_parent_id = parent_id
        elif suffix.upper() in {"X", "Y"}:
            sec.lab_parent_id = parent_id
