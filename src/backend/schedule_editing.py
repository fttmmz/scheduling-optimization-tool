"""Hand-editing a saved schedule: what is wrong with it, and where a class could go instead.

Two questions the Edit page asks, both answered with the same rules the evaluator counts by:

  find_problems()   which items are in a hard conflict, and which are unscheduled -- by item index,
                    so the page can name the course instead of only printing a count
  find_placements() every (room, timeslot) where one item could sit without any hard conflict,
                    cheapest soft cost first

Items are identified by their index in load_schedule() order, which is the order GET
/api/schedules/{id} returns them in (the frontend's entry `key`). An edit is
{"idx": i, "room_id": r, "timeslot_id": t}; apply_edits() copies the items before changing them, so
the cached copy of a saved schedule is never modified.

The counts in evaluate_schedule() stay the authority on "how many". find_problems() lists the
clashing PAIRS behind them, so a slot holding three classes shows as three pairs where the count says
two; every item involved is listed either way, which is what the page needs.
"""
import copy
from collections import defaultdict

from backend.Optimization.constraints import (
    classify_section,
    group_siblings,
    instructor_occupancy_ids,
    is_placeholder_slot,
    item_needs,
    get_valid_timeslots_for_section,
    rank_rooms,
    room_rule_violations,
    room_soft_penalty,
    same_class,
    sibling_key,
    slots_clash,
    time_soft_penalty,
)
from backend.Optimization.constraints.timeslots import is_placeholder_timeslot

ROOM_RULE_TEXT = {
    "room_type": "wrong room type",
    "lab_department": "another department's lab",
    "medical_campus": "wrong side of the medical campus line",
}


def apply_edits(items, edits):
    """A copy of *items* with each edit applied. Raises ValueError on an index out of range."""
    edited = list(items)
    for edit in edits or []:
        idx = int(edit["idx"])
        if not 0 <= idx < len(edited):
            raise ValueError(f"No item {idx} in this schedule")
        item = copy.copy(edited[idx])
        item.room_id = edit.get("room_id")
        item.timeslot_id = edit.get("timeslot_id")
        edited[idx] = item
    return edited


def unscheduled_indexes(items, sections):
    """Indexes of the items count_scheduled_sections() counts as not scheduled (same rule)."""
    lookup = {(s.course.id, s.no): s for s in sections}
    out = []
    for i, item in enumerate(items):
        if item.room_id is not None or item.timeslot_id is not None:
            continue
        section = lookup.get((item.course_id, item.section))
        if section is not None and classify_section(section) == "NEEDS_NOTHING":
            continue
        out.append(i)
    return out


def _clashing_pairs(indexed, exempt):
    """(i, j) pairs among [(idx, item)] whose slots clash in real time, minus exempt(a, b)."""
    pairs = []
    for n, (i, a) in enumerate(indexed):
        for j, b in indexed[n + 1:]:
            if slots_clash(a.timeslot_id, b.timeslot_id) and not exempt(a, b):
                pairs.append((i, j))
    return pairs


def find_problems(items, rooms, sections, max_hard=400, max_unscheduled=1000):
    """{"hard": [...], "unscheduled": [idx, ...], "hard_by_kind": {...}, "truncated": bool} for *items*.

    Each hard entry is {"kind", "items": [idx, ...], "resource"}: kind is room, instructor, sibling or
    room_rule; resource is the room id, the instructor id, or the broken rule's name. The lists are
    capped (an old save can hold tens of thousands of clashing pairs, more than a page can show);
    hard_by_kind always counts all of them.
    """
    found = _all_problems(items, rooms, sections)
    hard, unscheduled = found["hard"], found["unscheduled"]
    by_kind = defaultdict(list)
    for entry in hard:
        by_kind[entry["kind"]].append(entry)
    # When capping, take the kinds in turn, so thousands of room pairs can't hide every instructor clash.
    shown, depth = [], 0
    while len(shown) < min(max_hard, len(hard)):
        for entries in by_kind.values():
            if depth < len(entries) and len(shown) < max_hard:
                shown.append(entries[depth])
        depth += 1
    return {
        "hard": shown,
        "unscheduled": unscheduled[:max_unscheduled],
        "hard_by_kind": {kind: len(entries) for kind, entries in by_kind.items()},
        "hard_pairs": len(hard),
        "unscheduled_total": len(unscheduled),
        "truncated": len(hard) > max_hard or len(unscheduled) > max_unscheduled,
    }


def _all_problems(items, rooms, sections):
    # Placeholder slots (TBA) are no hour, so classes there can't clash: the same rule as the evaluator.
    timed = [(i, it) for i, it in enumerate(items)
             if it.timeslot_id is not None and not is_placeholder_slot(it.timeslot_id)]
    hard = []

    by_room = defaultdict(list)
    for i, it in timed:
        if it.room_id is not None:
            by_room[it.room_id].append((i, it))
    for room_id, group in by_room.items():
        for i, j in _clashing_pairs(group, same_class):
            hard.append({"kind": "room", "items": [i, j], "resource": room_id})

    by_instructor = defaultdict(list)
    for i, it in timed:
        for inst in instructor_occupancy_ids(it):
            by_instructor[inst].append((i, it))
    for inst, group in by_instructor.items():
        for i, j in _clashing_pairs(group, same_class):
            hard.append({"kind": "instructor", "items": [i, j], "resource": inst})

    by_sibling = defaultdict(list)
    for i, it in timed:
        by_sibling[sibling_key(it)].append((i, it))
    for group in by_sibling.values():
        if len(group) > 1:
            for i, j in _clashing_pairs(group, lambda a, b: False):
                hard.append({"kind": "sibling", "items": [i, j], "resource": None})

    room_map = {r.id: r for r in rooms}
    for i, it in enumerate(items):
        room = room_map.get(it.room_id)
        if room is not None:
            for rule in room_rule_violations(it, room):
                hard.append({"kind": "room_rule", "items": [i], "resource": rule,
                             "text": ROOM_RULE_TEXT.get(rule, rule)})

    return {"hard": hard, "unscheduled": unscheduled_indexes(items, sections)}


def _section_for(item, sections):
    """The Section an item was built from: same course, section and meeting block if possible."""
    same = [s for s in sections if s.course.id == item.course_id and str(s.no) == str(item.section)]
    for s in same:
        if s.pattern_index == getattr(item, "pattern_index", 0):
            return s
    return same[0] if same else None


def find_placements(items, idx, rooms, timeslots, sections, limit=60, rooms_per_time=12):
    """Where item *idx* could go with zero hard conflicts against every OTHER item, best first.

    Returns {"needs": {"room", "time"}, "candidates": [...], "free_total": n, "checked": n}, one
    candidate per free timeslot: {"timeslot_id", "cost", "free_rooms", "rooms": [{"room_id", "cost",
    "same_room_as_sibling"}, ...]}, the cheapest *rooms_per_time* rooms of that time. room_id is None
    for a class that needs a time but no room. free_total counts every free (room, time) pair.
    """
    target = items[idx]
    needs = item_needs(target)
    result = {"needs": {"room": needs.room, "time": needs.time}, "candidates": [], "free_total": 0, "checked": 0}
    if not needs.time:
        return result          # nothing the editor can place: it needs no hour

    section = _section_for(target, sections)
    slot_list = get_valid_timeslots_for_section(section, timeslots) if section else list(timeslots)
    slot_list = [ts for ts in slot_list if not is_placeholder_timeslot(ts)]
    room_list = rank_rooms(target, rooms, limit=None) if needs.room else [None]

    others = [(i, it) for i, it in enumerate(items) if i != idx and it.timeslot_id is not None]
    room_held = defaultdict(list)
    inst_held = defaultdict(list)
    sibling_slots, sibling_rooms = [], set()
    target_sibling = sibling_key(target)
    target_teachers = set(instructor_occupancy_ids(target))
    for _, it in others:
        if it.room_id is not None:
            room_held[it.room_id].append(it)
        for inst in instructor_occupancy_ids(it):
            if inst in target_teachers:
                inst_held[inst].append(it)
        if sibling_key(it) == target_sibling:
            sibling_slots.append(it.timeslot_id)
            if it.room_id is not None:
                sibling_rooms.add(it.room_id)

    def busy(held, slot):
        return any(slots_clash(slot, it.timeslot_id) and not same_class(it, target) for it in held)

    times = []
    for ts in slot_list:
        if any(slots_clash(ts.id, s) for s in sibling_slots):
            continue
        if any(busy(inst_held[inst], ts.id) for inst in target_teachers):
            continue
        time_cost = time_soft_penalty(target, ts)
        free = []
        for room in room_list:
            result["checked"] += 1
            if room is not None and busy(room_held[room.id], ts.id):
                continue
            room_cost = room_soft_penalty(target, room) if room is not None else 0.0
            free.append({
                "room_id": room.id if room is not None else None,
                "cost": round(room_cost + time_cost, 3),
                "same_room_as_sibling": room is not None and room.id in sibling_rooms,
            })
        if not free:
            continue
        # Cheapest first; a sibling's room first among equals (blocks of one section share a room).
        free.sort(key=lambda c: (c["cost"], not c["same_room_as_sibling"]))
        times.append({"timeslot_id": ts.id, "cost": free[0]["cost"],
                      "free_rooms": len(free), "rooms": free[:rooms_per_time]})

    # Stable: equally cheap times keep the registrar's preferred order from get_valid_timeslots.
    times.sort(key=lambda t: t["cost"])
    result["free_total"] = sum(t["free_rooms"] for t in times)
    result["candidates"] = times[:limit]
    return result
