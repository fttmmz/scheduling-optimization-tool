"""Real-time clashes between time slots.

Occupancy everywhere is keyed by slot ID: a room is busy at (room, slot id). Two DIFFERENT slots can still
overlap in real time -- `M 09:30-10:45` and `MW 09:30-10:45`, or `R 08:00-11:59` and `R 08:00-12:00` -- and
the slot-ID test treats them as free. Measured on the real snapshot, every algorithm produced 800 to 2,400
such clashes per run while the evaluator reported 0 (PAPER_RESULTS_DRAFT.md section 00).

Two slots CLASH when they are the same slot, or share a day letter and at least one minute. Touching ends
(09:15 and 09:15) do not clash. A slot whose times are unknown (not in models.SLOT_TIMES) clashes only with
itself -- the old behaviour, so nothing breaks where a test or caller builds slots another way.

The three containers below replace the plain set / Counter / dict an algorithm keeps its occupancy in. They
answer the same questions with the clash rule instead of exact slot equality, so an algorithm's own search
code does not change:

    OverlapSet       (resource, slot) pairs; `(r, s) in busy` is True if r holds any slot clashing with s
    OverlapCounter   Counter of (resource, slot); overlapping(key) counts bookings on clashing slots
    OwnerMap         (resource, slot) -> index; get() returns the owner of a clashing booking
"""
from collections import Counter, defaultdict

from backend.models.models import SLOT_TIMES, SLOT_TIMES_VERSION

from .classification import occupancy_key
from .timeslots import _parse_time_to_minutes

MULTIPLE_OWNERS = object()      # OwnerMap.get(): more than one class holds a clashing slot

_cache = {"version": None, "parsed": {}, "clashing": {}}


def _check_cache():
    if _cache["version"] != SLOT_TIMES_VERSION[0]:
        _cache.update(version=SLOT_TIMES_VERSION[0], parsed={}, clashing={})


def _parsed(slot_id):
    """(set of day letters, start minute, end minute) or None when unknown or unreadable."""
    parsed = _cache["parsed"]
    if slot_id not in parsed:
        times = SLOT_TIMES.get(slot_id)
        value = None
        if times is not None:
            day, start, end = times
            start, end = _parse_time_to_minutes(start), _parse_time_to_minutes(end)
            if day and start is not None and end is not None:
                value = (frozenset(str(day)), start, end)
        parsed[slot_id] = value
    return parsed[slot_id]


def overlap_minutes(slot_a, slot_b):
    """Minutes two slots (by id) share on a common day; 0 if none or unknown."""
    _check_cache()
    a, b = _parsed(slot_a), _parsed(slot_b)
    if a is None or b is None or not (a[0] & b[0]):
        return 0
    return max(0, min(a[2], b[2]) - max(a[1], b[1]))


def slots_clash(slot_a, slot_b):
    """True when the two slots are the same slot or overlap in real time."""
    if slot_a is None or slot_b is None:
        return False
    return slot_a == slot_b or overlap_minutes(slot_a, slot_b) > 0


def clashing_slots(slot_id):
    """Every known slot id that clashes with slot_id, itself included."""
    _check_cache()
    clashing = _cache["clashing"]
    if slot_id not in clashing:
        mine = _parsed(slot_id)
        if mine is None:
            clashing[slot_id] = frozenset((slot_id,))
        else:
            clashing[slot_id] = frozenset(
                [slot_id] + [other for other in SLOT_TIMES
                             if other != slot_id and overlap_minutes(slot_id, other) > 0])
    return clashing[slot_id]


def _clashes_any(slot_id, held):
    """Does any slot in *held* (a container of slot ids) clash with slot_id?"""
    if not held:
        return False
    if slot_id in held:
        return True
    clash = clashing_slots(slot_id)
    if len(held) < len(clash):
        return any(other in clash for other in held)
    return any(other in held for other in clash)


class OverlapSet(set):
    """A set of (resource, slot id) pairs whose `in` uses the clash rule.

    Stores exactly what a plain set would; only membership changes. Anything that is not a
    (resource, slot) pair behaves as in a plain set.
    """

    def __init__(self, items=()):
        super().__init__()
        self._held = defaultdict(Counter)
        for item in items:
            self.add(item)

    def _index(self, item, delta):
        if isinstance(item, tuple) and len(item) == 2:
            held = self._held[item[0]]
            held[item[1]] += delta
            if held[item[1]] <= 0:
                del held[item[1]]

    def add(self, item):
        if not set.__contains__(self, item):
            set.add(self, item)
            self._index(item, 1)

    def discard(self, item):
        if set.__contains__(self, item):
            set.discard(self, item)
            self._index(item, -1)

    def remove(self, item):
        if not set.__contains__(self, item):
            raise KeyError(item)
        self.discard(item)

    def update(self, *others):
        for other in others:
            for item in other:
                self.add(item)

    def clear(self):
        set.clear(self)
        self._held.clear()

    def __contains__(self, item):
        if set.__contains__(self, item):
            return True
        if not (isinstance(item, tuple) and len(item) == 2) or item[1] is None:
            return False
        return _clashes_any(item[1], self._held.get(item[0]))

    def copy(self):
        return OverlapSet(self)

    def __sub__(self, other):
        drop = set(other)
        return OverlapSet(x for x in set.__iter__(self) if x not in drop)

    def __or__(self, other):
        out = OverlapSet(self)
        out.update(other)
        return out


class OverlapCounter(Counter):
    """A Counter of (resource, slot id) bookings. Indexing and updates are exact, as in a Counter;
    overlapping(key) counts the resource's bookings on every slot clashing with key's slot."""

    def __init__(self):
        super().__init__()
        self._held = defaultdict(Counter)

    def __setitem__(self, key, value):
        old = Counter.__getitem__(self, key) if key in self else 0
        super().__setitem__(key, value)
        self._held[key[0]][key[1]] += value - old

    def overlapping(self, key):
        held = self._held.get(key[0])
        if not held:
            return 0
        return sum(n for slot, n in held.items() if slots_clash(slot, key[1]))


class OwnerMap(dict):
    """(resource, slot id) -> owner index. get() looks at every clashing slot of the resource and returns
    its single owner, None if there is none, or MULTIPLE_OWNERS if several classes hold clashing slots."""

    def __init__(self):
        super().__init__()
        self._held = defaultdict(dict)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._held[key[0]][key[1]] = value

    def pop(self, key, *default):
        if key in self:
            self._held[key[0]].pop(key[1], None)
        return super().pop(key, *default)

    def get(self, key, default=None):
        held = self._held.get(key[0])
        if not held:
            return default
        owners = {owner for slot, owner in held.items() if slots_clash(slot, key[1])}
        if not owners:
            return default
        return owners.pop() if len(owners) == 1 else MULTIPLE_OWNERS


def single_owner(owner_map, resources, slot_id):
    """The one class blocking any of *resources* at slot_id: None, its index, or MULTIPLE_OWNERS."""
    owners = {owner_map.get((resource, slot_id)) for resource in resources} - {None}
    if not owners:
        return None
    return owners.pop() if len(owners) == 1 else MULTIPLE_OWNERS


def same_class(a, b):
    """Two items that are ONE class, not a clash between two: blocks of one section, or one cross-listed
    class taught under several course codes (same rule as evaluation._count_occupants)."""
    if a.course_id == b.course_id and str(a.section) == str(b.section):
        return True
    ka, kb = occupancy_key(a.course_id, a.section), occupancy_key(b.course_id, b.section)
    return ka == kb and ka[0] == "combined" and a.course_id != b.course_id


def cross_slot_pairs(groups, same_class=None):
    """Pairs of bookings of one resource on DIFFERENT slot ids that clash in real time.

    *groups* is an iterable of item lists, one list per resource (room, instructor, sibling group).
    Pairs on the same slot id are left to the existing slot-ID counts, so adding this to them counts each
    clashing pair once. same_class(a, b) excludes pairs that are not two different classes.
    """
    pairs = 0
    for items in groups:
        if len(items) < 2 or len({item.timeslot_id for item in items}) < 2:
            continue
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if (a.timeslot_id != b.timeslot_id and slots_clash(a.timeslot_id, b.timeslot_id)
                        and not (same_class and same_class(a, b))):
                    pairs += 1
    return pairs
