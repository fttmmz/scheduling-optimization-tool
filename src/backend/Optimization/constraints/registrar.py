"""The registrar's own scheduling practice, learned once from the base (manual) schedule.

Production's model gives every course of a type the same rule (a lecture is MW/TR 75 min in a classroom,
a lab one day 170 min in its department's lab) and offers every section all 466 time slots, most of them
one-off variants that overlap the standard periods in real time. The manual schedule shows how the
registrar actually works, and this module carries that practice into the constraints:

  Slot grid        one-period classes at 08:00, 09:30, 11:00, 12:30, 14:00, 15:30, 17:00, 18:30 and longer
                   blocks at 08:00, 11:00, 14:00, 17:00; one slot per family of near-duplicates (same days,
                   start within 10 min, end within 15), the variant the registrar uses most. A campus listed
                   in institution_data.OWN_SLOT_GRID_CAMPUS_CODES keeps its own slots; graduate classes also get the evening blocks the registrar uses
                   at least 3 times for that course type and level; long (4h+) or unusual-day blocks only go to
                   the courses the registrar put there.
  Per course       each course meets in its own formats (days per week x minutes: 2x75, 2x50, 1x100 ...),
                   uses its own room types, lab departments and departments. An L/X/Y section of a lecture
                   course is that course's LAB; a Laboratory section with no suffix is taught as a lecture;
                   a T section is a tutorial (one day, 50 min, evening preferred).
  Demand           a meeting the manual leaves TBA is not timetabled; one it gives a time but no room needs
                   a time only.
  Team programme   the team-taught courses (institution_data.TEAM_ROSTER_COURSE_IDS) sit in their manual
                   slot and room.
  Soft costs       room type costs 0 (an allowed mismatch is policy, a forbidden one is already hard); no
                   campus cost for main-campus graduate classes (placed anywhere, in one graduate building
                   mostly) or for lab sections in the lab buildings every campus shares
                   (institution_data.SHARED_LAB_BUILDINGS).

A practice counts only when the registrar uses it for at least 2 of a course's sections or at least half of
them, so a one-off stays a one-off. Measured effect (PAPER_RESULTS_DRAFT.md section 00): with the real-time
clash rule, every algorithm reaches 0 or almost 0 hard conflicts and almost 0 unscheduled, against
100-275 unscheduled on the 466-slot model.

Life cycle: scripts/data_import/build_registrar_practice.py learns it (learn()) and stores the JSON in the
`registrar_practice` table; database/loader.py loads and activate()s it. While nothing is active every hook
returns production's answer unchanged -- tests and scripts keep the old model unless they load one.
"""
from collections import Counter, defaultdict
from itertools import combinations

from . import institution_data as _institution

PERIOD = 90
PERIOD_STARTS = range(8 * 60, 19 * 60, PERIOD)          # 08:00, 09:30, ... 18:30
BLOCK_STARTS = (8 * 60, 11 * 60, 14 * 60, 17 * 60)      # where blocks longer than one period start
START_TOLERANCE, END_SPILL = 5, 15
STANDARD_DAYS = {"M", "T", "W", "R", "MW", "TR"}
LONG_BLOCK = 240                # minutes; at or above this a slot is a special block
GRADUATE = {"Master", "Doctorate"}
MIN_EXTRA_USES = 3              # a graduate evening slot must carry this many manual blocks of its type/level
# Institution facts (institution_data.py, the Render secret file). getattr() so an older copy still imports;
# the rule is then off.
OWN_SLOT_GRID_CAMPUS_CODES = getattr(_institution, "OWN_SLOT_GRID_CAMPUS_CODES", frozenset())
SHARED_LAB_BUILDINGS = getattr(_institution, "SHARED_LAB_BUILDINGS", frozenset())
LAB_TYPES = {"Laboratory", "Lecture/Lab"}
LAB_SUFFIXES = {"L", "X", "Y"}
TUTORIAL_LENGTH = 50
LENGTH_TOLERANCE = 15           # as the near-duplicate families: end within 15 minutes
MIN_SECTIONS, MIN_SHARE = 2, 0.5

_ACTIVE = [None]


def active():
    """The loaded RegistrarPractice, or None."""
    return _ACTIVE[0]


def activate(practice):
    """Make *practice* (a RegistrarPractice or None) the one every hook uses."""
    _ACTIVE[0] = practice


# ── Identities ────────────────────────────────────────────────────────────────

def _minutes(value):
    from .timeslots import _parse_time_to_minutes
    return _parse_time_to_minutes(value)


def _duration(ts):
    from .timeslots import _timeslot_duration
    return _timeslot_duration(ts)


def _suffix(section_no):
    from .timeslots import _section_suffix
    return _section_suffix(section_no)


def _ident(obj):
    """(course id, course type, level, section no, pattern index) of a Section or a ScheduleItem."""
    course = getattr(obj, "course", None)
    if course is not None:
        return str(course.id), course.type, course.level, str(obj.no), getattr(obj, "pattern_index", 0)
    return (str(obj.course_id), obj.course_type, getattr(obj, "level", None), str(obj.section),
            getattr(obj, "pattern_index", 0))


def _block_key(obj):
    course_id, _type, _level, section_no, pattern = _ident(obj)
    return f"{course_id}|{section_no}|{pattern}"


def _is_lecture_type(course_type):
    return any(w in (course_type or "") for w in ("Lecture", "Lecutre", "Seminar"))


def section_kind(obj):
    """programme / graduate / tutorial / lab / connected_lab (an L/X/Y section of a lecture course) /
    lecture / other."""
    from .classification import TEAM_ROSTER_COURSE_IDS
    course_id, course_type, level, section_no, _pattern = _ident(obj)
    suffix = _suffix(section_no)
    if course_id in TEAM_ROSTER_COURSE_IDS:
        return "programme"
    if level in GRADUATE:
        return "graduate"
    if suffix == "T":
        return "tutorial"
    if suffix in LAB_SUFFIXES:
        return "lab" if course_type in LAB_TYPES else "connected_lab"
    if course_type in LAB_TYPES or _is_lecture_type(course_type):
        return "lecture"
    return "other"


def effective_course_type(obj):
    """The course type the room rules see: an L/X/Y section of a lecture course is a lab; a Laboratory
    section with no suffix is taught as a lecture."""
    _course_id, course_type, level, section_no, _pattern = _ident(obj)
    suffix = _suffix(section_no)
    if suffix in LAB_SUFFIXES and _is_lecture_type(course_type):
        return "Laboratory"
    if course_type in LAB_TYPES and suffix == "":
        return "Lecture Graduate" if level in GRADUATE else "Lecture Undergraduate"
    return course_type


def _group(obj):
    course_id, *_rest = _ident(obj)
    return f"{course_id}|{section_kind(obj)}"


# ── The slot grid ─────────────────────────────────────────────────────────────

def on_grid(ts):
    """Starts on a period boundary and, if longer than one period, on a block boundary."""
    start, end = _minutes(ts.start), _minutes(ts.end)
    if start is None or end is None:
        return False
    period = next((p for p in PERIOD_STARTS if abs(start - p) <= START_TOLERANCE), None)
    if period is None:
        return False
    if end <= period + PERIOD + END_SPILL:
        return True
    return any(abs(start - b) <= START_TOLERANCE for b in BLOCK_STARTS)


def is_special(ts):
    """A long block (4h+) or an unusual day pattern (MTW, MTWR, UF, Sunday ...)."""
    return ts.day not in STANDARD_DAYS or _duration(ts) >= LONG_BLOCK


def near_duplicate_families(timeslots):
    """{slot id: family root}. Same days, starts within 10 min, ends within 15 min."""
    parent = {t.id: t.id for t in timeslots}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    times = {t.id: (_minutes(t.start), _minutes(t.end)) for t in timeslots}
    for a, b in combinations(timeslots, 2):
        (sa, ea), (sb, eb) = times[a.id], times[b.id]
        if (a.day == b.day and None not in (sa, ea, sb, eb)
                and abs(sa - sb) <= 10 and abs(ea - eb) <= 15):
            parent[find(a.id)] = find(b.id)
    return {t.id: find(t.id) for t in timeslots}


def _practice(counter):
    """The values used for at least MIN_SECTIONS sections or at least MIN_SHARE of them."""
    total = sum(counter.values())
    return sorted(v for v, n in counter.items() if n >= MIN_SECTIONS or n >= MIN_SHARE * total)


# ── Learning ──────────────────────────────────────────────────────────────────

def learn(sections, timeslots, rooms, manual_slot_ids, manual_room_ids):
    """The practice of one schedule, as a JSON-ready dict.

    sections, timeslots, rooms: as get_scheduling_data() loads them (real slots only). manual_slot_ids /
    manual_room_ids: each section's slot and room in the base schedule, in the same order (None or an id
    not in *timeslots* for a TBA meeting). Call with no practice active.
    """
    from .classification import TEAM_ROSTER_COURSE_IDS, section_needs
    if active() is not None:
        raise RuntimeError("learn() must run with no practice active")
    by_id = {t.id: t for t in timeslots}
    room_by_id = {r.id: r for r in rooms}
    triples = list(zip(sections, manual_slot_ids, manual_room_ids))

    use = Counter(t for t in manual_slot_ids if t in by_id)
    family = near_duplicate_families(timeslots)
    members = defaultdict(list)
    for t in timeslots:
        members[family[t.id]].append(t.id)
    rep = {root: min(ids, key=lambda i: (-use[i], i)) for root, ids in members.items()}
    rep_of = {t.id: rep[family[t.id]] for t in timeslots}
    grid = sorted(r for r in set(rep.values()) if on_grid(by_id[r]))

    pinned, fam, special_owners, demand = {}, set(), defaultdict(set), {}
    extra = defaultdict(Counter)
    formats, room_types = defaultdict(Counter), defaultdict(Counter)
    lab_depts, depts = defaultdict(Counter), defaultdict(Counter)
    for section, slot_id, room_id in triples:
        ts = by_id.get(slot_id)
        needs = section_needs(section)
        key = _block_key(section)
        if str(section.course.id) in TEAM_ROSTER_COURSE_IDS and ts is not None:
            pinned[key] = [slot_id, room_id if room_id in room_by_id else None]
        if ts is not None:
            if is_special(ts):
                special_owners[rep_of[slot_id]].add(str(section.course.id))
            if getattr(section, "campus", None) in OWN_SLOT_GRID_CAMPUS_CODES:
                fam.add(slot_id)
            elif section.course.level in GRADUATE and needs.room:
                extra[f"{section.course.type}|{section.course.level}"][rep_of[slot_id]] += 1
        if needs.time and ts is None:
            demand[key] = [False, False]                        # TBA: not timetabled
        elif needs.room and room_id is None and ts is not None:
            demand[key] = [False, needs.time]                   # time only
        group = _group(section)
        if ts is not None and ts.day in STANDARD_DAYS:
            formats[group][f"{len(ts.day)}|{_duration(ts)}"] += 1
        room = room_by_id.get(room_id)
        if room is not None:
            room_types[group][room.type] += 1
            if room.dept_id:
                depts[str(section.course.id)][room.dept_id] += 1
                if room.type == "lab":
                    lab_depts[str(section.course.id)][room.dept_id] += 1

    return {
        "version": 1,
        "rep_of": {str(k): v for k, v in rep_of.items()},
        "grid": grid,
        "fam": sorted(fam),
        "graduate_extra": {k: sorted((i for i, n in c.items() if n >= MIN_EXTRA_USES), key=lambda i: (-c[i], i))
                           for k, c in extra.items()},
        "special_owners": {str(k): sorted(v) for k, v in special_owners.items()},
        "pinned": pinned,
        "demand": demand,
        "formats": {g: [[int(f.split("|")[0]), int(f.split("|")[1])] for f in _practice(c)]
                    for g, c in formats.items()},
        "room_types": {g: _practice(c) for g, c in room_types.items()},
        "lab_depts": {g: _practice(c) for g, c in lab_depts.items()},
        "depts": {g: _practice(c) for g, c in depts.items()},
    }


# ── The loaded practice ───────────────────────────────────────────────────────

class RegistrarPractice:
    def __init__(self, data):
        self.data = data
        self.rep_of = {int(k): v for k, v in data["rep_of"].items()}
        self.grid = set(data["grid"])
        self.fam = set(data["fam"])
        self.fam_only = self.fam - self.grid
        self.graduate_extra = data["graduate_extra"]
        self.special_owners = {int(k): set(v) for k, v in data["special_owners"].items()}
        self.pinned = data["pinned"]
        self.demand = data["demand"]
        self.formats = data["formats"]
        self.room_types = {k: set(v) for k, v in data["room_types"].items()}
        self.lab_depts = {k: set(v) for k, v in data["lab_depts"].items()}
        self.depts = {k: set(v) for k, v in data["depts"].items()}
        self.families = defaultdict(set)
        for slot_id, rep in self.rep_of.items():
            self.families[rep].add(slot_id)

    # what a section needs ----------------------------------------------------
    def needs(self, obj, needs):
        override = self.demand.get(_block_key(obj))
        if override is None:
            return needs
        from .classification import Needs
        return Needs(room=override[0], time=override[1])

    # time slots ----------------------------------------------------------------
    def candidates(self, section, timeslots, production_candidates):
        """Slots of this course's formats; production's rule when it has none."""
        from .timeslots import DURATION_75, SINGLE_DAY
        fmts = self.formats.get(_group(section))
        if fmts:
            out = [t for t in timeslots if t.day in STANDARD_DAYS and any(
                len(t.day) == days and abs(_duration(t) - minutes) <= LENGTH_TOLERANCE for days, minutes in fmts)]
            if out:
                return out
        if _suffix(section.no) == "T":
            out = [t for t in timeslots if t.day in SINGLE_DAY
                   and abs(_duration(t) - TUTORIAL_LENGTH) <= LENGTH_TOLERANCE]
            if out:
                return out
        if section.course.type in LAB_TYPES and effective_course_type(section) != section.course.type:
            out = [t for t in timeslots if t.day in {"MW", "TR"} and abs(_duration(t) - DURATION_75) <= 10]
            if out:
                return out
        return production_candidates(section, timeslots)

    def _owned(self, section, slots):
        course_id = str(section.course.id)
        return [t for t in slots
                if not is_special(t) or course_id in self.special_owners.get(self.rep_of.get(t.id, t.id), ())]

    def allowed(self, section, ranked, timeslots):
        """Narrow a section's ranked slot list to the registrar's slots for its kind, keeping the order."""
        from .classification import section_needs
        pin = self.pinned.get(_block_key(section))
        if pin:
            slot = next((t for t in timeslots if t.id == pin[0]), None)
            if slot is not None:
                return [slot]
        if getattr(section, "campus", None) in OWN_SLOT_GRID_CAMPUS_CODES:
            out = [t for t in ranked if t.id in self.fam]
        elif not section_needs(section).room:
            out = [t for t in ranked if t.id not in self.fam_only]
        else:
            out = [t for t in ranked if t.id in self.grid]
            if section.course.level in GRADUATE:
                have = {t.id for t in out}
                by_id = {t.id: t for t in timeslots}
                out += [by_id[i] for i in self.graduate_extra.get(f"{section.course.type}|{section.course.level}", ())
                        if i not in have and i in by_id]
        out = self._owned(section, out)
        if not out:
            out = self._owned(section, [t for t in ranked if t.id not in self.fam_only]) or list(ranked)
        return out

    def near_duplicates(self, slot_ids):
        """slot_ids plus every slot in the same near-duplicate family (08:00-10:40 and 08:00-10:50 are one
        period), so a schedule is not charged for the variant of an allowed slot."""
        out = set(slot_ids)
        for slot_id in slot_ids:
            out |= self.families.get(self.rep_of.get(slot_id, slot_id), set())
        return out

    # rooms ---------------------------------------------------------------------
    def pinned_room_id(self, obj):
        pin = self.pinned.get(_block_key(obj))
        return pin[1] if pin else None

    def room_rules(self, obj, room, broken):
        """Drop the room-type and lab-department rules where the course's practice uses that room."""
        if not broken:
            return broken
        course_id = _ident(obj)[0]
        keep = []
        for rule in broken:
            if rule == "room_type" and room.type in self.room_types.get(_group(obj), ()):
                continue
            if rule == "lab_department" and room.dept_id in self.lab_depts.get(course_id, ()):
                continue
            keep.append(rule)
        return tuple(keep)

    def room_parts(self, obj, room, parts, profile):
        from .rooms import get_building_campus, get_required_room_type
        parts["room_type"] = 0.0
        if parts["department"] and room.dept_id in self.depts.get(_ident(obj)[0], ()):
            parts["department"] = 0.0
        if parts["campus"]:
            campus, building = profile[3], str(room.building).upper()
            if campus == "MAIN" and _ident(obj)[2] in GRADUATE and get_building_campus(building) != "MEDICAL":
                parts["campus"] = 0.0
            elif (building in SHARED_LAB_BUILDINGS and room.type == "lab"
                  and get_required_room_type(profile[0]) == "lab"):
                parts["campus"] = 0.0
        return parts

    # time preference -----------------------------------------------------------
    @staticmethod
    def time_parts(obj, ts, parts):
        """Tutorials prefer the evening (16 of 19 meet at 17:00): a daytime tutorial costs what daytime
        supervision costs, and an evening one is not charged as teaching in the evening."""
        from .timeslots import SOFT_WEIGHT_SUPERVISION_DAYTIME, is_evening_timeslot
        if ts is None or _suffix(_ident(obj)[3]) != "T":
            return parts
        return {"supervision_daytime": 0.0 if is_evening_timeslot(ts) else SOFT_WEIGHT_SUPERVISION_DAYTIME,
                "teaching_evening": 0.0}


# ── Hooks: production's answer unchanged while no practice is active ──────────

def needs(obj, needs_):
    practice = active()
    return needs_ if practice is None else practice.needs(obj, needs_)


def course_type_for_rooms(obj, course_type):
    return course_type if active() is None else effective_course_type(obj)


def timeslots_for(section, timeslots, production_candidates, rank):
    """production: rank(candidates). With a practice: the course's formats, ranked, narrowed to the grid."""
    practice = active()
    if practice is None:
        return rank(section, production_candidates(section, timeslots))
    ranked = rank(section, practice.candidates(section, timeslots, production_candidates))
    return practice.allowed(section, ranked, timeslots)


def room_rules(obj, room, broken):
    practice = active()
    return broken if practice is None else practice.room_rules(obj, room, broken)


def room_parts(obj, room, parts, profile):
    practice = active()
    return parts if practice is None else practice.room_parts(obj, room, parts, profile)


def time_parts(obj, ts, parts):
    practice = active()
    return parts if practice is None else practice.time_parts(obj, ts, parts)


def pinned_rooms(obj, rooms, ranked):
    """The programme's manual room when it has one; otherwise the ranked list unchanged."""
    practice = active()
    if practice is None:
        return ranked
    room_id = practice.pinned_room_id(obj)
    if room_id is None:
        return ranked
    room = next((r for r in rooms if r.id == room_id), None)
    return [room] if room is not None else ranked


def guideline_ids(slot_ids):
    practice = active()
    return slot_ids if practice is None else practice.near_duplicates(slot_ids)


def room_cache_key(section):
    """Greedy caches room lists by course type; a practice is per course, so the key needs the course."""
    return () if active() is None else (str(section.course.id), str(section.no))
