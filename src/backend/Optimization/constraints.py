#function to check instructor conflicts
def check_instructors(schedule,data=None):

    seen = set()

    for item in schedule:
        key = (item.instructor_id, item.timeslot_id)

        if key in seen:
            return False  # conflict found

        seen.add(key)

    return True  #no conflicts

#function to check room confilct
def check_room(schedule,data=None):

    seen = set()

    for item in schedule:
        key = (item.room_id, item.timeslot_id)

        if key in seen:
            return False  # conflict found

        seen.add(key)

    return True  #no conflicts

#functions to check campus conflict
def get_section_campus(section):
    section = int(section)

    if 1 <= section <= 29:
        return "MEN"
    elif 30 <= section <= 59:
        return "WOMEN"
    else:
        return "MIXED"
def get_building_campus(building):
    building = building.upper()

    if building.startswith("M"):
        num = int(building[1:])
        if 1 <= num <= 6:
            return "MEN"
        elif 7 <= num <= 10:
            return "MIXED"

    elif building.startswith("W"):
        num = int(building[1:])
        if 1 <= num <= 6:
            return "WOMEN"
        elif 7 <= num <= 10:
            return "MIXED"

    return "MEDICAL"

def check_campus(schedule,rooms):

    # Build room lookup
    room_map = {room.room_id: room for room in rooms}

    for item in schedule:
        section_campus = get_section_campus(item.section)

        room = room_map.get(item.room_id)
        if not room:
            raise KeyError(f"Room ID {item.room_id} not found")

        building_campus = get_building_campus(room.building)

        # Constraint check
        if section_campus != building_campus:
            return False

    return True
#check all constraints
def check_constraints(schedule, data, constraints):
    return all(constraint(schedule, data) for constraint in constraints)
