import time

import streamlit as st
import pandas as pd
import networkx as nx

from backend.database.loader import get_scheduling_data, save_schedule, load_schedule
from backend.Optimization.constraints import tag_section_links, tag_intro_it_pairs
from backend.Optimization.evaluation import calculate_fitness, count_conflicts, debug_conflicts_ui
from backend.Optimization.engine import SchedulingEngine


#helper 
def preview_objects(title, objects):
    if not objects:
        st.write(f"No {title} available")
        return
    rows = []
    for obj in objects:
        row = {}
        for key, value in vars(obj).items():
            if isinstance(value, (str, int, float, bool, type(None))):
                row[key] = value
            elif hasattr(value, "__dict__"):
                nested = vars(value)
                row[key] = nested.get("name") or nested.get("id") or str(value)
            else:
                row[key] = str(value)
        rows.append(row)
    df = pd.DataFrame(rows)
    st.markdown(f"### {title} ({len(df)})")
    st.dataframe(df)

def schedule_to_dataframe(schedule, rooms, timeslots):
    room_map = {room.id: room for room in rooms}
    timeslot_map = {ts.id: ts for ts in timeslots}
    rows = []
    for item in schedule:
        room = room_map.get(item.room_id)
        timeslot = timeslot_map.get(item.timeslot_id)
        rows.append(
            {
                "course_id": item.course_id,
                "course_name": item.course_name,
                "section": item.section,
                "instructor_id": item.instructor_id,
                "room_type": room.type if room else None,
                "course_type":item.course_type,
                "room_no":room.no if room else None,
                "course_dept":item.course_dept,
                "room_dept":room.dept_id if room else None,
                "room_building": room.building if room else None,
                "day": timeslot.day if timeslot else None,
                "start": timeslot.start if timeslot else None,
                "end": timeslot.end if timeslot else None,
            }
        )
    return pd.DataFrame(rows)


#main
def main():
    st.set_page_config(layout="wide")
    st.title("University Timetabler")

    data = get_scheduling_data()
    if data is None:
        st.error("Failed to load data from database.")
        st.stop()

    # Run soft-constraint linking once after load (annotates section objects in-place)
    tag_section_links(data["sections"])
    tag_intro_it_pairs(data["sections"])

    st.markdown("## Loaded Database Data")
    preview_objects("Rooms", data["rooms"])
    preview_objects("Timeslots", data["timeslots"])
    preview_objects("Sections", data["sections"])

    # Choose an algorithm via buttons and store selection in Streamlit session state
    if "algorithm_name" not in st.session_state:
        st.session_state.algorithm_name = "greedy"

    col1, col2, col3 = st.columns(3)
    if col1.button("greedy"):
        st.session_state.algorithm_name = "greedy"
    if col2.button("genetic"):
        st.session_state.algorithm_name = "genetic"
    if col3.button("Load Base Schedule"):
        st.session_state.algorithm_name = "base"

    algorithm_name = st.session_state.algorithm_name
    st.markdown(f"**Selected algorithm:** {algorithm_name}")

    if algorithm_name == "base":
        with st.spinner("Loading base schedule..."):
            schedule_items = load_schedule(1)
            scheduled = len(schedule_items)
            unscheduled = 0
            exec_time = 0.0
    else:
        engine = SchedulingEngine(algorithm_name)
        with st.spinner(f"Running {algorithm_name} scheduling..."):
            start_time = time.perf_counter()
            schedule_items, scheduled, unscheduled = engine.run(data)
            exec_time = time.perf_counter() - start_time

    #retreive base schedule and apply fittness

    st.markdown("## Final Schedule")

    schedule_df = schedule_to_dataframe(
        schedule_items,
        data["rooms"],
        data["timeslots"],
    )

    conflict_count = count_conflicts(schedule_items, data["rooms"], data["sections"], data["timeslots"])
    fitness_score = calculate_fitness(
        schedule_items,
        data["rooms"],
        sections=data["sections"],
        timeslots=data["timeslots"],
    )
    debug_conflicts_ui(schedule_items, data["rooms"], data["sections"], data["timeslots"])
    st.success(
        f"Done — {scheduled} items scheduled, "
        f"{unscheduled} unscheduled, "
        f"{conflict_count} conflict(s), "
        f"fitness={fitness_score:.1f}, "
        f"time={exec_time:.3f}s"
    )
    st.dataframe(schedule_df)
    st.download_button(
        "Download Schedule CSV",
        schedule_df.to_csv(index=False),
        file_name="schedule.csv",
    )
    # admin base credits
    
    sch_name = st.text_input("Enter a name for schedule", value="Base Schedule" if algorithm_name == "base" else "")
    schedule_details = {
        "sch_name": sch_name,
        "alg": algorithm_name,
        "semester": "202510",
        "fitness_score": fitness_score,
        "conflicts": conflict_count,
        "user_id": 3,
        "exec_time": exec_time,
        "rule_set": None,
        "Scheduled": scheduled,
        "unscheduled":unscheduled
    }

    if st.button("Save Schedule"):
        try:
            save_result = save_schedule(schedule_items, schedule_details)
            st.success(f"Schedule saved successfully: {save_result}")
        except Exception as exc:
            st.error(f"Failed to save schedule: {exc}")


if __name__ == "__main__":
    main()