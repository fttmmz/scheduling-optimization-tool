import time

import streamlit as st

from backend.database.loader import get_scheduling_data, save_schedule
from backend.Optimization.constraints import (
    calculate_fitness,
    count_conflicts,
    tag_section_links,
)
from backend.Optimization.Algorithims.code3 import preview_objects, schedule_to_dataframe
from backend.Optimization.engine import SchedulingEngine


def main():
    st.set_page_config(layout="wide")
    st.title("University Timetabler")

    data = get_scheduling_data()
    if data is None:
        st.error("Failed to load data from database.")
        st.stop()

    # Run soft-constraint linking once after load (annotates section objects in-place)
    tag_section_links(data["sections"])

    st.markdown("## Loaded Database Data")
    preview_objects("Rooms", data["rooms"])
    preview_objects("Timeslots", data["timeslots"])
    preview_objects("Sections", data["sections"])

    # Choose an algorithm via buttons and store selection in Streamlit session state
    if "algorithm_name" not in st.session_state:
        st.session_state.algorithm_name = "greedy1"

    col1, col2, col3 = st.columns(3)
    if col1.button("greedy1"):
        st.session_state.algorithm_name = "greedy1"
    if col2.button("greedy2"):
        st.session_state.algorithm_name = "greedy2"
    if col3.button("genetic"):
        st.session_state.algorithm_name = "genetic"

    algorithm_name = st.session_state.algorithm_name
    st.markdown(f"**Selected algorithm:** {algorithm_name}")
    engine = SchedulingEngine(algorithm_name)


    with st.spinner(f"Running {algorithm_name} scheduling..."):
        start_time = time.perf_counter()
        schedule_items, scheduled, unscheduled = engine.run(data)
        exec_time = time.perf_counter() - start_time

    st.markdown("## Final Schedule")
    # build_conflict_graph is internal to greedy now; engine exposes the graph
    # if engine.last_graph is not None:
    #     render_conflict_graph(engine.last_graph, schedule_items)
    #     st.markdown(
    #         f"**Conflict Graph:** "
    #         f"{len(engine.last_graph.nodes)} nodes, "
    #         f"{len(engine.last_graph.edges)} edges"
    #     )

    schedule_df = schedule_to_dataframe(
        schedule_items,
        data["rooms"],
        data["timeslots"],
    )

    conflict_count = count_conflicts(schedule_items, data["rooms"])
    fitness_score = calculate_fitness(
        schedule_items,
        data["rooms"],
        scheduled,
        len(data["sections"]),
    )

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
    
    schedule_details = {
        "sch_name": input("enter a name for schedule"),
        "alg": algorithm_name,
        "semester": "202510",
        "fitness_score": fitness_score,
        "conflicts": conflict_count,
        "user_id": 3,
        "exec_time": exec_time,
        "rule_set": None,
    }

    if st.button("Save Schedule"):
        try:
            save_result = save_schedule(schedule_items, schedule_details)
            st.success(f"Schedule saved successfully: {save_result}")
        except Exception as exc:
            st.error(f"Failed to save schedule: {exc}")


if __name__ == "__main__":
    main()