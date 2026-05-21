import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="SPAI Challenge -CellVisionaries-",
    layout="wide"
)

st.title("Sperm Tracking Dashboard")

# =====================================================
# DUMMY VIDEO SELECTION
# Later: replace this with folders from tracked_output/
# =====================================================

video_names = [
    "video_001",
    "video_002",
    "video_003"
]

selected_video = st.selectbox(
    "Select video",
    video_names
)

st.divider()

# =====================================================
# LAYOUT
# =====================================================

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Tracked Video")

    # Later:
    # st.video(str(video_file))

    st.info("Video preview will be shown here.")

with right_col:
    st.subheader("Video Information")

    st.write(f"Selected video: `{selected_video}`")
    st.write("FPS: 30")
    st.write("Resolution: 1920 x 1080")
    st.write("Number of tracks: 128")

st.divider()

# =====================================================
# METRICS OVERVIEW
# =====================================================

st.subheader("Motility Metrics Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Mean VCL", "23.4")
col2.metric("Mean VSL", "12.8")
col3.metric("Mean VAP", "17.6")
col4.metric("Tracks", "128")

# =====================================================
# DUMMY TABLE
# =====================================================

dummy_metrics = pd.DataFrame({
    "track_id": [1, 2, 3, 4],
    "num_frames": [45, 61, 28, 73],
    "VCL": [22.4, 31.2, 18.9, 27.5],
    "VSL": [10.1, 18.4, 7.8, 15.2],
    "VAP": [15.3, 24.1, 11.2, 20.5],
    "mean_confidence": [0.81, 0.76, 0.69, 0.88],
})

st.dataframe(dummy_metrics, use_container_width=True)

# =====================================================
# TRAJECTORY PLOT PLACEHOLDER
# =====================================================

st.subheader("Trajectory Visualization")

selected_track = st.selectbox(
    "Select track ID",
    dummy_metrics["track_id"]
)

st.info(f"Trajectory plot for track `{selected_track}` will be shown here.")

# Later:
# plot x/y trajectory from trajectories.csv