import streamlit as st
import streamboard.utils

from pathlib import Path


st.title("Student-Teacher Anomaly Detection")
base = Path("/dgx/github/STAD/stad/experiments/")
base = streamboard.utils.sidebar(base)


add_selectbox = st.sidebar.selectbox(
    "Visualization",
    (
        "show_test_normal_results",
        "show_test_anomaly_results",
        "show_val_results",
        "show_probabilistic_crop",
        "show_yaml",
        "show_metric_table",
    ),
)

st.header(" ")
st.header(" ")
fn = getattr(streamboard.utils, add_selectbox)
fn(base)
