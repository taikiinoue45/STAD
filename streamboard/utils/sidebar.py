import streamlit as st

from pathlib import Path


def sidebar(base: Path):

    experiment = st.sidebar.selectbox("Experiment Name", sorted([p.stem for p in base.glob("*")]))

    base = base / experiment

    condition = st.sidebar.selectbox(
        "Condition ID", sorted([p.stem for p in base.glob("outputs/*/*/")])
    )

    return [p for p in base.glob(f"outputs/*/{condition}")][0]
