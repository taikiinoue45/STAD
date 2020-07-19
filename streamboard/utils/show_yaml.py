import streamlit as st

from pathlib import Path
from omegaconf import OmegaConf



def show_yaml(base: Path):

    yaml_path = base / 'hydra/config.yaml'
    cfg = OmegaConf.load(str(yaml_path))
    st.markdown(f'```{cfg.pretty()}```')
