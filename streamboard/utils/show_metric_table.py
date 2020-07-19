import streamlit as st
import pandas as pd

from pathlib import Path
from omegaconf import OmegaConf


def show_metric_table(base):

    di = {}
    for p in sorted(base.parent.glob('*/hydra/overrides.yaml')):
        for l in OmegaConf.load(str(p)):
            k, v = l.split('=')

            if k in di.keys():
                di[k].append(v)
            else:
                di[k] = [v]

        with open(base / 'mIoU.txt', 'r') as f:
            for l in f.readlines():
                l = l.replace('\n', '')
                k, v = l.split('_')
            
            if k in di.keys():
                di[k].append(v)
            else:
                di[k] = [v]

    st.dataframe(pd.DataFrame(di))


