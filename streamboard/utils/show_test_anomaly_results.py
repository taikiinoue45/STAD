import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path



def show_test_anomaly_results(base: Path):
    
    idxs = []
    for p in base.glob(f'* - test_anomaly_results.png'):
        idx, _ = p.stem.split(' - ')
        idxs.append(int(idx))
    
    idxs = sorted(idxs)
    idx = st.slider(label=' ', 
                    min_value=min(idxs),
                    max_value=max(idxs),
                    value=min(idxs),
                    step=idxs[1]-idxs[0])

    result = Image.open(base / f'{idx} - test_anomaly_results.png')

    plt.figure(figsize=(9, 3))
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    st.pyplot()
