import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path



def show_val_results(base: Path):

    epochs = []
    idxs = []
    for p in base.glob('* - val_results.png'):
        epoch, idx, _ = p.stem.split(' - ') 
        epochs.append(int(epoch))
        idxs.append(int(idx))
   
    epochs = sorted(set(epochs))
    epoch = st.slider(label=' ', 
                      min_value=min(epochs),
                      max_value=max(epochs),
                      value=min(epochs),
                      step=epochs[1]-epochs[0])

    idxs = sorted(set(idxs))
    idx = st.slider(label=' ', 
                    min_value=min(idxs),
                    max_value=max(idxs),
                    value=min(idxs),
                    step=idxs[1]-idxs[0])

    result = Image.open(base / f'{epoch} - {idx} - val_results.png')

    plt.figure(figsize=(9, 3))
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    st.pyplot()
