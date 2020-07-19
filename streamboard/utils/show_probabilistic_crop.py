import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path



def show_probabilistic_crop(base: Path):

    epochs = []
    for p in base.glob('* - probabilistic_crop.png'):
        epoch, _ = p.stem.split(' - ') 
        epochs.append(int(epoch))
   
    epochs = sorted(set(epochs))
    epoch = st.slider(label=' ', 
                      min_value=min(epochs),
                      max_value=max(epochs),
                      value=min(epochs),
                      step=epochs[1]-epochs[0])

    result = Image.open(base / f'{epoch} - probabilistic_crop.png')

    plt.figure(figsize=(9, 3))
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    st.pyplot()
