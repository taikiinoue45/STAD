import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path


CWD = Path('/Users/taiki/Downloads/MVTec/bottle')


def train_normal_images():
    img_dir = CWD / 'train/good'

    min_value = 0
    max_value = -1
    for p in img_dir.glob('*.png'):
        max_value = max(max_value, int(p.stem))

    idx = st.slider(' ', min_value, max_value, min_value)
    idx = str(idx)

    img_path = img_dir / f'{idx.zfill(3)}.png'
    img = Image.open(img_path)

    plt.figure(figsize=(15, 5))
    plt.subplot(132)
    plt.imshow(img)
    plt.axis('off')
    st.pyplot()


def test_nomal_images():
    img_dir = CWD / 'test/good'

    min_value = 0
    max_value = -1
    for p in img_dir.glob('*.png'):
        max_value = max(max_value, int(p.stem))

    idx = st.slider(' ', min_value, max_value, min_value)
    idx = str(idx)

    img_path = img_dir / f'{idx.zfill(3)}.png'
    img = Image.open(img_path)

    plt.figure(figsize=(15, 5))
    plt.subplot(132)
    plt.imshow(img)
    plt.axis('off')
    st.pyplot()


def test_anomaly_images():
    img_dir = CWD / 'test/broken_small'
    mask_dir = CWD / 'ground_truth/broken_small'
    npy_dir = CWD / 'anomaly_map/broken_small'

    min_value = 0
    max_value = -1
    for p in npy_dir.glob('*.npy'):
        max_value = max(max_value, int(p.stem))

    idx = st.slider(' ', min_value, max_value, min_value)
    idx = str(idx)

    npy_path = npy_dir / f'{idx.zfill(3)}.npy'
    anomaly_map = np.load(npy_path)

    img_path = img_dir / f'{idx.zfill(3)}.png'
    img = Image.open(img_path)

    mask_path = mask_dir / f'{idx.zfill(3)}_mask.png'
    mask = Image.open(mask_path)

    plt.figure(figsize=(15, 15))
    plt.subplot(331)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(332)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(333)
    plt.imshow(img)
    plt.imshow(mask, cmap='gray', alpha=0.3)
    plt.axis('off')

    plt.subplot(334)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(335)
    plt.imshow(anomaly_map)
    plt.axis('off')

    plt.subplot(336)
    plt.imshow(img)
    plt.imshow(anomaly_map, alpha=0.3)
    plt.axis('off')

    plt.subplot(337)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(338)
    plt.imshow(anomaly_map)
    plt.axis('off')

    plt.subplot(339)
    plt.imshow(anomaly_map)
    plt.imshow(mask, cmap='gray', alpha=0.3)
    plt.axis('off')

    st.pyplot()


st.title('Student-Teacher Anomaly Detection')

st.header(' ')
st.header(' ')
st.header('Training Dataset (All Images are Normal)')
train_normal_images()

st.header(' ')
st.header(' ')
st.header('Test - Anomaly Map of Normal Images')
test_nomal_images()

st.header(' ')
st.header(' ')
st.header('Test - Anomaly Map of Broken Small Images')
test_anomaly_images()
