import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

cfg = {}
cfg['data_dir'] = Path('/Users/taiki/Downloads/MVTec/bottle')

st.title('Student-Teacher Anomaly Detection')


train_dir = cfg['data_dir'] / 'train/good'

min_value = 0
max_value = -1
for p in train_dir.glob('*.png'):
    max_value = max(max_value, int(p.stem))

idx = st.slider('Train Normal Images', min_value, max_value, min_value)
idx = str(idx)

img_path = train_dir / f'{idx.zfill(3)}.png'
img = Image.open(img_path)

plt.figure(figsize=(15, 5))
plt.subplot(132)
plt.imshow(img)
plt.axis('off')
st.pyplot()


test_dir = cfg['data_dir'] / 'test/good'

min_value = 0
max_value = -1
for p in test_dir.glob('*.png'):
    max_value = max(max_value, int(p.stem))

idx = st.slider('Test Normal Images', min_value, max_value, min_value)
idx = str(idx)

img_path = test_dir / f'{idx.zfill(3)}.png'
img = Image.open(img_path)

plt.figure(figsize=(15, 5))
plt.subplot(132)
plt.imshow(img)
plt.axis('off')
st.pyplot()


options = ['contamination', 'broken_small', 'broken_large']
data_type = st.selectbox('Anomaly Type', options)

test_dir = cfg['data_dir'] / f'test/{data_type}'
gt_dir = cfg['data_dir'] / f'ground_truth/{data_type}'

min_value = 0
max_value = -1
for p in test_dir.glob('*.png'):
    max_value = max(max_value, int(p.stem))

idx = st.slider('Test Anomaly Images and Masks',
                min_value, max_value, min_value)
idx = str(idx)

img_path = test_dir / f'{idx.zfill(3)}.png'
img = Image.open(img_path)

gt_path = gt_dir / f'{idx.zfill(3)}_mask.png'
gt = Image.open(gt_path)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(img)
plt.axis('off')

plt.subplot(132)
plt.imshow(gt)
plt.axis('off')

plt.subplot(133)
plt.imshow(img)
plt.imshow(gt, alpha=0.3)
plt.axis('off')

st.pyplot()
