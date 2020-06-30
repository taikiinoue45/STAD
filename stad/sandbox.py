import sys
sys.path.append('/dgx/inoue/github/STAD')

import torch
import hydra
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from albumentations.pytorch import ToTensorV2



class MVTecDataset(Dataset):
    
    def __init__(self,
                 img_dir: Path,
                 augmentations: albu.Compose = None,
                 preprocessing: albu.Compose = None) -> None:
        
        self.img_paths = [p for p in img_dir.glob('*.png')]
        self.augmentations = augmentations
        self.preprocessing = preprocessing

        
    def __getitem__(self, 
                    idx: int):
            
            # The original image from the dataset has RGB channels, but flipped in the form of BGR. 
            # So, we use CV2 to revert back to the standard RGB.
            img_path = str(self.img_paths[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.augmentations:
                sample = self.augmentations(image=img)
                img = sample['image']

            if self.preprocessing:
                sample = self.preprocessing(image=img)
                img= sample['image']

            return img
        
        
    def __len__(self) -> int:
        
        return len(self.img_paths)
    
    

@hydra.main(config_path='/dgx/inoue/github/STAD/stad/yamls/base.yaml')
def my_app(cfg):

    pretrained_vgg = models.vgg19(pretrained=True)
    teacher = pretrained_vgg.features[:36]
    teacher = teacher.to(cfg.device)
    
    vgg = models.vgg19(pretrained=False)
    student = vgg.features[:36]
    student = student.to(cfg.device)
    
    train_augs = [albu.HorizontalFlip(p=0.5),
                  albu.RandomCrop(height=128, width=128, always_apply=True, p=1)]
    train_augs = albu.Compose(train_augs)    
    
    preprocessing = [albu.Normalize(always_apply=True, p=1),
                     ToTensorV2()]
    preprocessing = albu.Compose(preprocessing)
    
    mvtec = MVTecDataset(img_dir=Path(cfg.dataset.path),
                         augmentations=train_augs,
                         preprocessing=preprocessing)
    
    train_loader = DataLoader(dataset=mvtec,
                              batch_size=cfg.train.batch_size,
                              shuffle=True)
    
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0002, weight_decay=0.00001)
    
    teacher.eval()

    for epoch in range(cfg.epochs):
        for i, img in enumerate(train_loader):

            img = img.to(cfg.device)
            with torch.no_grad():
                surrogate_label = teacher(img)
            optimizer.zero_grad()
            pred = student(img)
            loss = criterion(pred, surrogate_label)
            loss.backward()
            optimizer.step()

        print(f'epoch: {epoch}')
        
        
    preprocessing = [albu.Normalize(always_apply=True, p=1),
                     ToTensorV2()]
    preprocessing = albu.Compose(preprocessing)
    
    img_path = '/dgx/shared/momo/Data/MVTec/bottle/test/broken_small/000.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    anomaly_map = np.zeros((900, 900))

    teacher.eval()
    student.eval()
    for i in range(64, 900-64):
        for j in range(64, 900-64):
            patch = img[i-64:i+64, j-64:j+64]
            sample = preprocessing(image=patch)
            patch = sample['image']
            patch = patch.unsqueeze(0)
            patch = patch.to(cfg['device'])

            surrogate_label = teacher(patch)
            pred = student(patch)
            loss = criterion(pred, surrogate_label)
            anomaly_map[i, j] = loss.item()
        print(i)
        
    img = cv2.imread('/dgx/shared/momo/Data/MVTec/bottle/test/broken_small/000.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread('/dgx/shared/momo/Data/MVTec/bottle/ground_truth/broken_small/000_mask.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(mask)
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(anomaly_map)
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(img)
    plt.imshow(anomaly_map, alpha=0.5)
    plt.axis('off')

    plt.savefig(cfg.inference.savefig.path)

        
if __name__ == "__main__":
    my_app()
