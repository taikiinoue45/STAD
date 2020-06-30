import sys
sys.path.append('/dgx/inoue/github/STAD')


from stad.datasets.MVTecDataset import MVTecDataset
from stad.models.school import School
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from torch2trt import torch2trt


class Builder:

    def __init__(self, cfg):
        self.cfg = cfg
        self.train_augs = self.get_train_augs()
        self.test_augs = self.get_test_augs()
        self.dataloader = self.get_dataloader()
        self.school = self.get_school()
        self.school = self.school.to(self.cfg.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        
        dummy_x = torch.ones((128, 128)).to(self.cfg.device)
        self.school.teacher = torch2trt(self.school.teacher, [dummy_x])

    def get_school(self):
        return School()

    def get_train_augs(self):
        augs = []
        for i in range(len(self.cfg.augs.train)):
            name = self.cfg['augs']['train'][i]['name']
            fn = getattr(albu, name)
            augs.append(fn(**self.cfg['augs']['train'][i]['args']))
        augs.append(ToTensorV2())
        [print(f'train aug: {aug}') for aug in augs]
        return albu.Compose(augs)

    def get_test_augs(self):
        augs = []
        for i in range(len(self.cfg.augs.test)):
            name = self.cfg['augs']['test'][i]['name']
            fn = getattr(albu, name)
            augs.append(fn(**self.cfg['augs']['test'][i]['args']))
        augs.append(ToTensorV2())
        [print(f'test aug: {aug}') for aug in augs]
        return albu.Compose(augs)

    def get_dataloader(self):
        dataset = MVTecDataset(img_dir=Path(self.cfg.dataset.path),
                               augs=self.train_augs)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=True)
        return dataloader

    def get_optimizer(self):
        parameters = self.school.student.parameters()
        lr = self.cfg.optim.lr
        weight_decay = self.cfg.optim.weight_decay
        optimizer = torch.optim.Adam(parameters,
                                     lr=lr,
                                     weight_decay=weight_decay)
        return optimizer

    def get_criterion(self):
        return torch.nn.MSELoss(reduction='mean')

    def run_train(self):
        
        self.school.teacher.eval()
        for epoch in tqdm(range(self.cfg.epochs)):
            for img in self.dataloader:
                img = img.to(self.cfg.device)
                surrogate_label, pred = self.school(img)
                loss = self.criterion(pred, surrogate_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def run_inference(self):
        img_path = self.cfg.inference.img.path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = self.cfg.inference.mask.path
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        anomaly_map = np.zeros((900, 900))

        self.school.teacher.eval()
        self.school.student.eval()
        for i in tqdm(range(64, 900-64)):
            for j in range(64, 900-64):
                patch = img[i-64:i+64, j-64:j+64]
                sample = self.test_augs(image=patch)
                patch = sample['image']
                patch = patch.unsqueeze(0)
                patch = patch.to(self.cfg.device)
                
                surrogate_label, pred = self.school(patch)
                loss = self.criterion(pred, surrogate_label)
                anomaly_map[i, j] = loss.item()

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

        plt.savefig(self.cfg.inference.savefig.path)
