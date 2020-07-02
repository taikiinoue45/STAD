import sys
sys.path.append('/app/github/STAD/')

import stad.datasets
import stad.models
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch2trt import torch2trt



class Trainer:

    def __init__(self, cfg):

        self.cfg = cfg
        self.train_augs = self.get_train_augs()
        self.test_augs = self.get_test_augs()
        self.dataloader = self.get_dataloader()
        self.school = self.get_school()
        self.school = self.school.to(self.cfg.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

        dummy_x = torch.ones((1, 3, 128, 128)).to(self.cfg.device)
        self.school.teacher = torch2trt(self.school.teacher, [dummy_x])
        print('---- Finish converting to TensorRT ----')
        
        
    def get_school(self):

        return stad.models.School()


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
        
        print(f'Dataset: {self.cfg.dataset.name}')
        Dataset = getattr(stad.datasets, self.cfg.dataset.name)
        dataset = Dataset(img_dir=Path(self.cfg.dataset.path),
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
        for epoch in tqdm(range(self.cfg.train.epochs)):
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
        img = img[320:800, 500:2000]

        h, w, c = img.shape
        anomaly_map = np.zeros((h, w))

        self.school.teacher.eval()
        self.school.student.eval()
        for i in tqdm(range(64, h-64)):
            for j in range(64, w-64):
                patch = img[i-64:i+64, j-64:j+64]
                sample = self.test_augs(image=patch)
                patch = sample['image']
                patch = patch.unsqueeze(0)
                patch = patch.to(self.cfg.device)
                
                surrogate_label, pred = self.school(patch)
                loss = self.criterion(pred, surrogate_label)
                anomaly_map[i-64:i+64, j-64:j+64] = loss.item()

        plt.figure(figsize=(12, 8))

        plt.subplot(131)
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(anomaly_map)
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(img)
        plt.imshow(anomaly_map, alpha=0.5)
        plt.axis('off')

        plt.savefig(self.cfg.inference.savefig.path)
