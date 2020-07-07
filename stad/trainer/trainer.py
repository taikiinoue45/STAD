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

        self.train_dataloader = self.get_dataloader(
            img_dir=self.cfg.dataset.train.img,
            mask_dir=self.cfg.dataset.train.mask,
            augs=self.train_augs,
            is_anomaly=False
        )

        self.test_normal_dataloader = self.get_dataloader(
            img_dir=self.cfg.dataset.test.normal.img,
            mask_dir=self.cfg.dataset.test.normal.mask,
            augs=self.test_augs,
            is_anomaly=False
        )

        self.test_anomaly_dataloader = self.get_dataloader(
            img_dir=self.cfg.dataset.test.anomaly.img,
            mask_dir=self.cfg.dataset.test.anomaly.mask,
            augs=self.test_augs,
            is_anomaly=True
        )

        self.school = self.get_school()
        self.school = self.school.to(self.cfg.device)
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()

        
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


    def get_dataloader(self,
                       img_dir: str,
                       mask_dir: str,
                       augs: albu.Compose,
                       is_anomaly: bool):
        
        print(f'Dataset: {self.cfg.dataset.name}')
        Dataset = getattr(stad.datasets, self.cfg.dataset.name)
        
        dataset = Dataset(img_dir=img_dir, 
                          mask_dir=mask_dir, 
                          augs=augs,
                          is_anomaly=is_anomaly)
                          
        dataloader = DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=False)
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
            for img in self.train_dataloader:
                img = img.to(self.cfg.device)
                surrogate_label, pred = self.school(img)
                loss = self.criterion(pred, surrogate_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def run_inference(self):
        
        self.school.teacher.eval()
        self.school.student.eval()

        patch_size = self.cfg.patch_size

        # Compute anomaly map of anomaly images
        for i, (img, mask) in tqdm(enumerate(self.test_anomaly_dataloader)):
            
            img.to(self.cfg.device)
            h, w, c = img.shape
            anomaly_map = np.zeros((h, w))

            for j in tqdm(range(0, h-patch_size)):
                for k in range(0, w-patch_size):

                    patch = img[j:j+patch_size, k:k+patch_size]
                    surrogate_label, pred = self.school(patch)
                    loss = self.criterion(pred, surrogate_label)
                    anomaly_map[j:j+patch_size, k:k+patch_size] = loss.item()
            
            self.savefig_anomaly_map(img, 
                                     mask, 
                                     anomaly_map,
                                     self.cfg.inference.savefig.normal)


        # Compute anomaly map of normal images
        for i, (img, mask) in tqdm(enumerate(self.test_normal_dataloader)):
            
            img.to(self.cfg.device)
            h, w, c = img.shape
            anomaly_map = np.zeros((h, w))

            for j in tqdm(range(0, h-patch_size)):
                for k in range(0, w-patch_size):

                    patch = img[j:j+patch_size, k:k+patch_size]
                    surrogate_label, pred = self.school(patch)
                    loss = self.criterion(pred, surrogate_label)
                    anomaly_map[j:j+patch_size, k:k+patch_size] = loss.item()

            self.savefig_anomaly_map(img,
                                     mask,
                                     anomaly_map,
                                     self.cfg.inference.savefig.anomaly)


    def savefig_anomaly_map(self,
                            img,
                            mask,
                            anomaly_map,
                            savefig_path):

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

            plt.savefig(savefig_path)
