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



class Trainer:

    def __init__(self, cfg):

        self.cfg = cfg
        self.train_augs = self.get_train_augs()
        self.test_augs = self.get_test_augs()

        self.train_dataloader = self.get_dataloader(
            img_dir=self.cfg.dataset.train.img,
            mask_dir='',
            augs=self.train_augs,
            is_anomaly=False
        )

        self.test_normal_dataloader = self.get_dataloader(
            img_dir=self.cfg.dataset.test.normal.img,
            mask_dir='',
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
        
        Dataset = getattr(stad.datasets, self.cfg.dataset.name)
        
        dataset = Dataset(img_dir=Path(img_dir), 
                          mask_dir=Path(mask_dir), 
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
            for img, arr, mask in self.train_dataloader:
                img = img.to(self.cfg.device)
                surrogate_label, pred = self.school(img)
                loss = self.criterion(pred, surrogate_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def compute_anomaly_map(self,
                            dataloader,
                            output_dir):
        
        self.school.teacher.eval()
        self.school.student.eval()
        
        for i, (img, raw_img, mask) in enumerate(dataloader):
                        
            unfold = torch.nn.Unfold(
                kernel_size=(self.cfg.patch_size, self.cfg.patch_size), 
                stride=1)
            patches = unfold(img)
            patches = patches.permute(0, 2, 1)
            patches = patches.view(-1, 3, self.cfg.patch_size, self.cfg.patch_size)
            
            anomaly_map = np.zeros(patches.shape[0])
            quotient, remainder = divmod(patches.shape[0], self.cfg.test.batch_size)
            
            for j in tqdm(range(quotient)):
        
                start = j * self.cfg.test.batch_size
                end = start + self.cfg.test.batch_size

                patch = patches[start:end, :, :, :]
                patch = patch.to(self.cfg.device)

                surrogate_label, pred = self.school(patch)

                losses = pred - surrogate_label
                losses = losses.view(self.cfg.test.batch_size, -1)
                losses = losses.pow(2).mean(1)
                losses = losses.cpu().detach().numpy()
                anomaly_map[start:end] = losses
                
            patch = patches[-remainder:, :, :, :]
            patch = patch.to(self.cfg.device)
            
            surrogate_label, pred = self.school(patch)
            
            losses = pred - surrogate_label
            losses = losses.view(remainder, -1)
            losses = losses.pow(2).mean(1)
            losses = losses.cpu().detach().numpy()
            anomaly_map[-remainder:] = losses
            
            b, c, h, w = img.shape
            anomaly_map = anomaly_map.reshape(h-self.cfg.patch_size+1, w-self.cfg.patch_size+1)
            anomaly_map = np.pad(anomaly_map, 
                                 ((self.cfg.patch_size//2, self.cfg.patch_size//2-1), 
                                  (self.cfg.patch_size//2, self.cfg.patch_size//2-1)), 
                                 'constant')
            
            cv2.imwrite(str(output_dir / f'{i:02}_img.jpg'), raw_img[0].detach().numpy())
            cv2.imwrite(str(output_dir / f'{i:02}_anomaly_map.png'), anomaly_map)
            cv2.imwrite(str(output_dir / f'{i:02}_mask.png'), mask[0].detach().numpy())

            
    def run_test(self):
        
        self.compute_anomaly_map(self.test_anomaly_dataloader, Path(self.cfg.test.output_dir.anomaly))
        self.compute_anomaly_map(self.test_normal_dataloader, Path(self.cfg.test.output_dir.normal))
        
        