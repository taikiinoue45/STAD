import stad.datasets
import stad.models
import torch
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

from stad import albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

log = logging.getLogger(__name__)



class Trainer:

    def __init__(self, cfg):

        self.cfg = cfg
        
        self.train_augs = self.get_augs(train_or_test='train')
        self.test_augs = self.get_augs(train_or_test='test')
        
        self.dataloader = {}
        
        self.dataloader['train'] = self.get_dataloader(
            img_dir=self.cfg.dataset.train.img,
            mask_dir=self.cfg.dataset.train.mask,
            augs=self.train_augs,
            is_anomaly=False
        )

        self.dataloader['normal'] = self.get_dataloader(
            img_dir=self.cfg.dataset.test.normal.img,
            mask_dir=self.cfg.dataset.test.normal.mask,
            augs=self.test_augs,
            is_anomaly=False
        )

        self.dataloader['anomaly'] = self.get_dataloader(
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


    def get_augs(self,
                 train_or_test: str):
        
        augs = []
        for i in range(len(self.cfg['augs'][train_or_test])):
            name = self.cfg['augs'][train_or_test][i]['name']
            fn = getattr(albu, name)
            augs.append(fn(**self.cfg['augs'][train_or_test][i]['args']))
        augs.append(ToTensorV2())
        
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
    
    def load_school_pth(self):
        
        self.school.load_state_dict(torch.load(self.cfg.pretrained_models.school))


    def run_train_student(self):
        
        self.school.teacher.eval()
        for epoch in range(1, self.cfg.train.epochs+1):
            
            loss_sum = 0
            
            for img, _, _ in self.dataloader['train']:
                img = img.to(self.cfg.device)
                surrogate_label, pred = self.school(img)
                loss = self.criterion(pred, surrogate_label)
                loss_sum += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
#                 if epoch % self.cfg.train.mini_epochs == 0:
#                     anomaly_map = self.compute_anomaly_map(img)
            
            epoch_loss = loss_sum / len(self.dataloader['train'])
            log.info(f'epoch: {epoch:04},  loss: {epoch_loss}')
        
        # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
        # https://hydra.cc/docs/tutorial/working_directory
        torch.save(self.school.state_dict(), 'school.pth')

    
    def compute_anomaly_map(self, img):
        
        unfold = torch.nn.Unfold(
            kernel_size=(self.cfg.patch_size, self.cfg.patch_size), 
            stride=self.cfg.test.unfold_stride)
        patches = unfold(img)
        patches = patches.permute(0, 2, 1)
        patches = patches.view(-1, 3, self.cfg.patch_size, self.cfg.patch_size)
        
        anomaly_map = np.zeros(patches.shape[0])
        quotient, remainder = divmod(patches.shape[0], self.cfg.test.batch_size)

        for i in tqdm(range(quotient)):
            
            start = i * self.cfg.test.batch_size
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
        
        # img.shape -> (b, c, h, w)
        _, _, img_h, img_w = img.shape
        anomaly_map_h = int((img_h - self.cfg.patch_size) / self.cfg.test.unfold_stride + 1)
        anomaly_map_w = int((img_w - self.cfg.patch_size) / self.cfg.test.unfold_stride + 1)
        anomaly_map = anomaly_map.reshape((anomaly_map_h, anomaly_map_w))
        anomaly_map = cv2.resize(anomaly_map, (img_h, img_w))
        
        del patches
        
        return anomaly_map

    
    
    def run_test(self):
        
        self.school.teacher.eval()
        self.school.student.eval()
        
        for anomaly_or_normal in ['anomaly', 'normal']:
            for i, (img, raw_img, mask) in enumerate(self.dataloader[anomaly_or_normal]):

                anomaly_map = self.compute_anomaly_map(img)

                # Save raw_img, mask and anomaly map
                # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
                # https://hydra.cc/docs/tutorial/working_directory

                raw_img = raw_img.squeeze().detach().numpy()
                mask = mask.squeeze().detach().numpy()
                
                cv2.imwrite(f'test/{anomaly_or_normal}/{i:02}_img.jpg', raw_img)
                cv2.imwrite(f'test/{anomaly_or_normal}/{i:02}_mask.png', mask)                
                
                with open(f'test/{anomaly_or_normal}/{i:02}_anomaly_map.npy', 'wb') as f:
                    np.save(f, anomaly_map)
