import torch

from torch.utils.data import DataLoader

from stad.models.school import School
from stad.datasets.MVTecDataset import MVTecDataset


class Trainer:

    def __init__(self,
                 school,
                 dataloader,
                 optimizer,
                 criterion):
        
        self.school = school
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion

    def run(self):

        for epoch in range(100):
            for img in self.dataloader:
                surrogate_label, pred = self.school(img)
                loss = self.criterion(pred, surrogate_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

