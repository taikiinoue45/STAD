import torch
import torch.nn as nn
import torchvision


class School(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_vgg = torchvision.models.vgg19(pretrained=True)
        vgg = torchvision.models.vgg19(pretrained=False)
        self.teacher = pretrained_vgg.features[:36]
        self.student = vgg.features[:36]
        self.teacher

    def forward(self, x):
        with torch.no_grad():
            surrogate_label = self.teacher(x)
        pred = self.student(x)
        return surrogate_label, pred
