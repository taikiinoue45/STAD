from typing import Tuple

import torch
import torchvision
from torch import Tensor
from torch.nn import Module


class VGG19(Module):
    def __init__(self) -> None:

        super().__init__()
        pretrained_vgg = torchvision.models.vgg19(pretrained=True)
        vgg = torchvision.models.vgg19(pretrained=False)
        self.teacher = pretrained_vgg.features[:36]
        self.student = vgg.features[:36]

    def initialize_student(self) -> None:

        vgg = torchvision.models.vgg19(pretrained=False)
        self.student = vgg.features[:36]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        with torch.no_grad():
            teacher_pred = self.teacher(x)
        student_pred = self.student(x)
        return (student_pred, teacher_pred)
