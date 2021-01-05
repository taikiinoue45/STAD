import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from typing_extensions import Literal


class MSELoss(Module):
    def __init__(self, reduction: Literal["mean", "sum"]) -> None:

        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, is_train: bool) -> Tensor:

        if is_train:
            return F.mse_loss(input, target, reduction=self.reduction)
        else:
            mb_losses = F.mse_loss(input, target, reduction="none")
            b, c, h, w = mb_losses.shape
            mb_losses = mb_losses.view(b, -1)
            return torch.mean(mb_losses, dim=1)
