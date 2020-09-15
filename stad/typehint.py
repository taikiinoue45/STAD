from typing import Any, TypeVar

from nptyping import NDArray

Tensor = TypeVar("torch.tensor")
Loss = TypeVar("torch.nn.modules.loss._Loss")
Optimizer = TypeVar("torch.optim.Optimizer")
DataLoader = TypeVar("torch.utils.data.DataLoader")
Module = TypeVar("torch.nn.Module")
DictConfig = TypeVar("omegaconf.DictConfig")
Compose = TypeVar("stad.albu.Compose")
Logger = TypeVar("logging.Logger")
