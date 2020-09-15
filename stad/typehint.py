from typing import Any, Dict, TypeVar

from nptyping import NDArray

Tensor = TypeVar("torch.tensor")
Loss = TypeVar("torch.nn.modules.loss._Loss")
Optimizer = TypeVar("torch.optim.Optimizer")
DataLoader = TypeVar("torch.utils.data.DataLoader")
Dataset = TypeVar("torch.utils.data.Dataset")
Module = TypeVar("torch.nn.Module")
DictConfig = TypeVar("omegaconf.DictConfig")
Compose = TypeVar("stad.albu.Compose")
Logger = TypeVar("logging.Logger")
Path = TypeVar("pathlib.Path")
