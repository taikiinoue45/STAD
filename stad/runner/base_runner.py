from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

from omegaconf.dictconfig import DictConfig
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from stad import transforms
from stad.transforms import Compose


class BaseRunner(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.preprocess_dict = {}
        self.dataset_dict = {}
        self.dataloader_dict = {}
        for data_type in ["train", "val"]:
            self.preprocess_dict[data_type] = self._init_preprocess(data_type)
            self.dataset_dict[data_type] = self._init_dataset(data_type)
            self.dataloader_dict[data_type] = self._init_dataloader(data_type)

        self.school = self._init_school().to(self.cfg.runner.device)
        self.optimizer = self._init_optimizer()
        self.criterion = self._init_criterion()

        self.best_auroc = 0.0

    def _init_preprocess(self, data_type: str) -> Compose:

        cfg = self.cfg.preprocess[data_type]
        return transforms.load(cfg.yaml, data_format="yaml")

    def _init_dataset(self, data_type: str) -> Dataset:

        cfg = self.cfg.dataset[data_type]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), preprocess=self.preprocess_dict[data_type])

    def _init_dataloader(self, data_type: str) -> DataLoader:

        cfg = self.cfg.dataloader[data_type]
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), dataset=self.dataset_dict[data_type])

    def _init_school(self) -> Module:

        cfg = self.cfg.school
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_criterion(self) -> Module:

        cfg = self.cfg.criterion
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}))

    def _init_optimizer(self) -> Optimizer:

        cfg = self.cfg.optimizer
        attr = self._get_attr(cfg.name)
        return attr(**cfg.get("args", {}), params=self.school.parameters())

    def _get_attr(self, name: str) -> Any:

        module_path, attr_name = name.split(" - ")
        module = import_module(module_path)
        return getattr(module, attr_name)

    @abstractmethod
    def run(self) -> None:

        raise NotImplementedError()

    @abstractmethod
    def _train(self) -> None:

        raise NotImplementedError()

    @abstractmethod
    def _validate(self) -> None:

        raise NotImplementedError()

    @abstractmethod
    def _test(self) -> None:

        raise NotImplementedError()
