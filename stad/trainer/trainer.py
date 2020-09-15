import logging

import stad.typehint as T
from stad.trainer.augs import TrainerAugs
from stad.trainer.criterion import TrainerCriterion
from stad.trainer.dataloader import TrainerDataLoader
from stad.trainer.dataset import TrainerDataset
from stad.trainer.optimizer import TrainerOptimizer
from stad.trainer.run_train_val_test import TrainerRunTrainValTest
from stad.trainer.school import TrainerSchool


class Trainer(
    TrainerAugs,
    TrainerCriterion,
    TrainerDataLoader,
    TrainerDataset,
    TrainerOptimizer,
    TrainerRunTrainValTest,
    TrainerSchool,
):
    def __init__(self, cfg: T.DictConfig):
        super().__init__()

        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        self.augs_dict = {}
        self.dataset_dict = {}
        self.dataloader_dict = {}
        for data_type in ["train", "val", "test"]:
            self.augs_dict[data_type] = self.init_augs(data_type)
            self.dataset_dict[data_type] = self.init_dataset(data_type)
            self.dataloader_dict[data_type] = self.init_dataloader(data_type)

        self.school = self.init_school()
        self.optimizer = self.init_optimizer()
        self.criterion = self.init_criterion()
