import stad.datasets
import stad.typehint as T


class TrainerDataset:

    cfg: T.DictConfig
    augs_dict: T.Dict[str, T.Compose]

    def init_dataset(self, data_type: str) -> T.Dataset:

        dataset_attr = getattr(stad.datasets, self.cfg.dataset.name)
        dataset = dataset_attr(self.cfg, self.augs_dict, data_type)
        return dataset
