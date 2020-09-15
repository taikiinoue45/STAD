from torch.utils.data import DataLoader

import stad.typehint as T


class TrainerDataLoader:

    dataset_dict: T.Dict[str, T.Dataset]
    cfg: T.DictConfig

    def init_dataloader(self, data_type: str) -> T.DataLoader:

        dataloader = DataLoader(
            dataset=self.dataset_dict[data_type], **self.cfg.dataloader[data_type].args
        )
        return dataloader
