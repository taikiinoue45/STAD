from torch.utils.data import DataLoader

import stad.typehint as T


class TranerDataLoader:

    dataset: T.Dataset
    cfg: T.DictConfig

    def init_dataloader(self, data_type: str) -> T.DataLoader:

        dataloader = DataLoader(
            dataset=self.dataset[data_type], **self.cfg.dataloader[data_type].args
        )
        return dataloader
