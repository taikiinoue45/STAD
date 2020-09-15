import torch.nn

import stad.typehint as T


class TrainerCriterion:

    cfg: T.DictConfig

    def init_criterion(self, data_type: str) -> T.Loss:

        criterion_attr = getattr(torch.nn, self.cfg.criterion.name)
        args = self.cfg.criterion[data_type].args
        if args:
            return criterion_attr(**args)
        else:
            return criterion_attr()
