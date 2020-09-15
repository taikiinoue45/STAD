import torch

import stad.typehint as T


class TrainerOptimizer:

    school: T.Module
    cfg: T.DictConfig

    def init_optimizer(self) -> T.Optimizer:

        params = self.school.student.parameters()
        optimizer_attr = getattr(torch.optim, self.cfg.optimizer.name)
        args = self.cfg.optimizer.args
        if args:
            return optimizer_attr(params, **args)
        else:
            return optimizer_attr(params)
