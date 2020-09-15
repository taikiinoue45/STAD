import torch

import stad.models
import stad.typehint as T


class TrainerSchool:

    cfg: T.DictConfig
    model: T.Module

    def init_school(self) -> T.Module:

        return stad.models.School()

    def load_pretrained_model(self):

        self.school.load_state_dict(torch.load(self.cfg.model.school.pretrained))
