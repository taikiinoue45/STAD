import torch

import stad.models
import stad.typehint as T


class TrainerModel:

    cfg: T.DictConfig
    model: T.Module

    def init_model(self, model_type: str):

        return stad.models.School()

    def load_pretrained_model(self):

        self.school.load_state_dict(torch.load(self.cfg.model.school.pretrained))
