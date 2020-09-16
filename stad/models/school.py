import torch
import torch.nn as nn

import stad.typehint as T
from stad.models import EfficientNet


class School(nn.Module):
    def __init__(self, cfg: T.DictConfig):
        super().__init__()

        self.student = EfficientNet.from_name(cfg.school.model_name)
        self.teacher = EfficientNet.from_pretrained(cfg.school.model_name)

    def forward(self, x):
        with torch.no_grad():
            teacher_feature_list = self.teacher.get_feature_list(x)
        student_feature_list = self.student.get_feature_list(x)
        return teacher_feature_list, student_feature_list
