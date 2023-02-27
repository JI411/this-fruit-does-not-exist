import typing as tp

import torch
from torch.nn.modules.loss import _Loss


class MultipleCriterionWrapper(_Loss):
    def __init__(self, losses: tp.Sequence[_Loss]):
        super().__init__()
        self.losses = losses

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        scores = [loss(y_pred=y_pred, y_true=y_true) for loss in self.losses]
        return sum(scores)
