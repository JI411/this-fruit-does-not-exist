import typing as tp

import torch
from torch.nn.modules.loss import _Loss


class AggregatedLoss(_Loss):
    """Wrapper for multiple criterions aggregation."""
    def __init__(self, criterions: tp.Sequence[_Loss]):
        """Create wrapper for loss aggregation."""
        super().__init__()
        self.criterion = criterions

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate loss."""
        scores = [loss(y_pred=y_pred, y_true=y_true) for loss in self.criterion]
        return sum(scores)
