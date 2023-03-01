import segmentation_models_pytorch as smp
import torch
from torch import nn

class BaseModel(nn.Module):
    """Base model class."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run model on tensor with shape (batch, 3, height, width).
        Return tensor with shape (batch, 1, height, width).
        """
        raise NotImplementedError


class UnetWrapper(BaseModel):
    """Wrapper for smp.Unet model."""
    def __init__(self, encoder_name: str = 'resnet34'):
        """Create Unet model."""
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.unet(x=batch)
