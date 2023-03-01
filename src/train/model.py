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


class UnetMetricLearningClassifier(UnetWrapper):
    """Wrapper for smp.Unet model."""
    def __init__(self, encoder_name: str = 'resnet34'):
        """Create Unet model."""
        super().__init__(encoder_name=encoder_name)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
