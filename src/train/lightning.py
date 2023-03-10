# pylint: disable=unused-argument, arguments-differ, too-many-ancestors

import typing as tp

import numpy as np
import pytorch_lightning as pl
import torch
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import const
from src.train.dataset import SyntheticFruitDataset, RealFruitDataset
from src.train.loss import AggregatedLoss
from src.train.model import BaseModel


def tensor_to_numpy_image(tensor: torch.Tensor) -> tp.Any:
    """Convert tensor to numpy."""
    tensor = tensor.detach().cpu().squeeze(0).squeeze(0)
    return np.asarray(tensor)


class BaseFruitSegmentationModule(pl.LightningModule):
    """Lightning wrapper for models, connect loss, dataloader and model."""

    mask_logging_thresholds = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

    def __init__(self, model: BaseModel, batch_size: int) -> None:
        """Create model for training."""
        super().__init__()

        self.model = model
        self.loss = AggregatedLoss((DiceLoss(mode='binary'), FocalLoss(mode='binary')))
        self.batch_size = batch_size

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Train model on batch."""
        predict = self.model(batch=batch['image'])
        score = self.loss.forward(y_pred=predict, y_true=batch['mask'])
        self.log("train_loss", score, on_step=True, logger=True)

        if self.logger is not None and self.current_epoch % 10 == 0 and batch_idx == 0:
            sample, original_sample = batch['image'][0][None], batch['original_image'][0][None]
            self._log_images(sample, original_sample, key=f'synthetic_image_{batch_idx}')
        return score

    @torch.no_grad()
    def _log_images(self, sample: torch.Tensor, original_sample: torch.Tensor, key: str, normalize: bool = False):
        predict = self.model(sample)
        predict = tensor_to_numpy_image(torch.sigmoid(predict))
        if normalize:
            predict -= predict.min()
            predict /= predict.max() + 1e-5
        original_image = tensor_to_numpy_image(original_sample)
        self.logger.log_image(
            key=f'{key}_normalized' if normalize else key,
            images=[original_image for _ in self.mask_logging_thresholds],
            masks=[
                {"prediction": {"mask_data": (predict > th), "class_labels": {0: "fruit"}}}
                for th in self.mask_logging_thresholds
            ]
        )

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        return DataLoader(SyntheticFruitDataset(), batch_size=self.batch_size, shuffle=True, num_workers=0)

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

class FruitSegmentationModule(BaseFruitSegmentationModule):  # pylint: disable=too-many-ancestors
    """Lightning wrapper for models, connect loss, dataloader and model."""

    def __init__(self, model: BaseModel, batch_size: int, real_images: tp.Sequence[const.PathType]) -> None:
        """Create model for training."""
        super().__init__(model, batch_size)
        self.real_images = real_images

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Validate model on batch."""
        if self.logger is not None and self.current_epoch % 10 == 0:
            self._log_images(batch['image'], batch['original_image'], key=f'real_image_{batch_idx}')

    def val_dataloader(self) -> DataLoader:
        """Get inference dataloader."""
        return DataLoader(RealFruitDataset(self.real_images), batch_size=1, shuffle=False, num_workers=0)
