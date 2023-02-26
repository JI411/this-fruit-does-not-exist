# pylint: disable=super-init-not-called

import typing as tp

import albumentations as albu
import numpy as np
import torch
from torch.utils.data import Dataset

import const
from src.components import utils
from src.train import augmentations
from src.train.augmentations import transform__to_tensor, transform__resize


class Batch(tp.TypedDict):
    """Contains batch from dataset."""
    image: tp.Union[np.ndarray, torch.Tensor]
    mask: tp.Union[np.ndarray, torch.Tensor]
    original_image: tp.Union[np.ndarray, torch.Tensor]


def paste_image(background: np.ndarray, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paste image on background."""
    background[mask] = image[mask]
    return background

def augmentation_with_mask(image, mask, augmentation) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Augment image and mask."""
    if augmentation is None:
        return image, mask
    augmented = augmentation(image=image, mask=mask.astype(np.uint8))
    image: tp.Union[torch.Tensor, np.ndarray] = augmented['image']
    mask: tp.Union[torch.Tensor, np.ndarray] = augmented['mask'] > 0.5
    return image, mask

class BaseFruitDataset(Dataset):
    """Base class for fruit dataset."""
    resize = transform__resize()
    transform = transform__to_tensor()

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        """Get batch from dataset."""
        raise NotImplementedError

class SyntheticFruitDataset(BaseFruitDataset):
    """Dataset for synthetic fruit images."""

    num_samples_on_bg: tp.Tuple[int, ...] = (1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 5)

    def __init__(
            self,
            dataset_config: const.PathType = const.DATA_DIR / 'generated_dataset.json',
            background_dir: const.PathType = const.BACKGROUND_DIR,
            augmentation_full_image: albu.Compose = augmentations.get_default_augmentation_full_image(),
            augmentation_synthetic_samples: albu.Compose = augmentations.get_default_augmentation_synthetic_samples(),
    ) -> None:
        """Create dataset class."""
        self.dataset_config: tp.List[const.SampleType] = sorted(
            utils.read_json(dataset_config), key=lambda x: x['image_path']
        )
        self.backgrounds = sorted(list(background_dir.rglob('*.jpeg')))
        self.augmentation_full_image = augmentation_full_image
        self.augmentation_synthetic_samples = augmentation_synthetic_samples

    def __getitem__(self, idx: int) -> Batch:
        """Get batch from dataset."""
        background = utils.read_image(self.backgrounds[idx])
        num_samples = np.random.choice(self.num_samples_on_bg)
        samples = np.random.choice(self.dataset_config, size=num_samples)

        full_mask = np.zeros_like(background, dtype=bool)
        for sample in samples:
            image = utils.read_image(sample['image_path'], bgr2rgb=True)
            mask = utils.read_image(sample['mask_path'])
            image, mask = augmentation_with_mask(image, mask, self.augmentation_synthetic_samples)
            full_mask += mask
            background = paste_image(background, image, mask)
        background, full_mask = augmentation_with_mask(background, full_mask, self.augmentation_full_image)
        background, full_mask = augmentation_with_mask(background, full_mask, self.resize)
        background_tensor, full_mask_tensor = augmentation_with_mask(background.copy(), full_mask, self.transform)
        return {'image': background_tensor, 'mask': full_mask_tensor[..., 0], 'original_image': background}

    def __len__(self):
        return len(self.backgrounds)


class RealFruitDataset(BaseFruitDataset):
    """Dataset for real fruit images."""
    def __init__(self, real_images: tp.Sequence[const.PathType]) -> None:
        """Create inference dataset class."""
        self.real_images = real_images
        self.transform = transform__to_tensor()
        self.resize = transform__resize()

    def __getitem__(self, idx: int) -> tp.Dict[str, np.ndarray]:
        original_image = utils.read_image(self.real_images[idx])
        original_image = self.resize(image=original_image.copy())['image']
        image = self.transform(image=original_image.copy())['image']
        return {'image': image, 'original_image': original_image}

    def __len__(self):
        return len(self.real_images)
