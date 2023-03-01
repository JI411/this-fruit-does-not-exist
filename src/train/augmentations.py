import albumentations as albu
import cv2
from albumentations.pytorch import ToTensorV2

def get_default_augmentation_synthetic_samples():
    """
    Default train augmentation only for samples - don't apply it to background.
    """
    return albu.Compose([
        albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
        albu.ColorJitter(p=0.2),

        albu.Resize(384, 384, p=1),
        albu.RandomScale(scale_limit=(-0.5, 0.25), p=1),
        albu.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),

        albu.RandomRotate90(p=0.1),
        albu.VerticalFlip(p=0.05),
        albu.HorizontalFlip(p=0.15),
        albu.GridDistortion(
            num_steps=2, distort_limit=0.1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.05
        ),
    ])

def get_default_augmentation_full_image():
    """
    Default train augmentation.
    """
    return albu.Compose([
        albu.GaussianBlur(p=0.05),
        albu.GaussNoise(p=0.05),
        albu.GridDropout(random_offset=True, fill_value=0, mask_fill_value=0, p=0.2),
        albu.HorizontalFlip(p=0.1),
    ])


def transform__resize():
    """Default transform for all data."""
    return albu.Compose([
        albu.Resize(512, 512, p=1),
    ])

def transform__to_tensor():
    """Default transform for all data."""
    return albu.Compose([
        albu.Normalize(p=1),
        ToTensorV2(p=1),
    ])
