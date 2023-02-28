import typing as tp

import albumentations as albu
import cv2
import numpy as np
import timm
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import const


def find_objects(binary_map: np.ndarray, min_area: int = 600) -> np.ndarray:
    """Remove small connected components from a binary image."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros(labels.shape, np.uint8)
    for i, area in enumerate(areas):
        if area >= min_area:
            result[labels == i + 1] = 255
    return result
def bg_remove_hsv(
        image: np.ndarray, hsv_lower: tp.Iterable[const.ColorRegionType], hsv_upper: tp.Iterable[const.ColorRegionType]
) -> np.ndarray:
    """Remove background using HSV color space."""
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = np.zeros_like(image_hsv[..., 0])
    for lower, upper in zip(hsv_lower, hsv_upper):
        lower, upper = np.array(lower), np.array(upper)
        mask += cv2.inRange(image_hsv, lower, upper)
    mask = find_objects(mask)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=7)
    return mask

def bg_remove_grabcut(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove background using GrabCut algorithm."""
    mask += 2  # 3 - probably foreground, 2 - probably background
    try:
        mask, _, _ = cv2.grabCut(
            img=image, mask=mask, rect=None,
            bgdModel=np.zeros((1, 65), np.float64), fgdModel=np.zeros((1, 65), np.float64),
            iterCount=5, mode=cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        # Catch error when initial mask is empty. Strange opencv behaviour sometimes :(
        return np.zeros_like(mask)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    return mask


def bg_remove_grad_cam(
        image: np.ndarray, cam: tp.Optional[BaseCAM] = None, category: int = 954, threshold: float = 0.3,
) -> np.ndarray:
    """Remove background using Grad-CAM."""
    if cam is None:
        model = timm.create_model('resnet34', pretrained=True)
        cam = HiResCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=False)

    transform = albu.Compose([albu.Resize(224, 224), albu.Normalize(), ToTensorV2()])

    input_tensor = transform(image=image)['image'][None]
    targets = [ClassifierOutputTarget(category)]
    # noinspection PyTypeChecker
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
    grayscale_cam = cv2.cvtColor(grayscale_cam, cv2.COLOR_GRAY2BGR)
    return grayscale_cam > threshold

def bg_remove_grabcut_and_hsv(
        image: np.ndarray,
        hsv_lower: tp.Iterable[const.ColorRegionType],
        hsv_upper: tp.Iterable[const.ColorRegionType],
        mask_threshold: float = 0.95,
) -> np.ndarray:
    """Remove background using GrabCut algorithm. Initial mask is generated using HSV color space."""
    mask = bg_remove_hsv(image, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=7)
    mask = (mask > mask_threshold).astype(np.uint8)
    return bg_remove_grabcut(image, mask)

def get_contours(mask: np.ndarray) -> tp.Tuple[tp.List[np.ndarray], np.ndarray]:
    """Find contours with hierarchy."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def split_mask(mask: np.ndarray) -> tp.List[np.ndarray]:
    """Split mask into separate masks."""
    mask = find_objects(mask)
    contours = get_contours(mask)
    masks = []
    for cnt in contours:
        zero_mask = np.zeros_like(mask)
        cv2.drawContours(zero_mask, [cnt], -1, 255, -1)
        masks.append(zero_mask)
    return masks

