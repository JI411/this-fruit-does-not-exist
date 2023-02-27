import typing as tp

import cv2
import numpy as np

import const


def find_objects(binary_map: np.ndarray, min_area: int = 600)-> np.ndarray:
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
    if mask.sum() <= 10:
        return np.zeros_like(mask)
    mask += 2
    mask, _, _ = cv2.grabCut(
        img=image, mask=mask, rect=None,
        bgdModel=np.zeros((1, 65), np.float64), fgdModel=np.zeros((1, 65), np.float64),
        iterCount=5, mode=cv2.GC_INIT_WITH_MASK,
    )
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    return mask

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
