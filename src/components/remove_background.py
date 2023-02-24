import typing as tp

import albumentations as albu
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import const


def find_objects(binary_map: np.ndarray) -> np.ndarray:
    """Remove small connected components from a binary image."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros(labels.shape, np.uint8)
    for i, area in enumerate(areas):
        if area >= 600:
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


def bg_remove_grad_cam(image: np.ndarray, cam: BaseCAM) -> np.ndarray:
    """Remove background using Grad-CAM."""
    transform = albu.Compose([albu.Resize(224, 224), albu.Normalize(), ToTensorV2()])
    input_tensor = transform(image=image)['image'][None]
    targets = [ClassifierOutputTarget(954)]
    # noinspection PyTypeChecker
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
    grayscale_cam = cv2.cvtColor(grayscale_cam, cv2.COLOR_GRAY2BGR)
    return (grayscale_cam > 0.3) * image


    # model = timm.create_model('resnet34', pretrained=True)
    # target_layers = [model.layer4[-1]]
    # cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=False)
    # components.show_image(bg_remove_gradcam(img, cam=cam))
    # components.show_image(bg_remove_hsv(bg_remove_gradcam(img, cam=cam)))
