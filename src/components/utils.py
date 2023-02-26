import json
import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt

import const


def read_image(path: const.PathType, bgr2rgb: bool = True) -> np.ndarray:
    """Read cv2 image, check it exists. Convert to RGB if needed."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f'Image {path} not found')
    if bgr2rgb and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def show_image(image: np.ndarray) -> None:
    """Show image with matplotlib."""
    _, ax = plt.subplots(figsize=[10, 10])  # pylint: disable=invalid-name
    ax.imshow(image, cmap='gray', interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def save_json(path: const.PathType, data: tp.Any) -> None:
    """Save json to path."""
    with open(str(path), 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def read_json(path: const.PathType) -> tp.Any:
    """Read json from path."""
    with open(str(path), 'r', encoding='utf-8') as file:
        return json.load(file)
