import typing as tp
from dataclasses import dataclass

import const


@dataclass
class FruitConfig:
    """Fruits class config for hsv segmentation."""
    name: str
    prompt: str = None
    num_images: int = 60
    hsv_lower: tp.Iterable[tp.Tuple[int, int, int]] = ((0, 40, 0),)
    hsv_upper: tp.Iterable[tp.Tuple[int, int, int]] = ((255, 255, 255),)
    size_limit: tp.Tuple[int, int] = (30 * 30, 350 * 350)
    mask_threshold: float = 0.5

    def __post_init__(self):
        """Post init hook. Convert lists to tuples and set default prompt."""
        self.hsv_lower = tuple(self.hsv_lower)
        self.hsv_upper = tuple(self.hsv_upper)
        self.prompt = self.prompt or f'A single {self.name.lower()} in plain white or gray background.'


banana = FruitConfig(
    name='banana',
    prompt='A single yellow banana in plain white or gray background.',
    hsv_lower=[(20, 30, 0)],
    hsv_upper=[(100, 255, 255)],
)

tomato_with_calyx = FruitConfig(
    name='tomato',
    num_images=10,
    prompt='A single red tomato in plain white or gray background.',
)

tomato_without_calyx = FruitConfig(
    name='tomato',
    prompt='A single red tomato in plain white or gray background.',
    num_images=50,
    hsv_lower=[(0, 40, 0), (140, 40, 0)],
    hsv_upper=[(30, 255, 255), (200, 255, 255)],
)

def generate_config_from_name(fruit_name: str):
    """Generate config for fruit."""
    return FruitConfig(name=fruit_name)


FRUIT_CONFIGS_FOR_ALL_NAMES: tp.Tuple[FruitConfig, ...] = tuple(
    generate_config_from_name(fruit_name=name) for name in const.FRUITS_NAMES
)
