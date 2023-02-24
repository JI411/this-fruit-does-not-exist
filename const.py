"""
Main constants: paths, params and etc.
"""
import typing as tp

from pathlib import Path


# Typing aliases
PathType = tp.Union[Path, str]
ColorRegionType = tp.Tuple[int, int, int]
SampleType = tp.TypedDict('SampleType', {'image_path': str, 'mask_path': str, 'fruit_name': str})

# Models
STABLE_DIFFUSION_BATCH_SIZE: tp.Final[int] = 5
SEED: tp.Final[int] = 411

# Paths
ROOT_DIR = Path(__file__).resolve(strict=True).parent

SRC_DIR: tp.Final[Path] = ROOT_DIR / 'src'
DATA_DIR: tp.Final[Path] = ROOT_DIR / 'data'
LOG_DIR: tp.Final[Path] = ROOT_DIR / 'logs'


# Fruits
FRUITS_NAMES: tp.Tuple[str, ...] = (
    'Apple', 'Banana', 'Carambola',
    # 'Guava', 'Kiwi', 'Mango', 'muskmelon',
    # 'Orange', 'Peach', 'Pear', 'Persimmon',
    # 'Pitaya', 'Plum', 'Pomegranate', 'Tomatoes'
)
