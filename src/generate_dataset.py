import typing as tp
from pathlib import Path

import cv2
import numpy as np

import const
from src.components import utils
from src.components import remove_background as rb
from src.components.stable_diffusion import StableDiffusionGenerator
from src.fruit_config import FruitConfig, FRUIT_CONFIGS_FOR_ALL_NAMES


def generate_images_for_fruit(cfg: FruitConfig, generator: tp.Optional[StableDiffusionGenerator] = None):
    """Generate images for fruit."""
    generator = generator or StableDiffusionGenerator()
    is_batch_size_divisor_of_num_images = cfg.num_images % const.STABLE_DIFFUSION_BATCH_SIZE == 0
    images = generator.generate_images(
        prompt=cfg.prompt,
        num_images_per_prompt=const.STABLE_DIFFUSION_BATCH_SIZE,
        num_runs=(cfg.num_images // const.STABLE_DIFFUSION_BATCH_SIZE) + int(not is_batch_size_divisor_of_num_images),
    )
    images = images[:cfg.num_images]
    generator.save_images(images, const.DATA_DIR / cfg.name, cfg.prompt)
    return images

def generate_images_for_fruits():
    """Generate images for all fruits."""
    generator = StableDiffusionGenerator()
    for cfg in FRUIT_CONFIGS_FOR_ALL_NAMES:
        generate_images_for_fruit(cfg, generator=generator)

def generate_masks(
        cfg: FruitConfig, save_examples: bool = True, grabcut: bool = True, split_mask: bool = True
) -> tp.List[const.SampleType]:
    """Generate and return masks for fruit, use parameters from config."""
    samples: tp.List[const.SampleType] = []
    for image_path in (const.DATA_DIR / cfg.name).rglob('*.jpg'):
        image = utils.read_image(image_path)
        masks_dir = image_path.parent / 'masks'
        masks_dir.mkdir(exist_ok=True)
        example_dir = image_path.parent / 'examples'
        example_dir.mkdir(exist_ok=True)

        if grabcut:
            mask = rb.bg_remove_grabcut_and_hsv(image=image, hsv_lower=cfg.hsv_lower, hsv_upper=cfg.hsv_upper)
        else:
            mask = rb.bg_remove_hsv(image=image, hsv_lower=cfg.hsv_lower, hsv_upper=cfg.hsv_upper)

        masks: tp.List[np.ndarray] = rb.split_mask(mask) + [mask] if split_mask else [mask]
        for i, mask in enumerate(masks):
            mask = mask > cfg.mask_threshold
            if cfg.size_limit[0] < mask.sum() < cfg.size_limit[1]:
                mask_path = str(masks_dir / f'{image_path.stem}_mask{i}.png')
                cv2.imwrite(mask_path, mask * 255)
                samples.append({'image_path': str(image_path), 'mask_path': mask_path, 'fruit_name': cfg.name})
                if save_examples and i == len(masks) - 1:
                    image_path = Path(image_path)
                    example_path = str(example_dir / f'{image_path.stem}_example.jpeg')
                    mask = (mask * 255).astype('uint8')
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) > 0.5
                    example = cv2.cvtColor(mask * image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(example_path, example)
    return samples

def generate_background():
    """Generate background."""
    generator = StableDiffusionGenerator()
    images = generator.generate_images(
        prompt='A shiny squared metal tray view from above.',
        num_images_per_prompt=const.STABLE_DIFFUSION_BATCH_SIZE,
        num_runs=(const.NUM_BACKGROUND_IMAGES // const.STABLE_DIFFUSION_BATCH_SIZE) + 1,
    )
    images = images[:const.NUM_BACKGROUND_IMAGES]
    generator.save_images(images, const.BACKGROUND_DIR, 'background', suffix='.jpeg')

def generate_masks_for_all_fruits(dataset_name: str = 'generated_dataset.json', grabcut: bool = True):
    """Generate images for all fruits."""
    samples: tp.List[const.SampleType] = []
    for cfg in FRUIT_CONFIGS_FOR_ALL_NAMES:
        samples.extend(generate_masks(cfg, grabcut=grabcut))
    utils.save_json(const.DATA_DIR / dataset_name, samples)


def show_masks(cfg: FruitConfig) -> None:
    """Show masks for fruit, use parameters from config."""
    for image_path in (const.DATA_DIR / cfg.name).rglob('*.jpg'):
        image = utils.read_image(image_path)
        mask = rb.bg_remove_hsv(image=image, hsv_lower=cfg.hsv_lower, hsv_upper=cfg.hsv_upper)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask: np.ndarray = mask > cfg.mask_threshold
        mask_size = mask[..., 0].sum()
        if cfg.size_limit[0] < mask_size < cfg.size_limit[1]:
            masked_img = mask * image
            utils.show_image(masked_img)
