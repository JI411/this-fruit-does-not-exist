from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

import const
from src.components import utils


class StableDiffusionGenerator:
    """Stable Diffusion Generator."""

    def __init__(self):
        """Init Stable Diffusion Generator."""
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        utils.seed_everything(const.SEED)

    def run(self, prompt: str, num_images_per_prompt: int = 3):
        """Generate images from prompt."""
        return self.pipe(prompt, num_images_per_prompt=num_images_per_prompt).images

    def generate_images(self, prompt: str, num_images_per_prompt: int = 3, num_runs: int = 10):
        """Generate images from prompt multiple times."""
        all_images = []
        for _ in range(num_runs):
            images = self.run(prompt, num_images_per_prompt=num_images_per_prompt)
            all_images.extend(images)
        return all_images

    @staticmethod
    def save_images(images, save_path: const.PathType, name: str, suffix: str = '.jpg') -> None:
        """Save images to disk."""
        name = name.lower().replace(' ', '_').replace('.', '')
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        for i, image in enumerate(images):
            image.save(save_path / f'{name}_{i}{suffix}')
