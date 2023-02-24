from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

import const
from src import generate_dataset
from src.train.lightning import BaseFruitSegmentationModule, FruitSegmentationModule
from src.train.model import UnetWrapper


def main(args):
    """Main function. Generate images and masks for all fruits, train segmentation model."""
    seed_everything(const.SEED, workers=True)
    if not args.skip_generation:
        generate_dataset.generate_images_for_fruits()
        generate_dataset.generate_masks_for_all_fruits()

    if args.skip_train:
        return

    batch_size = args.batch
    net = UnetWrapper
    if not batch_size:
        model = BaseFruitSegmentationModule(model=net(), batch_size=1)
        trainer = Trainer(auto_scale_batch_size='power')
        batch_size = trainer.tune(model)['scale_batch_size']

    model = FruitSegmentationModule(
        model=net(), batch_size=batch_size, real_images=tuple(const.REAL_FRUITS_DIR.rglob('*.jpeg'))
    )
    wandb_logger = WandbLogger(project="this-fruit-does-not-exist", save_dir=const.LOG_DIR)
    wandb_logger.watch(model)
    trainer = Trainer.from_argparse_args(args, logger=wandb_logger)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--batch', type=int, action='store', default=None, help='Batch size. Default: find batch size with Trainer'
    )
    parser.add_argument(
        '--skip-generation', action='store_true', help='If used, skip images generation'
    )
    parser.add_argument(
        '--skip-train', action='store_true', help='If used, skip model training'
    )
    parser = Trainer.add_argparse_args(parser)
    arguments = parser.parse_args()
    main(arguments)
