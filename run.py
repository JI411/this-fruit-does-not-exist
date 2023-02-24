from src import generate_dataset

def main():
    """Main function. Generate images and masks for all fruits."""
    generate_dataset.generate_images_for_fruits()
    generate_dataset.generate_masks_for_all_fruits()


if __name__ == '__main__':
    main()
