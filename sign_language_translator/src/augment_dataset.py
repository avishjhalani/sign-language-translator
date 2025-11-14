"""
Dataset Augmentation Utility
Generates additional training images with common image transformations
to improve model robustness.
"""

import argparse
import os
import glob
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DEFAULT_INPUT_DIR = "../dataset"
DEFAULT_OUTPUT_DIR = "../dataset_augmented"


def build_datagen():
    """Create an ImageDataGenerator with useful augmentations."""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.7, 1.3),
        horizontal_flip=True,
        fill_mode="nearest"
    )


def augment_image(image_path, output_dir, datagen, augmentations_per_image, prefix):
    """Generate augmented variants for a single image."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Failed to read image {image_path}")
        return 0

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.expand_dims(image_rgb, 0)

    generated = 0
    # flow() yields batches indefinitely; stop after desired count
    for batch in datagen.flow(image_rgb, batch_size=1, shuffle=False):
        augmented = batch[0].astype(np.float32)
        augmented = np.clip(augmented, 0, 255).astype(np.uint8)
        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        output_name = f"{prefix}_aug_{generated:03d}.jpg"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), augmented_bgr)

        generated += 1
        if generated >= augmentations_per_image:
            break

    return generated


def augment_dataset(input_dir, output_dir, augmentations_per_image, copy_original):
    """Augment every class folder in the input directory."""
    datagen = build_datagen()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_folders = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not class_folders:
        print(f"No class folders found in {input_dir}. Nothing to augment.")
        return

    total_generated = 0
    for class_folder in class_folders:
        image_paths = sorted(
            glob.glob(str(class_folder / "*.jpg")) + glob.glob(str(class_folder / "*.png"))
        )
        if not image_paths:
            print(f"Skipping {class_folder.name}: no images found.")
            continue

        class_output_dir = output_dir / class_folder.name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nAugmenting class '{class_folder.name}' ({len(image_paths)} images)...")

        # Optionally copy originals to the augmented directory
        if copy_original:
            for image_path in image_paths:
                image = cv2.imread(image_path)
                if image is None:
                    continue
                output_name = Path(image_path).name
                cv2.imwrite(str(class_output_dir / output_name), image)

        for image_path in image_paths:
            prefix = Path(image_path).stem
            generated = augment_image(
                image_path=image_path,
                output_dir=class_output_dir,
                datagen=datagen,
                augmentations_per_image=augmentations_per_image,
                prefix=prefix
            )
            total_generated += generated

    print(f"\nAugmentation complete! Generated {total_generated} new images.")
    print(f"Augmented dataset saved at: {output_dir.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment the sign language image dataset with synthetic variants."
    )
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Path to the original dataset directory (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write augmented images (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--augmentations_per_image",
        type=int,
        default=3,
        help="Number of augmented images to generate per original image (default: 3)"
    )
    parser.add_argument(
        "--copy_original",
        action="store_true",
        help="If set, copy original images into the output directory alongside augmented ones."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        augmentations_per_image=args.augmentations_per_image,
        copy_original=args.copy_original
    )


if __name__ == "__main__":
    main()

