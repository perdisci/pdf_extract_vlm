"""
This script compares the perceptual hash distance between two images.
Perceptual hashing (pHash) is useful for identifying similar images even if
they have been resized, compressed, or slightly modified.

Usage:
    source venv/bin/activate
    python3 tests/compare_images.py --img1 path/to/image1.png --img2 path/to/image2.png
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import imagehash


def compare_images(img1_path, img2_path):
    """
    Calculates the Hamming distance between the perceptual hashes of two images.
    A distance of 0 indicates the images are likely identical.
    Higher distances indicate more significant differences.
    """
    path1 = Path(img1_path)
    path2 = Path(img2_path)

    if not path1.exists():
        print(f"Error: Image 1 not found at {path1}")
        sys.exit(1)
    if not path2.exists():
        print(f"Error: Image 2 not found at {path2}")
        sys.exit(1)

    try:
        # Open images and calculate pHash
        hash0 = imagehash.phash(Image.open(path1))
        hash1 = imagehash.phash(Image.open(path2))

        # The difference between hashes is the Hamming distance
        distance = hash1 - hash0

        print(f"Image 1: {path1.name} (Hash: {hash0})")
        print(f"Image 2: {path2.name} (Hash: {hash1})")
        print(f"Perceptual Distance (Hamming): {distance}")

        if distance == 0:
            print("Conclusion: The images are likely identical.")
        elif distance <= 5:
            print("Conclusion: The images are very similar.")
        elif distance <= 10:
            print("Conclusion: The images are somewhat similar.")
        else:
            print("Conclusion: The images are significantly different.")

        return distance

    except Exception as e:
        print(f"An error occurred while comparing images: {e}")
        sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare two images using perceptual hashing."
    )
    parser.add_argument(
        "--img1", type=str, required=True, help="Path to the first image."
    )
    parser.add_argument(
        "--img2", type=str, required=True, help="Path to the second image."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    compare_images(args.img1, args.img2)
