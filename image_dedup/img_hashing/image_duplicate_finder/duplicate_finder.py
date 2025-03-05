import csv
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import imagehash
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


class ImageDuplicateFinder:
    HASH_FUNCTIONS = {
        "ahash": imagehash.average_hash,
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "whash-haar": imagehash.whash,
        "whash-db4": lambda img: imagehash.whash(img, mode="db4"),
        "colorhash": imagehash.colorhash,
    }

    def __init__(self, hash_type: str = "phash", threshold: int = 8):
        """Initialize the duplicate finder.

        Args:
            hash_type (str): Type of hash function to use
            threshold (int): Maximum hash difference to consider as similar

        """
        if hash_type not in self.HASH_FUNCTIONS:
            raise ValueError(
                f"Invalid hash type. Choose from: {', '.join(self.HASH_FUNCTIONS.keys())}",
            )

        self.hash_func = self.HASH_FUNCTIONS[hash_type]
        self.threshold = threshold

    def find_duplicates(self, folder_path: str) -> dict[str, set[Path]]:
        """Find duplicate or similar images in the given folder.

        This method computes perceptual hashes for all image files in the specified folder
        and groups images with hash differences within the defined threshold.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            Dict[str, Set[Path]]: A dictionary where each key is a hash (as a string), and
                                  the value is a set of image paths considered similar.

        Raises:
            ValueError: If the specified folder does not exist.

        """
        image_hashes = self._compute_image_hashes(folder_path)
        return self._group_similar_images(image_hashes)

    def save_results_to_csv(
        self,
        groups: dict[str, set[Path]],
        csv_path: str,
    ) -> None:
        """Save the groups of similar images to a CSV file.

        Args:
            groups (Dict[str, Set[Path]]): The dictionary of duplicate groups.
            csv_path (str): Path to save the CSV file.

        """
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Group ID", "Image Paths"])

            for group_id, paths in groups.items():
                # Extract only the filenames
                filenames = [path.name for path in sorted(paths)]
                writer.writerow([group_id, "; ".join(filenames)])

        print(f"Duplicate groups saved to {csv_path}")

    def print_results(self, groups: dict[str, set[Path]]) -> None:
        """Print the groups of similar images.

        Args:
            groups (Dict[str, Set[Path]]): The dictionary of duplicate groups.

        """
        if not groups:
            print("No duplicate images found.")
            return

        for i, (group_id, paths) in enumerate(groups.items(), 1):
            print(f"\nGroup {i}:")
            for path in sorted(paths):
                print(f"  {path}")

    def _compute_image_hashes(
        self,
        folder_path: str,
    ) -> dict[Path, imagehash.ImageHash]:
        """Compute hashes for all valid images in the folder.

        Args:
            folder_path (str): Path to the folder containing images.

        Returns:
            Dict[Path, imagehash.ImageHash]: A dictionary of image paths and their computed hashes.

        Raises:
            ValueError: If the folder does not exist.

        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder {folder_path} does not exist")

        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".heic",
        }
        image_hashes = {}
        image_paths = list(folder.rglob("*"))

        for image_path in tqdm(image_paths, desc="Processing images"):
            if image_path.suffix.lower() in image_extensions:
                image_hash = self._compute_hash(image_path)
                if image_hash:
                    image_hashes[image_path] = image_hash

        return image_hashes

    def _compute_hash(self, image_path: Path) -> imagehash.ImageHash:
        """Compute hash for a single image.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            imagehash.ImageHash: The computed hash of the image, or None if processing fails.

        """
        try:
            with Image.open(image_path) as img:
                return self.hash_func(img)
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None

    def _group_similar_images(
        self,
        image_hashes: dict[Path, imagehash.ImageHash],
    ) -> dict[str, set[Path]]:
        """Group images based on hash similarity.

        Args:
            image_hashes (Dict[Path, imagehash.ImageHash]): Dictionary of image paths and their hashes.

        Returns:
            Dict[str, Set[Path]]: A dictionary of grouped similar images.

        """
        groups = defaultdict(set)
        sorted_items = sorted(image_hashes.items(), key=lambda x: str(x[1]))

        for i, (path1, hash1) in enumerate(sorted_items):
            if path1 in groups:
                continue

            group = {path1}
            for path2, hash2 in sorted_items[i + 1 :]:
                if abs(hash1 - hash2) <= self.threshold:
                    group.add(path2)
                else:
                    break

            if len(group) > 1:
                group_id = str(hash1)
                groups[group_id].update(group)

        return dict(groups)

    def remove_duplicates(
        self,
        folder_path: str,
        groups: dict[str, set[Path]],
    ) -> None:
        """Remove duplicate images, keeping only one image from each duplicate group.

        Args:
            folder_path (str): Path to the folder containing images.
            groups (Dict[str, Set[Path]]): The dictionary of duplicate groups.

        This will delete all images in duplicate groups except for one.

        """
        folder = Path(folder_path)

        # Iterate over each group and remove all but one image
        for group_id, paths in groups.items():
            paths = sorted(
                paths,
            )  # Sort paths to keep the first image in the list

            # Keep the first image, remove the rest
            for path in paths[1:]:
                try:
                    os.remove(path)  # Remove the file
                    print(f"Removed duplicate image: {path}")
                except Exception as e:
                    logging.error(f"Error removing {path}: {str(e)}")

        print("Duplicate removal completed.")
