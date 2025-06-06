"""
quality.py
==========
Quality checking utilities for image datasets.

Classes:
    QualityChecker: Provides methods for checking image paths and directory structure.
"""

import os
import logging
from skimage.io import imread
import pandas as pd
from typing import Optional, Tuple

class QualityChecker:
    """
    Provides methods for checking image paths and directory structure.
    """
    @staticmethod
    def check_image_paths(df: pd.DataFrame, image_path_column: str) -> None:
        """
        Checks if image paths in a DataFrame column exist.

        Args:
            df (pd.DataFrame): DataFrame containing the image paths.
            image_path_column (str): Name of the column containing image paths.
        """
        missing_images = df[~df[image_path_column].apply(os.path.exists)][image_path_column].tolist()
        if missing_images:
            logging.warning("Missing images:")
            for image_path in missing_images:
                logging.warning(image_path)
        else:
            logging.info("All images found.")

    @staticmethod
    def print_directory_tree(root_dir: str, indent: int = 0) -> None:
        """
        Prints a directory tree starting from the given root directory.

        Args:
            root_dir (str): Root directory to print.
            indent (int): Indentation level (used internally).
        """
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            print("  " * indent + item)
            if os.path.isdir(item_path):
                QualityChecker.print_directory_tree(item_path, indent + 1)

    @staticmethod
    def image_info(image_path: str) -> Tuple[Optional[Tuple[int, int]], Optional[int], str]:
        """
        Get image size, number of channels, and pixel size (if available).

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: (image_size, num_channels, pixel_size_nm)
        """
        try:
            img = imread(image_path)
            image_size = img.shape[:2]  # height, width
            if len(img.shape) == 2:
                num_channels = 1
            elif len(img.shape) == 3:
                num_channels = img.shape[2]
            else:
                num_channels = "Unknown"
            pixel_size_nm = "Unknown"  # Replace with actual pixel size if available
            return image_size, num_channels, pixel_size_nm
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            return None, None, None

    @staticmethod
    def check_image_readability(df: pd.DataFrame, image_path_column: str) -> None:
        """
        Checks if images in a DataFrame column can be read (not just if the path exists).

        Args:
            df (pd.DataFrame): DataFrame containing the image paths.
            image_path_column (str): Name of the column containing image paths.
        """
        unreadable_images = []
        for image_path in df[image_path_column]:
            try:
                _ = imread(image_path)
            except Exception as e:
                unreadable_images.append((image_path, str(e)))
        if unreadable_images:
            logging.warning("Unreadable images:")
            for image_path, err in unreadable_images:
                logging.warning(f"{image_path}: {err}")
        else:
            logging.info("All images are readable.")
