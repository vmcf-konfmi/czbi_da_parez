"""
quality.py
==========
Quality checking utilities for image datasets.

Classes:
    QualityChecker: Provides methods for checking image paths and directory structure.
"""

import os
from skimage.io import imread
import pandas as pd

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
        missing_images = []
        for index, row in df.iterrows():
            image_path = row[image_path_column]
            if not os.path.exists(image_path):
                missing_images.append(image_path)
        if missing_images:
            print("Missing images:")
            for image_path in missing_images:
                print(image_path)
        else:
            print("All images found.")

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
    def image_info(image_path: str):
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
            num_channels = 1 if len(img.shape) == 2 else img.shape[0]
            pixel_size_nm = "Unknown"  # Replace with actual pixel size if available
            return image_size, num_channels, pixel_size_nm
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None, None
