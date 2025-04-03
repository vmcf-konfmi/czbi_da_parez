import os
from skimage.io import imread
import pandas as pd


def check_image_paths(df, image_path_column):
    """
    Checks if image paths in a DataFrame column exist.

    Args:
        df: Pandas DataFrame containing the image paths.
        image_path_column: Name of the column containing image paths.

    Returns:
        None. Prints missing image paths.
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

def print_directory_tree(root_dir, indent=0):
    """Prints a directory tree starting from the given root directory."""
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        print("  " * indent + item)
        if os.path.isdir(item_path):
            print_directory_tree(item_path, indent + 1)

def image_info(image_path):
    try:
        img = imread(image_path)
        image_size = img.shape[:2]  # Get height and width
        num_channels = 1 if len(img.shape) == 2 else img.shape[0]
        # Placeholder for pixel size, replace with actual calculation if available
        pixel_size_nm = "Unknown" # Replace with actual pixel size in nm
        return image_size, num_channels, pixel_size_nm
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None

def print_directory_tree(root_dir, indent=0):
    """Prints a directory tree starting from the given root directory."""

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        print("  " * indent + item)  # Print with indentation
        if os.path.isdir(item_path):
            print_directory_tree(item_path, indent + 1)