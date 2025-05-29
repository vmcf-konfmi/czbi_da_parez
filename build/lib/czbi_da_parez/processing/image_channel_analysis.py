"""
image_channel_analysis.py
========================
Image channel analysis and mask extraction utilities.

Classes:
    ImageChannelAnalyzer: Provides methods for image info and versioning.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import gaussian, threshold_otsu, sobel
from skimage.measure import label, regionprops, regionprops_table, perimeter
from skimage.morphology import remove_small_objects, skeletonize
from skimage.exposure import rescale_intensity
from stardist.models import StarDist2D
from scipy.ndimage import distance_transform_edt

class ImageChannelAnalyzer:
    """
    Provides methods for image channel analysis and mask extraction.
    """
    @staticmethod
    def version() -> str:
        """
        Returns the version of the package.

        Returns:
            str: Version string.
        """
        return "2025-04-25"

    @staticmethod
    def image_info(image_path: str):
        """
        Get image size and number of channels.

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
