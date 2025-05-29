"""
image_channel_analysis.py
========================
Image channel analysis and mask extraction utilities.

Module-level functions for image channel analysis and mask extraction.
All functions are at the module level for Sphinx compatibility and batch workflows.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.exposure import rescale_intensity
from stardist.models import StarDist2D

def image_info(image_path):
    """
    Get image size and number of channels.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: Image size and number of channels.
    """
    try:
        img = imread(image_path)
        return img.shape[:2], img.shape[0]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def normalize(image):
    """Normalizes the image intensity to the range [0, 1]."""
    return rescale_intensity(image, in_range='image', out_range=(0, 1))

def plot_normalized_channels(channel_1, channel_2):
    """
    Plots the normalized channels.

    Args:
        channel_1: The first image channel (e.g., nuclei).
        channel_2: The second image channel (e.g., membrane).
    """
    normalized_channel_1 = normalize(channel_1)
    normalized_channel_2 = normalize(channel_2)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(normalized_channel_1, cmap='gray')
    plt.title('Normalized Channel 1 (Nuclei)')
    plt.subplot(1, 2, 2)
    plt.imshow(normalized_channel_2, cmap='gray')
    plt.title('Normalized Channel 2 (Membrane)')
    plt.tight_layout()
    plt.show()

def image_th_ch1(image):
    """
    Processes channel 1 of the image using StarDist2D model.

    Args:
        image: The input image.

    Returns:
        tuple: Labeled image, None, and binary mask.
    """
    try:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        label_image, _ = model.predict_instances(rescale_intensity(image, in_range='image', out_range=(0, 1)))
        return label_image, None, (label_image > 0).astype(int)
    except Exception as e:
        print(f"Error processing {image}: {e}")
        return None, None, None

def image_th_ch2(image):
    """
    Processes channel 2 of the image using Gaussian blur and Otsu thresholding.

    Args:
        image: The input image.

    Returns:
        tuple: Labeled image, blurred image, and thresholded image.
    """
    try:
        blurred_image = gaussian(image, sigma=5)
        thresholded_image = blurred_image >= threshold_otsu(blurred_image)
        return label(thresholded_image), blurred_image, thresholded_image
    except Exception as e:
        print(f"Error processing {image}: {e}")
        return None, None, None
