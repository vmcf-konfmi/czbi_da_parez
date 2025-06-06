"""
quality.py
==========
Quality and region-based analysis utilities.

Functions for measuring brightness variability, nuclei under membrane, and related metrics.
"""

import numpy as np

def measure_brightness_variability(intensity_image, mask):
    """
    Measures the variability of brightness values under a mask.

    Args:
        intensity_image: The image containing brightness values.
        mask: A binary mask where 1 indicates the region of interest.

    Returns:
        dict: Brightness variability metrics (e.g., standard deviation).
    """
    masked_intensities = intensity_image[mask > 0]
    if len(masked_intensities) == 0:
        return None
    return {'std_dev': np.std(masked_intensities)}

def count_nuclei_under_membrane(nuclei_labels, membrane_mask):
    """
    Counts the number of nuclei under the membrane.

    Args:
        nuclei_labels: Labeled nuclei image.
        membrane_mask: Binary mask of the membrane.

    Returns:
        int: Number of nuclei under the membrane.
    """
    nuclei_under_membrane = np.unique(nuclei_labels[membrane_mask > 0])
    return len(nuclei_under_membrane[nuclei_under_membrane != 0])

def percentage_nuclei_pixels(nuclei_labels, membrane_mask):
    """
    Calculates the percentage of nuclei pixels within the membrane.

    Args:
        nuclei_labels: Labeled nuclei image.
        membrane_mask: Binary mask of the membrane.

    Returns:
        float: Percentage of nuclei pixels within the membrane.
    """
    total_nuclei_pixels_under_membrane = np.sum(membrane_mask[nuclei_labels > 0])
    total_nuclei_pixels = np.sum(nuclei_labels > 0)
    return (total_nuclei_pixels_under_membrane / total_nuclei_pixels) * 100 if total_nuclei_pixels else 0
