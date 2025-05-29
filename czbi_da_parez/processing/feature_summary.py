"""
feature_summary.py
==================
Feature summary and measurement utilities.

Functions for measuring membrane pixel count, thickness, and summarizing features.
"""

import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table

def measure_membrane_pixel_count(label_image):
    """
    Measures the total pixel count of the membrane regions.

    Args:
        label_image: Labeled image of membranes.

    Returns:
        int: Total pixel count of membrane regions.
    """
    return np.sum(label_image > 0)

def measure_membrane_thickness(label_image, intensity_image):
    """
    Measures membrane thickness statistics (median, min, max).

    Args:
        label_image: Labeled image of membranes.
        intensity_image: The original intensity image.

    Returns:
        dict: Median, minimum, and maximum thickness values.
    """
    regions = regionprops(label_image, intensity_image=intensity_image)
    thickness_values = [region.equivalent_diameter for region in regions]
    if thickness_values:
        return {'median_thickness': np.median(thickness_values), 'min_thickness': np.min(thickness_values), 'max_thickness': np.max(thickness_values)}
    return None

def measure_image(label_image, intensity_image, properties):
    """
    Measures image properties using regionprops_table.

    Args:
        label_image: Labeled image.
        intensity_image: Intensity image.
        properties: List of properties to measure.

    Returns:
        pandas.DataFrame: DataFrame of measured properties.
    """
    def quartiles(regionmask, intensity):
        return np.percentile(intensity[regionmask], q=(5, 10, 25, 50, 75, 90, 95))
    feature_table = regionprops_table(label_image, intensity_image, properties, extra_properties=(quartiles,))
    return pd.DataFrame(feature_table)

def summarize_features(df):
    """
    Calculates summary statistics for a DataFrame and returns a new DataFrame with one row.

    Args:
        df: Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with summary statistics.
    """
    summary_stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary_stats.update({f"{col}_{stat}": getattr(df[col], stat)() for stat in ['mean', 'std', 'min', 'max', 'median', 'count']})
        else:
            summary_stats[f"{col}_unique_count"] = df[col].nunique()
    return pd.DataFrame([summary_stats])
