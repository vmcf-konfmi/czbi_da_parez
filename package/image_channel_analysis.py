#!/usr/bin/env python
# coding: utf-8
# # X-channel analysis, mask extraction, extended features
# from colab/01_x-channel-analysis_mask-extraction_extended-features-v2_better-names-20250327.py

# Last updated: 2025-03-27T14:33:41.507795+00:00
#
#  * Python implementation: CPython
#  * Python version       : 3.11.11
#  * IPython version      : 7.34.0
#
#  * Compiler    : GCC 11.4.0
#  * OS          : Linux
#  * Release     : 6.1.85+
#  * Machine     : x86_64
#  * Processor   : x86_64
#  * CPU cores   : 2
#  * Architecture: 64bit
#
#  * cv2       : 4.11.0
#  * skimage   : 0.25.2
#  * pandas    : 2.2.2
#  * numpy     : 1.26.4
#  * stardist  : 0.9.1
#  * tensorflow: 2.18.0

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
    # Normalize channels
    normalized_channel_1 = normalize(channel_1)
    normalized_channel_2 = normalize(channel_2)

    # Plot normalized channels
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(normalized_channel_1, cmap='gray')
    plt.title('Normalized Channel 1 (Nuclei)')

    plt.subplot(1, 2, 2)
    plt.imshow(normalized_channel_2, cmap='gray')
    plt.title('Normalized Channel 2 (Membrane)')

    plt.tight_layout()
    plt.show()

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

def save_label_mask(image_path, labels, output_folder):
    """
    Saves the label image and mask as TIFF files in the specified folder.

    Args:
        image_path (str): Path to the input image.
        labels (numpy.ndarray): Label image data.
        output_folder (str): Path to the output folder.
    """
    try:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        imsave(os.path.join(output_folder, f"{image_name}_label.tiff"), labels.astype(np.uint16))
        imsave(os.path.join(output_folder, f"{image_name}_mask.tiff"), (labels > 0).astype(np.uint8))
    except Exception as e:
        print(f"Error saving images: {e}")

def create_mask_from_labels(labels):
    """
    Creates a binary mask from a label image.

    Args:
        labels: A 2D numpy array representing the label image.

    Returns:
        numpy.ndarray: Binary mask.
    """
    return np.where(labels > 0, 1, 0)

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
        return label_image, None, create_mask_from_labels(label_image)
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

def quartiles(regionmask, intensity):
    """
    Calculates quartiles for the given region mask and intensity.

    Args:
        regionmask: Binary mask of the region.
        intensity: Intensity image.

    Returns:
        numpy.ndarray: Quartiles.
    """
    return np.percentile(intensity[regionmask], q=(5, 10, 25, 50, 75, 90, 95))

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

def process_imag_old(image_path):
    """
    Processes an image and returns information about it.

    Args:
        image_path: The path to the image file.

    Returns: summary_df_ch1, summary_df_ch2 as statistical summary
    """
    try:
        image_size, num_channels = image_info(image_path)
        if image_size is None or num_channels != 2:
            return None

        img = imread(image_path)
        output_folder = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        channel_2 = img[1, :, :]
        label_image_ch2, _, _ = image_th_ch2(channel_2)
        save_label_mask(image_path, label_image_ch2, output_folder)
        df_features_ch2 = measure_image(label_image_ch2, channel_2, properties=['area', 'perimeter', 'centroid', 'bbox', 'solidity', 'mean_intensity', 'major_axis_length', 'minor_axis_length'])
        df_features_ch2.to_csv(os.path.join(output_folder, f"{image_name}_features_ch2.csv"), index=False)
        summary_df_ch2 = summarize_features(df_features_ch2)

        channel_1 = img[0, :, :]
        label_image_ch1, _, _ = image_th_ch1(channel_1)
        save_label_mask(image_path, label_image_ch1, output_folder)
        df_features_ch1 = measure_image(label_image_ch1, channel_1, properties=['area', 'perimeter', 'solidity', 'max_intensity', 'mean_intensity', 'min_intensity', 'major_axis_length', 'minor_axis_length'])
        df_features_ch1.to_csv(os.path.join(output_folder, f"{image_name}_features_ch1.csv"), index=False)
        summary_df_ch1 = summarize_features(df_features_ch1)

        specific_df = pd.DataFrame(index=[0])
        specific_df['nuclei_under_membrane'] = count_nuclei_under_membrane(label_image_ch1, create_mask_from_labels(label_image_ch2))
        specific_df['percentage_nuclei_pixels'] = percentage_nuclei_pixels(label_image_ch1, create_mask_from_labels(label_image_ch2))
        specific_df['membrane_pixel_count'] = measure_membrane_pixel_count(label_image_ch2)
        thickness_stats = measure_membrane_thickness(label_image_ch2, channel_2)
        if thickness_stats:
            specific_df.update(thickness_stats)
        brightness_variability = measure_brightness_variability(channel_2, create_mask_from_labels(label_image_ch2))
        if brightness_variability:
            specific_df['membrane_brightness_std_dev'] = brightness_variability['std_dev']

        return summary_df_ch1, summary_df_ch2, specific_df
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_image(image_path):
    """
    Processes an image and returns information about it.

    Args:
        image_path: The path to the image file.

    Returns: summary_df_ch1, summary_df_ch2 as statistical summary
    """
    try:
        image_size, num_channels = image_info(image_path)
        if image_size is None or num_channels != 2:
            return None

        img = imread(image_path)
        output_folder = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        #####################################################
        ## membrane
        channel_2 = img[1, :, :]
        label_image_ch2, _, _ = image_th_ch2(channel_2)
        save_label_mask(image_path, label_image_ch2, output_folder)
        df_features_ch2 = measure_image(label_image_ch2, channel_2, properties=['area', 'perimeter', 'centroid', 'bbox', 'solidity', 'mean_intensity', 'major_axis_length', 'minor_axis_length'])
        ## save full image features csv before squashing in one row
        df_features_ch2.to_csv(os.path.join(output_folder, f"{image_name}_features_ch2.csv"), index=False)
        summary_df_ch2 = summarize_features(df_features_ch2)

        #####################################################
        ## membrane gradient
        # TODO: gradient_analysis(image_path,vis=False) make proper moprhometrics
        gradient_mask = gradient_analysis(image_path, verbose=False, vis=False)
        df_gradient_features_ch2 = measure_image(label(gradient_mask), channel_2,
                                        properties=['area', 'perimeter', 'centroid', 'bbox', 'solidity',
                                                    'mean_intensity', 'major_axis_length', 'minor_axis_length'])

        #####################################################
        ## membrane bright spots
        # TODO: analyze_bright_spots(image_path,vis=False) make proper moprhometrics
        bright_spots_mask = analyze_bright_spots(image_path, vis=False)
        df_bright_spots_features_ch2 = measure_image(label(bright_spots_mask), channel_2,
                                                 properties=['area', 'perimeter', 'centroid', 'bbox', 'solidity',
                                                             'mean_intensity', 'major_axis_length',
                                                             'minor_axis_length'])

        #####################################################
        ## membrane detect boundary
        # TODO: detect_boundary(image_path,vis=False) make proper moprhometrics
        boundary_mask = detect_boundary(image_path, vis=False)
        df_boundary_features_ch2 = measure_image(label(boundary_mask), channel_2,
                                                     properties=['area', 'perimeter', 'centroid', 'bbox', 'solidity',
                                                                 'mean_intensity', 'major_axis_length',
                                                                 'minor_axis_length'])

        #####################################################
        ## membrane continuity
        cleaned_channel_2, cleaned_membrane = prep_for_membrane_analysis(image_path,vis=False)
        membrane_continuity_metrics = analyze_membrane_continuity(cleaned_channel_2, cleaned_membrane)
        summary_df_ch2 = pd.concat([summary_df_ch2, membrane_continuity_metrics, df_gradient_features_ch2, df_bright_spots_features_ch2, df_boundary_features_ch2], axis=1)

        #####################################################
        ## nuclei
        channel_1 = img[0, :, :]
        label_image_ch1, _, _ = image_th_ch1(channel_1)
        save_label_mask(image_path, label_image_ch1, output_folder)
        df_features_ch1 = measure_image(label_image_ch1, channel_1, properties=['area', 'perimeter', 'solidity', 'max_intensity', 'mean_intensity', 'min_intensity', 'major_axis_length', 'minor_axis_length'])
        df_features_ch1.to_csv(os.path.join(output_folder, f"{image_name}_features_ch1.csv"), index=False)
        summary_df_ch1 = summarize_features(df_features_ch1)

        specific_df = pd.DataFrame(index=[0])
        specific_df['nuclei_under_membrane'] = count_nuclei_under_membrane(label_image_ch1, create_mask_from_labels(label_image_ch2))
        specific_df['percentage_nuclei_pixels'] = percentage_nuclei_pixels(label_image_ch1, create_mask_from_labels(label_image_ch2))
        specific_df['membrane_pixel_count'] = measure_membrane_pixel_count(label_image_ch2)
        thickness_stats = measure_membrane_thickness(label_image_ch2, channel_2)
        if thickness_stats:
            # Convert thickness_stats to DataFrame with index
            thickness_df = pd.DataFrame(thickness_stats, index=[0])
            specific_df = pd.concat([specific_df, thickness_df], axis=1) # Concatenate with specific_df
            # specific_df.update(thickness_stats) # Remove this line
        brightness_variability = measure_brightness_variability(channel_2, create_mask_from_labels(label_image_ch2))
        if brightness_variability:
            specific_df['membrane_brightness_std_dev'] = brightness_variability['std_dev']

        return summary_df_ch1, summary_df_ch2, specific_df
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def detect_boundary(image_path, vis=False):
    """
    Detects the boundary of the membrane in the image.

    Args:
        image_path: Path to the image file.
    """
    img = imread(image_path)
    channel_2 = img[1, :, :]
    image_8bit = cv2.convertScaleAbs(channel_2, alpha=(255.0/65535.0))
    blurred = cv2.GaussianBlur(cv2.medianBlur(image_8bit, 15), (13, 13), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
    if vis:
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(image_8bit, cmap='gray')
        axes[0].set_title("Original Image (8-bit Scaled)")
        axes[1].imshow(result)
        axes[1].set_title("Detected Boundary")
        axes[2].imshow(blurred)
        axes[2].set_title("Blurred")
        for ax in axes:
            ax.axis("off")
        plt.show()
    # Create a blank mask with the same dimensions as the thresholded image
    mask = np.zeros_like(thresh, dtype=np.uint8)

    # Draw the contours onto the mask
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    return mask

def detect_boundary_wider(image_path, vis=False):
    """
        Detects the boundary of the membrane in the image.

        Args:
            image_path: Path to the image file.
    """
    img = imread(image_path)

    #####################################################
    ## membrane
    channel_2 = img[1, :, :]

    #####################################################
    ## nuclei
    channel_1 = img[0, :, :]

    # def detect_boundary(image_path):
    # Load image
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    gray = channel_2

    # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize to 8-bit for processing
    image_8bit = cv2.convertScaleAbs(gray, alpha=(255.0/65535.0))

    # Apply Median Filter to reduce salt-and-pepper noise
    median_filtered = cv2.medianBlur(image_8bit, 15)

    # median_filtered = cv2.medianBlur(median_filtered, 5)

    # median_filtered = cv2.medianBlur(median_filtered, 5)

    # median_filtered = cv2.medianBlur(median_filtered, 5)

    # median_filtered = cv2.medianBlur(median_filtered, 5)

    # blurred = cv2.medianBlur(median_filtered, 5)

    # Apply Gaussian Blur to further smooth the image
    blurred = cv2.GaussianBlur(median_filtered, (13, 13), 0)
    blurred = cv2.GaussianBlur(blurred, (13, 13), 0)
    blurred = cv2.GaussianBlur(blurred, (13, 13), 0)
    blurred = cv2.GaussianBlur(blurred, (13, 13), 0)
    blurred = cv2.GaussianBlur(blurred, (13, 13), 0)
    blurred = cv2.GaussianBlur(blurred, (13, 13), 0)
    blurred = cv2.GaussianBlur(blurred, (13, 13), 0)

    # Apply Otsu's Thresholding (only works on 8-bit images)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_TRUNC)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale 16-bit image to RGB for visualization
    image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    result = image_rgb.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
    if vis:
        # Show the results
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(image_8bit, cmap='gray')
        axes[0].set_title("Original Image (8-bit Scaled)")
        axes[0].axis("off")

        axes[1].imshow(result)
        axes[1].set_title("Detected Boundary")
        axes[1].axis("off")

        axes[2].imshow(blurred)
        axes[2].set_title("blurred")
        axes[2].axis("off")

        plt.show()
    # Create a blank mask with the same dimensions as the thresholded image
    mask = np.zeros_like(thresh, dtype=np.uint8)

    # Draw the contours onto the mask
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    return mask

def gradient_analysis(image_path, verbose=False, vis=False):
    """
    Analyzes the gradient of the membrane channel and returns bright spots.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list: Brightness values for each expansion step.
    """
    img = imread(image_path)

    # Membrane channel
    channel_2 = img[1, :, :]

    # Normalize to 8-bit for processing
    image_8bit = cv2.convertScaleAbs(channel_2, alpha=(255.0/65535.0))

    # Apply Median Filter to reduce salt-and-pepper noise
    median_filtered = cv2.medianBlur(image_8bit, 15)

    # Apply Gaussian Blur to further smooth the image
    blurred = cv2.GaussianBlur(median_filtered, (13, 13), 0)
    for _ in range(6):
        blurred = cv2.GaussianBlur(blurred, (13, 13), 0)

    # Apply Otsu's Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale 16-bit image to RGB for visualization
    image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    result = image_rgb.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

    # Expand the mask outward from the detected boundary
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    step_size = 60
    max_steps = 10  # Number of steps to expand in each direction
    brightness_values = []
    start = True

    for i in range(1, max_steps + 1):
        # Expand outward by dilation
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=i * step_size // 3)

        # Get the new region (difference between expanded mask and original mask)
        if start:
            expansion_region = cv2.bitwise_xor(expanded_mask, mask)
            old_mask = expanded_mask
        else:
            expansion_region = cv2.bitwise_xor(expanded_mask, old_mask)
            old_mask = expanded_mask

        # Find separate connected regions within the expansion region
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(expansion_region, connectivity=8)

        region_brightness = []
        for label in range(1, num_labels):  # Skip background (label 0)
            brightness_values_list = channel_2[labels == label].flatten()
            percentiles = np.percentile(brightness_values_list, [25, 50, 75, 95])
            region_brightness.append((label, percentiles))

        brightness_values.append((i * step_size, region_brightness))
        start = False

    # Show the results
    if vis:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_8bit, cmap='gray')
        axes[0].set_title("Original Image (8-bit Scaled)")
        axes[0].axis("off")

        axes[1].imshow(result)
        axes[1].set_title("Detected Boundary")
        axes[1].axis("off")

        plt.show()

    if verbose:
        # Print brightness percentiles for each expansion step
        print("Expansion Step (pixels) - Brightness in Regions:")
        for step, regions in brightness_values:
            print(f"Step {step} pixels:")
            for region_id, percentiles in regions:
                print(f"  Region {region_id}: 25th={percentiles[0]:.2f}, 50th={percentiles[1]:.2f}, 75th={percentiles[2]:.2f}, 95th={percentiles[3]:.2f}")

    return brightness_values

def remove_large_objects(binary_image, max_size):
    labeled_image = label(binary_image)
    sizes = np.bincount(labeled_image.ravel())  # Count sizes of objects
    mask = sizes <= max_size  # Keep only objects smaller than max_size
    mask[0] = 0  # Ensure background remains
    return mask[labeled_image]

def analyze_bright_spots(image_path, vis=False):
    """
    Analyzes bright spots in the membrane channel of the image.

    Args:
        image_path: Path to the image file.
    """
    img = imread(image_path)

    #####################################################
    ## membrane
    channel_2 = img[1, :, :]

    #####################################################
    ## nuclei
    channel_1 = img[0, :, :]

    # removes small spots, dont use
    # channel_2_blurred = cv2.GaussianBlur(channel_2, (13, 13), 0)

    # Threshold the membrane channel using Otsu's method
    thresh = threshold_otsu(channel_2)
    binary_membrane = channel_2 > thresh

    # #dilate and erode to connect parts that are too close to membrane
    # # Expand outward by dilation
    # kernel = np.ones((3, 3), np.uint8)
    # expanded_mask = cv2.dilate(binary_membrane, kernel)

    # # Erode mask back
    # erode_mask = cv2.erode(expanded_mask, kernel)

    # Convert binary mask to uint8 for OpenCV
    binary_membrane_uint8 = (binary_membrane * 255).astype(np.uint8)

    # Define kernel
    kernel = np.ones((5, 5), np.uint8)

    # Expand outward by dilation
    expanded_mask = cv2.dilate(binary_membrane_uint8, kernel)
    expanded_mask = cv2.dilate(expanded_mask, kernel)
    expanded_mask = cv2.dilate(expanded_mask, kernel)

    # Erode mask back
    eroded_mask = cv2.erode(expanded_mask, kernel)
    eroded_mask = cv2.erode(eroded_mask, kernel)
    eroded_mask = cv2.erode(eroded_mask, kernel)

    # Convert back to boolean if needed
    final_mask = eroded_mask > 0

    # Remove large objects (connected components) larger than 81 pixels
    cleaned_membrane = remove_large_objects(
        final_mask, max_size=81
    )  # min_size is exclusive, so use 51 to remove objects > 50

    # Remove too small objects
    cleaned_membrane = remove_small_objects(cleaned_membrane, min_size=3)

    # Optional: Label the remaining objects (if you need to analyze them individually)
    labeled_membrane = label(cleaned_membrane)

    # Example: Count the remaining objects
    num_objects = np.max(labeled_membrane)
    print(f"Number of remaining objects: {num_objects}")

    if vis:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(channel_2, cmap="gray")
        plt.title("Original Membrane Channel")

        plt.subplot(1, 3, 2)
        plt.imshow(binary_membrane, cmap="gray")
        plt.title("Binary Membrane (Otsu Threshold)")

        plt.subplot(1, 3, 3)
        plt.imshow(cleaned_membrane, cmap="gray")
        plt.title("Cleaned Membrane (Objects smaller than 3 pixels)")

        plt.show()

    return cleaned_membrane

def prep_for_membrane_analysis(image_path, vis=False):
    """
    Prepares the membrane channel for analysis by thresholding and cleaning.

    Args:
        image_path: Path to the image file.
        vis (bool): Whether to visualize the results.

    Returns:
        tuple: Membrane channel and cleaned membrane mask.
    """
    img = imread(image_path)
    channel_2 = img[1, :, :]
    binary_membrane = channel_2 > threshold_otsu(channel_2)
    final_mask = (binary_membrane * 255).astype(np.uint8) > 0
    cleaned_membrane = remove_small_objects(final_mask, min_size=51)
    if vis:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(channel_2, cmap="gray")
        plt.title("Original Membrane Channel")
        plt.subplot(1, 2, 2)
        plt.imshow(cleaned_membrane, cmap="gray")
        plt.title("Binary Mask")
        plt.show()
    return channel_2, cleaned_membrane

def analyze_membrane_continuity(image, mask):
    """
    Computes various metrics to assess membrane continuity.

    Args:
        image: The original membrane channel image.
        mask: The binary mask of the cleaned membrane.

    Returns:
        pandas.DataFrame: DataFrame containing the computed metrics.
    """
    metrics = {}
    skeleton = skeletonize(mask)
    metrics["Brightness_Std_Dev"] = np.std(image[skeleton])
    metrics["Mean_Gradient"] = np.mean(sobel(image)[mask])
    metrics["Number_of_Holes"] = np.max(label(~mask))
    metrics["Jaggedness"] = perimeter(mask) / np.sum(mask) if np.sum(mask) > 0 else 0
    props = regionprops(mask.astype(int))
    metrics["Solidity"] = props[0].solidity if props else 0
    metrics["Number_of_Fragments"] = label(mask, return_num=True)[1]
    metrics["Thickness_Variation"] = np.std(distance_transform_edt(mask))
    return pd.DataFrame([metrics])

def process_images_in_subfolders(root_folder):
    """
    Processes all images in subfolders and returns a DataFrame with results.

    Args:
        root_folder: Path to the root folder containing images.

    Returns:
        pandas.DataFrame: DataFrame with results.
    """
    all_results = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):
                image_path = os.path.join(subdir, file)
                print(f"Processing: {image_path}")
                results = process_image(image_path)
                if results:
                    summary_df_ch1, summary_df_ch2, specific_df = results
                    merged_df = pd.merge(summary_df_ch1, summary_df_ch2, left_index=True, right_index=True, suffixes=('_nuclei', '_membrane'))
                    merged_df = pd.concat([merged_df, specific_df], axis=1)
                    merged_df['image_name'] = file
                    all_results.append(merged_df)
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return None

# root_folder = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples"
# results_df = process_images_in_subfolders(root_folder)
# if results_df is not None:
#     results_df.to_csv(f"/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/{datetime.now().strftime('%Y-%m-%d-%H-%M')}_image_analysis_results.csv", index=False)
# else:
#     print("No images found or processed successfully.")