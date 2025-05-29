"""
membrane_analysis.py
====================
Membrane mask, boundary, and continuity analysis utilities.

Functions for mask creation, boundary detection, gradient analysis, and membrane continuity.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import remove_small_objects, skeletonize
from scipy.ndimage import distance_transform_edt

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
    mask = np.zeros_like(thresh, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    return mask

def detect_boundary_wider(image_path, vis=False):
    """
    Detects the boundary of the membrane in the image (wider version).

    Args:
        image_path: Path to the image file.
    """
    img = imread(image_path)
    channel_2 = img[1, :, :]
    image_8bit = cv2.convertScaleAbs(channel_2, alpha=(255.0/65535.0))
    median_filtered = cv2.medianBlur(image_8bit, 15)
    blurred = cv2.GaussianBlur(median_filtered, (13, 13), 0)
    for _ in range(6):
        blurred = cv2.GaussianBlur(blurred, (13, 13), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    result = image_rgb.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
    if vis:
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
    mask = np.zeros_like(thresh, dtype=np.uint8)
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
    channel_2 = img[1, :, :]
    image_8bit = cv2.convertScaleAbs(channel_2, alpha=(255.0/65535.0))
    median_filtered = cv2.medianBlur(image_8bit, 15)
    blurred = cv2.GaussianBlur(median_filtered, (13, 13), 0)
    for _ in range(6):
        blurred = cv2.GaussianBlur(blurred, (13, 13), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    result = image_rgb.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    step_size = 60
    max_steps = 10
    brightness_values = []
    start = True
    for i in range(1, max_steps + 1):
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=i * step_size // 3)
        if start:
            expansion_region = cv2.bitwise_xor(expanded_mask, mask)
            old_mask = expanded_mask
        else:
            expansion_region = cv2.bitwise_xor(expanded_mask, old_mask)
            old_mask = expanded_mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(expansion_region, connectivity=8)
        region_brightness = []
        for label_id in range(1, num_labels):
            brightness_values_list = channel_2[labels == label_id].flatten()
            percentiles = np.percentile(brightness_values_list, [25, 50, 75, 95])
            region_brightness.append((label_id, percentiles))
        brightness_values.append((i * step_size, region_brightness))
        start = False
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
        print("Expansion Step (pixels) - Brightness in Regions:")
        for step, regions in brightness_values:
            print(f"Step {step} pixels:")
            for region_id, percentiles in regions:
                print(f"  Region {region_id}: 25th={percentiles[0]:.2f}, 50th={percentiles[1]:.2f}, 75th={percentiles[2]:.2f}, 95th={percentiles[3]:.2f}")
    return brightness_values

def remove_large_objects(binary_image, max_size):
    labeled_image = label(binary_image)
    sizes = np.bincount(labeled_image.ravel())
    mask = sizes <= max_size
    mask[0] = 0
    return mask[labeled_image]

def analyze_bright_spots(image_path, verbose=False, vis=False):
    """
    Analyzes bright spots in the membrane channel of the image.

    Args:
        image_path: Path to the image file.
    """
    img = imread(image_path)
    channel_2 = img[1, :, :]
    thresh = threshold_otsu(channel_2)
    binary_membrane = channel_2 > thresh
    kernel = np.ones((3, 3), np.uint8)
    # Convert boolean mask to uint8 for OpenCV compatibility
    binary_membrane_uint8 = binary_membrane.astype(np.uint8)
    expanded_mask = cv2.dilate(binary_membrane_uint8, kernel)
    expanded_mask = cv2.dilate(expanded_mask, kernel)
    expanded_mask = cv2.dilate(expanded_mask, kernel)
    eroded_mask = cv2.erode(expanded_mask, kernel)
    eroded_mask = cv2.erode(eroded_mask, kernel)
    eroded_mask = cv2.erode(eroded_mask, kernel)
    final_mask = eroded_mask > 0
    cleaned_membrane = remove_large_objects(final_mask, max_size=81)
    cleaned_membrane = remove_small_objects(cleaned_membrane, min_size=3)
    labeled_membrane = label(cleaned_membrane)
    num_objects = np.max(labeled_membrane)
    if verbose:
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
