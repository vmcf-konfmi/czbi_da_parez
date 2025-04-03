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
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_objects, skeletonize
from skimage.exposure import rescale_intensity
from stardist.models import StarDist2D
from scipy.ndimage import distance_transform_edt

def convert_ipynb_to_py(ipynb_file, py_file):
    """
    Convert a Jupyter Notebook (.ipynb) file to a Python (.py) script.

    Parameters:
    ipynb_file (str): The path to the input .ipynb file.
    py_file (str): The path to the output .py file.
    """
    import nbformat
    from nbconvert import PythonExporter

    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook_content)

    with open(py_file, 'w', encoding='utf-8') as f:
        f.write(python_code)

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

def detect_boundary(image_path):
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

def analyze_bright_spots(image_path):
    """
    Analyzes bright spots in the membrane channel of the image.

    Args:
        image_path: Path to the image file.
    """
    img = imread(image_path)
    channel_2 = img[1, :, :]
    binary_membrane = channel_2 > threshold_otsu(channel_2)
    binary_membrane_uint8 = (binary_membrane * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.erode(cv2.dilate(binary_membrane_uint8, kernel, iterations=3), kernel, iterations=3) > 0
    cleaned_membrane = remove_small_objects(final_mask, min_size=3)
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