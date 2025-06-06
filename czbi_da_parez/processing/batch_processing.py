"""
batch_processing.py
===================
Batch and high-level image processing utilities.

Functions for processing images, subfolders, and formatting data for analysis.
"""

import os
import pandas as pd
from skimage.io import imread
from .image_channel_analysis import image_info, image_th_ch1, image_th_ch2
from .membrane_analysis import save_label_mask, create_mask_from_labels, detect_boundary, detect_boundary_wider, gradient_analysis, analyze_bright_spots, prep_for_membrane_analysis, analyze_membrane_continuity
from .feature_summary import measure_image, summarize_features, measure_membrane_pixel_count, measure_membrane_thickness
from .quality import count_nuclei_under_membrane, percentage_nuclei_pixels, measure_brightness_variability
from skimage.measure import label

def process_imag_old(image_path):
    """
    Processes an image and returns information about it (legacy version).

    Args:
        image_path: The path to the image file.

    Returns:
        summary_df_ch1, summary_df_ch2, specific_df as statistical summary
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

def process_image(image_path, verbose=False):
    """
    Processes an image and returns information about it.

    Args:
        image_path: The path to the image file.
        verbose (bool): Whether to print debug information.

    Returns:
        summary_df_ch1, summary_df_ch2, specific_df as statistical summary
    """
    morph_properties_to_measure = [
        'area', 'perimeter', 'centroid', 'solidity', 'major_axis_length',
        'minor_axis_length', 'eccentricity', 'solidity', 'extent',
        'mean_intensity', 'perimeter', 'equivalent_diameter', 'max_intensity',
        'mean_intensity', 'min_intensity', 'weighted_moments',
    ]
    try:
        image_size, num_channels = image_info(image_path)
        if image_size is None or num_channels != 2:
            print(f"Invalid image: {image_path}")
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
        summary_df_ch2 = summary_df_ch2.add_prefix("membrane_")
        if verbose:
            print("summary_df_ch2")
            print(summary_df_ch2.columns)
        gradient_arr = gradient_analysis(image_path, verbose=False, vis=False)
        df_gradient_features_ch2 = format_data_to_dataframe(gradient_arr)
        df_gradient_features_ch2 = df_gradient_features_ch2.add_prefix("membrane_gradient_")
        if verbose:
            print("df_gradient_features_ch2")
            print(df_gradient_features_ch2.columns)
        bright_spots_mask = analyze_bright_spots(image_path, vis=False)
        df_bright_spots_features_ch2 = measure_image(label(bright_spots_mask), channel_2, properties=morph_properties_to_measure)
        df_bright_spots_features_ch2_sum = summarize_features(df_bright_spots_features_ch2)
        df_bright_spots_features_ch2_sum = df_bright_spots_features_ch2_sum.add_prefix("membrane_bright_spots_")
        if verbose:
            print("df_bright_spots_features_ch2_sum")
            print(df_bright_spots_features_ch2_sum.columns)
        boundary_mask = detect_boundary(image_path, vis=False)
        df_boundary_features_ch2 = measure_image(label(boundary_mask), channel_2, properties=morph_properties_to_measure)
        df_boundary_features_ch2_sum = summarize_features(df_boundary_features_ch2)
        df_boundary_features_ch2_sum = df_boundary_features_ch2_sum.add_prefix("membrane_boundary_")
        if verbose:
            print("df_boundary_features_ch2_sum")
            print(df_boundary_features_ch2_sum.columns)
        cleaned_channel_2, cleaned_membrane = prep_for_membrane_analysis(image_path, vis=False)
        membrane_continuity_metrics = analyze_membrane_continuity(cleaned_channel_2, cleaned_membrane)
        membrane_continuity_metrics = membrane_continuity_metrics.add_prefix("membrane_continuity_")
        if verbose:
            print("membrane_continuity_metrics")
            print(membrane_continuity_metrics.columns)
        channel_1 = img[0, :, :]
        label_image_ch1, _, _ = image_th_ch1(channel_1)
        save_label_mask(image_path, label_image_ch1, output_folder)
        df_features_ch1 = measure_image(label_image_ch1, channel_1, properties=['area', 'perimeter', 'solidity', 'max_intensity', 'mean_intensity', 'min_intensity', 'major_axis_length', 'minor_axis_length'])
        df_features_ch1.to_csv(os.path.join(output_folder, f"{image_name}_features_ch1.csv"), index=False)
        summary_df_ch1 = summarize_features(df_features_ch1)
        summary_df_ch1 = summary_df_ch1.add_prefix("nuclei_")
        if verbose:
            print("summary_df_ch1")
            print(summary_df_ch1.columns)
        specific_df = pd.DataFrame(index=[0])
        specific_df['nuclei_under_membrane'] = count_nuclei_under_membrane(label_image_ch1, create_mask_from_labels(label_image_ch2))
        specific_df['nuclei_percentage_pixels'] = percentage_nuclei_pixels(label_image_ch1, create_mask_from_labels(label_image_ch2))
        specific_df['nuclei_membrane_pixel_count'] = measure_membrane_pixel_count(label_image_ch2)
        thickness_stats = measure_membrane_thickness(label_image_ch2, channel_2)
        if thickness_stats:
            thickness_df = pd.DataFrame(thickness_stats, index=[0])
            specific_df = pd.concat([specific_df, thickness_df], axis=1)
        if verbose:
            print("specific_df")
            print(specific_df.columns)
        thickness_df = thickness_df.add_prefix("membrane_thickness_")
        if verbose:
            print("thickness_df")
            print(thickness_df.columns)
        brightness_variability = measure_brightness_variability(channel_2, create_mask_from_labels(label_image_ch2))
        if verbose:
            print("brightness_variability")
            print(brightness_variability)
        if brightness_variability:
            specific_df['membrane_brightness_std_dev'] = brightness_variability['std_dev']
        summary_df_ch2 = pd.concat([summary_df_ch2, membrane_continuity_metrics, df_gradient_features_ch2, df_bright_spots_features_ch2_sum, df_boundary_features_ch2_sum, thickness_df], axis=1)
        return summary_df_ch1, summary_df_ch2, specific_df
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

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

def format_data_to_dataframe(data):
    """
    Converts a specific data structure (list of tuples with nested region data and numpy arrays)
    into a pandas DataFrame with column headers representing the step, region, and percentile,
    and numerical values all in a single row.

    Args:
        data: A list of tuples. Each tuple contains:
            - An integer representing the step (e.g., 60, 120).
            - A list of tuples. Each inner tuple contains:
                - An integer representing the region ID (e.g., 1, 2).
                - A numpy array of four floats representing the 25th, 50th, 75th, and 95th percentiles.

    Returns:
        pandas.DataFrame: A DataFrame with columns like 'Step_X_Region_Y_25th',
        'Step_X_Region_Y_50th', etc., and numerical values in a single row.
    """
    all_data = {}
    percentile_names = ["25th", "50th", "75th", "95th"]
    for step, regions_data in data:
        for region, percentiles in regions_data:
            for i, percentile_value in enumerate(percentiles):
                column_name = f"Step_{step}_Region_{region}_{percentile_names[i]}"
                all_data[column_name] = percentile_value
    df = pd.DataFrame([all_data])
    return df
