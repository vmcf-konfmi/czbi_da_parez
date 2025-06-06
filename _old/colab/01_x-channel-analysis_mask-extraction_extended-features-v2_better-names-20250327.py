#!/usr/bin/env python
# coding: utf-8


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

# In[2]:


from skimage.io import imread
from skimage import filters
from skimage import measure
# from pyclesperanto_prototype import imshow
import pandas as pd
import numpy as np


# In[3]:


# @title
import os


def image_info(image_path):
    try:
        img = imread(image_path)
        image_size = img.shape[:2]  # Get height and width
        num_channels = img.shape[0]
        # Placeholder for pixel size, replace with actual calculation if available
        #pixel_size_nm = "Unknown" # Replace with actual pixel size in nm
        return image_size, num_channels#, pixel_size_nm
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None


# In[4]:


# @title
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from stardist.models import StarDist2D
from skimage.io import imsave
from skimage.measure import regionprops
import pandas as pd
from skimage.measure import label, regionprops

def measure_brightness_variability(intensity_image, mask):
    """Measures the variability of brightness values under a mask.

    Args:
        intensity_image: The image containing brightness values.
        mask: A binary mask where 1 indicates the region of interest.

    Returns:
        A dictionary containing brightness variability metrics (e.g., standard deviation).
        Returns None if the mask is invalid or empty.
    """

    masked_intensities = intensity_image[mask > 0]

    if len(masked_intensities) == 0:
        return None  # Handle empty mask case

    std_dev = np.std(masked_intensities)
    # Add other metrics as needed (e.g., mean, variance)
    return {'std_dev': std_dev}

def count_nuclei_under_membrane(nuclei_labels, membrane_mask):
    """Counts the number of nuclei under the membrane."""

    # Find nuclei that overlap with the membrane
    nuclei_under_membrane = np.unique(nuclei_labels[membrane_mask > 0])
    # Remove background label (usually 0) if present
    nuclei_under_membrane = nuclei_under_membrane[nuclei_under_membrane != 0]
    return len(nuclei_under_membrane)

def percentage_nuclei_pixels(nuclei_labels, membrane_mask):
    """Calculates the percentage of nuclei pixels within the membrane."""

    # Total number of nuclei pixels under the membrane
    total_nuclei_pixels_under_membrane = np.sum(membrane_mask[nuclei_labels > 0])

    # Total number of pixels in nuclei
    total_nuclei_pixels = np.sum(nuclei_labels > 0)

    if total_nuclei_pixels == 0:
        return 0  # Avoid division by zero if there are no nuclei pixels
    else:
        percentage = (total_nuclei_pixels_under_membrane / total_nuclei_pixels) * 100
        return percentage

def measure_membrane_pixel_count(label_image):
    """Measures the total pixel count of the membrane regions.

    Args:
        label_image: Labeled image of membranes.

    Returns:
        The total pixel count of membrane regions. Returns 0 if no membrane regions are found.
    """

    # Create a binary mask from the labeled image
    membrane_mask = label_image > 0

    # Calculate total pixel count of membrane mask
    membrane_pixel_count = np.sum(membrane_mask)
    return membrane_pixel_count

def measure_membrane_thickness(label_image, intensity_image):
    """
    Measures membrane thickness statistics (median, min, max).

    Args:
        label_image: Labeled image of membranes.
        intensity_image: The original intensity image.

    Returns:
        A dictionary containing median, minimum, and maximum thickness values.
        Returns None if no valid regions are found.
    """

    regions = regionprops(label_image, intensity_image=intensity_image)
    thickness_values = []

    for region in regions:
        # Calculate the thickness for each membrane region.
        # Replace this with your actual thickness calculation based on the intensity image and labels
        # Example: Assuming 'equivalent_diameter' is a reasonable proxy for thickness
        thickness = region.equivalent_diameter
        thickness_values.append(thickness)

    if thickness_values:
        median_thickness = np.median(thickness_values)
        min_thickness = np.min(thickness_values)
        max_thickness = np.max(thickness_values)
        return {'median_thickness': median_thickness, 'min_thickness': min_thickness, 'max_thickness': max_thickness}
    else:
        return None

def save_label_mask(image_path, labels, output_folder):
    """Saves the label image and mask as TIFF files in the specified folder.

    Args:
        image_path (str): Path to the input image.
        labels (numpy.ndarray): Label image data.
        output_folder (str): Path to the output folder.
    """
    try:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(output_folder, f"{image_name}_label.tiff")
        mask_path = os.path.join(output_folder, f"{image_name}_mask.tiff")

        imsave(label_path, labels.astype(np.uint16))  # Save labels as uint16
        imsave(mask_path, (labels > 0).astype(np.uint8))  # Save mask as binary image (uint8)

        print(f"Label image saved to: {label_path}")
        print(f"Mask image saved to: {mask_path}")
    except Exception as e:
        print(f"Error saving images: {e}")

def create_mask_from_labels(labels):
  """Creates a binary mask from a label image.

  Args:
    labels: A 2D numpy array representing the label image.

  Returns:
    A 2D numpy array representing the binary mask.
  """
  mask = np.where(labels > 0, 1, 0)
  return mask

def image_th_ch1(image):
    try:
      # # denoising
      # blurred_image = filters.gaussian(image, sigma=2)

      # # binarization
      # threshold = filters.threshold_otsu(blurred_image)
      # thresholded_image = blurred_image >= threshold

      # # labeling
      # label_image = measure.label(thresholded_image)

      # prints a list of available models
      # StarDist2D.from_pretrained()

      # creates a pretrained model
      model = StarDist2D.from_pretrained('2D_versatile_fluo')

      model.predict_instances(normalize(image))

      label_image, _ = model.predict_instances(normalize(image))

      return label_image, None, create_mask_from_labels(label_image)
    except Exception as e:
        print(f"Error processing {image}: {e}")
        return None, None, None

def image_th_ch2(image):
    try:
      # denoising
      blurred_image = filters.gaussian(image, sigma=5)

      # binarization
      threshold = filters.threshold_otsu(blurred_image)
      thresholded_image = blurred_image >= threshold

      # labeling
      label_image = measure.label(thresholded_image)

      return label_image, blurred_image, thresholded_image
    except Exception as e:
        print(f"Error processing {image}: {e}")
        return None, None, None

# prompt: extract all features using region_props
def quartiles(regionmask, intensity):
    return np.percentile(intensity[regionmask], q=(5, 10, 25, 50, 75, 90, 95))

def measure_image(label_image, intensity_image, properties):
  feature_table = regionprops_table(label_image, intensity_image, properties, extra_properties=(quartiles,))

  # Convert to DataFrame
  features_df = pd.DataFrame(feature_table)
  return features_df

def summarize_features(df):
    """
    Calculates summary statistics for a DataFrame and returns a new DataFrame with one row.
    """

    summary_stats = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # Check if the column is numeric
            summary_stats[f"{col}_mean"] = df[col].mean()
            summary_stats[f"{col}_std"] = df[col].std()
            summary_stats[f"{col}_min"] = df[col].min()
            summary_stats[f"{col}_max"] = df[col].max()
            summary_stats[f"{col}_median"] = df[col].median()
            summary_stats[f"{col}_count"] = df[col].count()
            # Add more summary statistics if needed (e.g., percentiles)

        # Handle non-numeric columns (e.g., count unique values)
        else:
            summary_stats[f"{col}_unique_count"] = df[col].nunique()


    summary_df = pd.DataFrame([summary_stats])
    return summary_df

def process_image(image_path):
    """
    Processes an image and returns information about it.

    Args:
        image_path: The path to the image file.

    Returns: summary_df_ch1, summary_df_ch2 as statistical summary
    """
    try:
        image_size, num_channels = image_info(image_path)
        if image_size is None:
            return None

        image_size, num_channels = image_info(image_path)
        print(f"Image Size: {image_size}")
        print(f"Number of Channels: {num_channels}")
        if num_channels == 2:
          image_name = os.path.splitext(os.path.basename(image_path))[0]
          img = imread(image_path)

          #####################################################
          ## membrane
          channel_2 = img[1, :, :]
          # threshold
          label_image_ch2, blurred_image_ch2, thresholded_image_ch2 = image_th_ch2(channel_2)
          output_folder = os.path.dirname(image_path)
          save_label_mask(image_path, label_image_ch2, output_folder)

          # get general features of membrane
          #properties = ['label', 'area', 'perimeter', 'centroid', 'bbox', 'solidity', 'mean_intensity', 'major_axis_length', 'minor_axis_length']
          properties = ['area', 'perimeter', 'centroid', 'bbox', 'solidity', 'mean_intensity', 'major_axis_length', 'minor_axis_length']
          df_features_ch2 = measure_image(label_image_ch2, channel_2, properties=properties)

          #save features
          df_features_ch2.to_csv(os.path.join(output_folder, f"{image_name}_features_ch2.csv"), index=False)

          #summarize
          summary_df_ch2 = summarize_features(df_features_ch2)

          ####################################################
          ## nuclei
          channel_1 = img[0, :, :]
          # threshold
          label_image_ch1, blurred_image_ch1, thresholded_image_ch1 = image_th_ch1(channel_1)
          # output_folder = os.path.dirname(image_path)
          save_label_mask(image_path, label_image_ch1, output_folder)

          # get general features of nuclei
          #properties_nuclei = ['label', 'area', 'perimeter', 'solidity', 'max_intensity', 'mean_intensity', 'min_intensity', 'major_axis_length', 'minor_axis_length']
          properties_nuclei = ['area', 'perimeter', 'solidity', 'max_intensity', 'mean_intensity', 'min_intensity', 'major_axis_length', 'minor_axis_length']
          df_features_ch1 = measure_image(label_image_ch1, channel_1, properties=properties_nuclei)

          #save features
          df_features_ch1.to_csv(os.path.join(output_folder, f"{image_name}_features_ch1.csv"), index=False)

          #summarize
          summary_df_ch1 = summarize_features(df_features_ch1)

          # get specific membrane features
          specific_df = pd.DataFrame(index=[0]) # Initialize with a single row

          ## number of nuclei under membrane
          num_nuclei = count_nuclei_under_membrane(label_image_ch1, create_mask_from_labels(label_image_ch2))
          specific_df['nuclei_under_membrane'] = num_nuclei
          print(f"nuclei_under_membrane: {num_nuclei}")

          ## percentage of nuclei pixels
          percentage = percentage_nuclei_pixels(label_image_ch1, create_mask_from_labels(label_image_ch2))
          specific_df['percentage_nuclei_pixels'] = percentage
          print(f"percentage_nuclei_pixels: {percentage}")

          ## pixel count of membrane
          membrane_pixel_count = measure_membrane_pixel_count(label_image_ch2)
          specific_df['membrane_pixel_count'] = membrane_pixel_count
          print(f"Total membrane pixel count: {membrane_pixel_count}")


          ## thickness of membrane
          thickness_stats = measure_membrane_thickness(label_image_ch2, channel_2)

          if thickness_stats:
              specific_df['median_thickness'] = thickness_stats['median_thickness']
              print(f"median_thickness: {thickness_stats['median_thickness']}")
              specific_df['min_thickness'] = thickness_stats['min_thickness']
              specific_df['max_thickness'] = thickness_stats['max_thickness']
          else:
              print("Warning: No valid membrane regions found for thickness calculation.")

          ## number of holes in membrane

          ## pixel count of holes in membrane

          ## Measure brightness variability under the membrane mask
          brightness_variability = measure_brightness_variability(channel_2, create_mask_from_labels(label_image_ch2))

          if brightness_variability:
              specific_df['membrane_brightness_std_dev'] = brightness_variability['std_dev']
              print(f"Membrane brightness standard deviation: {brightness_variability['std_dev']}")
          else:
              print("Warning: Invalid or empty membrane mask for brightness variability calculation.")

          return summary_df_ch1, summary_df_ch2, specific_df
        else:
          print("Error: Number of channels is not 2.")
          # return None, None, None

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# ## Tests

# ### Thickness of membrane

# In[5]:


image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 1/Filaggrin/Healthy1_Filaggrin_1.tif"
img = imread(image_path)

#####################################################
## membrane
channel_2 = img[1, :, :]

#####################################################
## nuclei
channel_1 = img[0, :, :]


# In[6]:


# prompt: normalize and plot both channels

from skimage.exposure import rescale_intensity

def normalize(image):
    """Normalizes the image intensity to the range [0, 1]."""
    normalized_image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    return normalized_image

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


# Test contour detection

# In[7]:


# @title
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Example usage:
# detect_boundary("path_to_your_image.jpg")



# In[8]:


# @title
def detect_boundary(image_path):
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


# In[9]:


image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 3/Filaggrin/Healthy3_Filaggrin_1.tif"
detect_boundary(image_path)
image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 3/Involucrin/Healthy3_Involucrin_1.tif"
detect_boundary(image_path)
image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 3/Loricrin/Healthy3_Loricrin_1.tif"
detect_boundary(image_path)


# In[10]:


image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Patients/Px 1/Filaggrin/Px1_Filaggrin_1.tif"
detect_boundary(image_path)
image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Patients/Px 1/Involucrin/Px1_Involucrin_2.tif"
detect_boundary(image_path)
image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Patients/Px 1/Loricrin/Px1_Loricrin_1.tif"
detect_boundary(image_path)


# In[16]:


# test function
# !todo: change output of function

from google.colab import drive
import os
from watermark import watermark
from skimage.io import imread
from skimage import filters
from skimage import measure
import pandas as pd
import numpy as np
from skimage.measure import regionprops_table
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from stardist.models import StarDist2D
from skimage.io import imsave
from skimage.measure import regionprops
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
import cv2
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import label, remove_small_objects


def detect_boundary(image_path):
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


def remove_large_objects(binary_image, max_size):
    labeled_image = label(binary_image)
    sizes = np.bincount(labeled_image.ravel())  # Count sizes of objects
    mask = sizes <= max_size  # Keep only objects smaller than max_size
    mask[0] = 0  # Ensure background remains
    return mask[labeled_image]


# In[17]:


image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Patients/Px 1/Involucrin/Px1_Involucrin_2.tif"
detect_boundary(image_path)

image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 2/Involucrin/Healthy2_Involucrin_1.tif"
detect_boundary(image_path)


# ### Pattern/continuity

# ### Gradient from membrane

# In[11]:


image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 1/Filaggrin/Healthy1_Filaggrin_1.tif"
img = imread(image_path)

#####################################################
## membrane
channel_2 = img[1, :, :]

#####################################################
## nuclei
channel_1 = img[0, :, :]

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

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.imshow(expansion_region, cmap='gray')

    # Find separate connected regions within the expansion region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(expansion_region, connectivity=8)

    region_brightness = []
    for label in range(1, num_labels):  # Skip background (label 0)
        region_mask = (labels == label).astype(np.uint8) * 255
        brightness_values_list = channel_2[labels == label].flatten()

        percentiles = np.percentile(brightness_values_list, [25, 50, 75, 95])
        region_brightness.append((label, percentiles))

    brightness_values.append((i * step_size, region_brightness))
    start = False

# Show the results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_8bit, cmap='gray')
axes[0].set_title("Original Image (8-bit Scaled)")
axes[0].axis("off")

axes[1].imshow(result)
axes[1].set_title("Detected Boundary")
axes[1].axis("off")

plt.show()

# Print brightness percentiles for each expansion step
print("Expansion Step (pixels) - Brightness in Regions:")
for step, regions in brightness_values:
    print(f"Step {step} pixels:")
    for region_id, percentiles in regions:
        print(f"  Region {region_id}: 25th={percentiles[0]:.2f}, 50th={percentiles[1]:.2f}, 75th={percentiles[2]:.2f}, 95th={percentiles[3]:.2f}")

# Example usage:
# detect_boundary("path_to_your_image.tif")

# for i in range(1, max_steps + 1):
#     # Expand outward by dilation
#     kernel = np.ones((3, 3), np.uint8)
#     expanded_mask = cv2.dilate(mask, kernel, iterations=i * step_size // 3)

#     # Get the new region (difference between expanded mask and original mask)
#     if start:
#       expansion_region = cv2.bitwise_xor(expanded_mask, mask)
#       old_mask = mask
#     else:
#       expansion_region = cv2.bitwise_xor(expanded_mask, old_mask)
#       old_mask = expanded_mask

#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].imshow(expansion_region, cmap='gray')
#     axes[0].axis("off")

#     # Measure brightness in the expansion region
#     brightness = cv2.mean(channel_2, mask=expansion_region)[0]
#     brightness_values.append((i * step_size, brightness))
#     start = False

# # Show the results
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(image_8bit, cmap='gray')
# axes[0].set_title("Original Image (8-bit Scaled)")
# axes[0].axis("off")

# axes[1].imshow(result)
# axes[1].set_title("Detected Boundary")
# axes[1].axis("off")

# plt.show()

# # Print brightness values for each expansion step
# print("Expansion Step (pixels) - Brightness:")
# for step, brightness in brightness_values:
#     print(f"{step} - {brightness}")

# # Example usage:
# # detect_boundary("path_to_your_image.tif")


# ### Bright spots

# In[15]:


## Bright spots
# image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Patients/Px 1/Involucrin/Px1_Involucrin_2.tif"

import numpy as np
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import label, remove_small_objects


def remove_large_objects(binary_image, max_size):
    labeled_image = label(binary_image)
    sizes = np.bincount(labeled_image.ravel())  # Count sizes of objects
    mask = sizes <= max_size  # Keep only objects smaller than max_size
    mask[0] = 0  # Ensure background remains
    return mask[labeled_image]


image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Patients/Px 1/Involucrin/Px1_Involucrin_2.tif"
# image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 2/Involucrin/Healthy2_Involucrin_1.tif"

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

# Example: Display the results (if you have matplotlib installed)
import matplotlib.pyplot as plt

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


# In[19]:


# test function
# !todo: change output of function

import matplotlib.pyplot as plt
import numpy as np
def analyze_bright_spots(image_path):
  """
  Analyzes bright spots in a given image, specifically targeting the membrane channel.

  Args:
      image_path: Path to the multi-channel image file (e.g., a TIFF file).

  Returns:
      None: The function displays the results of the analysis, including:
          - The original membrane channel.
          - The binary membrane obtained after Otsu's thresholding.
          - The cleaned membrane after removing small and large objects.
  """

  img = imread(image_path)
  channel_2 = img[1, :, :]  # Membrane channel

  # Threshold the membrane channel using Otsu's method
  thresh = threshold_otsu(channel_2)
  binary_membrane = channel_2 > thresh

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

  def remove_large_objects(binary_image, max_size):
      labeled_image = label(binary_image)
      sizes = np.bincount(labeled_image.ravel())  # Count sizes of objects
      mask = sizes <= max_size  # Keep only objects smaller than max_size
      mask[0] = 0  # Ensure background remains
      return mask[labeled_image]

  # Remove large objects (connected components) larger than 81 pixels
  cleaned_membrane = remove_large_objects(
      final_mask, max_size=81
  )  # min_size is exclusive, so use 51 to remove objects > 50

  # Remove too small objects
  cleaned_membrane = remove_small_objects(cleaned_membrane, min_size=3)

  # Example: Display the results (if you have matplotlib installed)
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


# In[20]:


image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Patients/Px 1/Involucrin/Px1_Involucrin_2.tif"
analyze_bright_spots(image_path)
image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 2/Involucrin/Healthy2_Involucrin_1.tif"
analyze_bright_spots(image_path)


# ## Continuity of label

# ### What This Measures
# 
# | Metric             | Meaning                                    | Higher Value Indicates             |
# |--------------------|--------------------------------------------|------------------------------------|
# | **Brightness Std Dev** | Variation in intensity along the membrane | Discontinuous brightness           |
# | **Mean Gradient**     | Sharpness of membrane edges               | More abrupt changes               |
# | **Number of Holes**   | Internal gaps in membrane                  | More fragmented membrane          |
# | **Jaggedness**        | Perimeter-to-area ratio                   | Irregular edges                   |
# | **Solidity**          | Ratio of filled area to convex hull        | Fragmented membrane if lower      |
# | **Number of Fragments** | Disconnected membrane parts              | More broken structure             |
# | **Thickness Variation** | Std dev of distance transform            | Uneven membrane width             |

# In[28]:


import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters import sobel, threshold_otsu
from skimage.measure import perimeter, regionprops
from skimage.morphology import label, skeletonize

def compute_brightness_continuity(image, mask):
    """
    Measures the standard deviation of pixel intensity along the skeletonized mask.

    Args:
        image (ndarray): Grayscale image (membrane channel).
        mask (ndarray): Binary mask of the membrane.

    Returns:
        float: Standard deviation of brightness along the membrane skeleton.
    """
    skeleton = skeletonize(mask)
    intensities = image[skeleton]
    return np.std(intensities)


def compute_gradient(image, mask):
    """
    Computes the mean gradient magnitude within the mask.

    Args:
        image (ndarray): Grayscale image (membrane channel).
        mask (ndarray): Binary mask of the membrane.

    Returns:
        float: Mean gradient magnitude.
    """
    gradient = sobel(image)
    return np.mean(gradient[mask])


def count_holes(mask):
    """
    Counts the number of holes inside the membrane.

    Args:
        mask (ndarray): Binary mask of the membrane.

    Returns:
        int: Number of holes.
    """
    # label() returns only the labeled array when called with default arguments.
    labeled_array = label(~mask)
    # To get the number of labels (objects), find the maximum label value.
    num_holes = np.max(labeled_array)
    return num_holes


def compute_jaggedness(mask):
    """
    Computes the perimeter-to-area ratio as a measure of jaggedness.

    Args:
        mask (ndarray): Binary mask of the membrane.

    Returns:
        float: Jaggedness metric (higher = more irregular).
    """
    area = np.sum(mask)
    perim = perimeter(mask)
    return perim / area if area > 0 else 0


def compute_solidity(mask):
    """
    Computes solidity (compactness) of the membrane.

    Args:
        mask (ndarray): Binary mask of the membrane.

    Returns:
        float: Solidity (1 = perfectly filled shape, <1 = fragmented).
    """
    props = regionprops(mask.astype(int))
    return props[0].solidity if props else 0


def count_fragments(mask):
    """
    Counts the number of separate membrane fragments.

    Args:
        mask (ndarray): Binary mask of the membrane.

    Returns:
        int: Number of connected components.
    """
    _, num_fragments = label(mask, return_num=True)
    return num_fragments


def compute_thickness_variation(mask):
    """
    Computes the standard deviation of membrane thickness.

    Args:
        mask (ndarray): Binary mask of the membrane.

    Returns:
        float: Standard deviation of membrane thickness.
    """
    thickness = distance_transform_edt(mask)
    return np.std(thickness)


# In[43]:


def prep_for_membrane_analysis(image_path, vis: bool = False):

  img = imread(image_path)
  channel_2 = img[1, :, :]  # Membrane channel

  # Threshold the membrane channel using Otsu's method
  thresh = threshold_otsu(channel_2)
  binary_membrane = channel_2 > thresh

  # Convert binary mask to uint8 for OpenCV
  binary_membrane_uint8 = (binary_membrane * 255).astype(np.uint8)

  # Convert back to boolean if needed
  final_mask = binary_membrane_uint8 > 0

  # Remove too small objects
  cleaned_membrane = remove_small_objects(final_mask, min_size=51)

  if vis:
    # Example: Display the results (if you have matplotlib installed)
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
      image (ndarray): The original membrane channel image.
      mask (ndarray): The binary mask of the cleaned membrane.

  Returns:
      dict: A dictionary containing the computed metrics.
  """

  metrics = {}
  metrics["Brightness_Std_Dev"] = compute_brightness_continuity(image, mask)
  metrics["Mean_Gradient"] = compute_gradient(image, mask)
  metrics["Number_of_Holes"] = count_holes(mask)
  metrics["Jaggedness"] = compute_jaggedness(mask)
  metrics["Solidity"] = compute_solidity(mask)
  metrics["Number_of_Fragments"] = count_fragments(mask)
  metrics["Thickness_Variation"] = compute_thickness_variation(mask)

  return pd.DataFrame([metrics])



# In[44]:


# example

# Example usage:
image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Patients/Px 1/Involucrin/Px1_Involucrin_2.tif"
# image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 2/Involucrin/Healthy2_Involucrin_1.tif"

channel_2, cleaned_membrane = prep_for_membrane_analysis(image_path,vis=True)

image, mask = prep_for_membrane_analysis(image_path)
metrics = analyze_membrane_continuity(channel_2, cleaned_membrane)
metrics



# ## Code

# In[ ]:


from stardist.models import StarDist2D

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')


# In[ ]:


from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

img = channel_1

labels, _ = model.predict_instances(normalize(img))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")


# In[ ]:


image_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples/Healthy Individuals/Healthy 1/Filaggrin/Healthy1_Filaggrin_1.tif"
summary_df_ch1, summary_df_ch2, specific_df = process_image(image_path)

merged_df = pd.merge(summary_df_ch1, summary_df_ch2, left_index=True, right_index=True, suffixes=('_nuclei', '_membrane'))
merged_df = pd.concat([merged_df, specific_df], axis=1)
merged_df


# In[ ]:


# process all images in subfolders keep image names in output table
get_ipython().run_line_magic('timeit', '')
from datetime import datetime



def process_images_in_subfolders(root_folder):
    all_results = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff')):  # Add more extensions if needed
                image_path = os.path.join(subdir, file)
                print("------------------------------------")
                print("------------------------------------")
                print(f"Processing: {image_path}")  # Print the current image being processed
                print("------------------------------------")
                results = process_image(image_path)
                if results:
                    summary_df_ch1, summary_df_ch2, specific_df = results
                    merged_df = pd.merge(summary_df_ch1, summary_df_ch2, left_index=True, right_index=True, suffixes=('_nuclei', '_membrane')) #, suffixes=('_channel1', '_channel2'))
                    merged_df = pd.concat([merged_df, specific_df], axis=1)
                    merged_df['image_name'] = file # Add image name column
                    all_results.append(merged_df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        return final_df
    else:
        return None

# Example usage:
root_folder = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images/Skin samples"  # Replace with your root folder
results_df = process_images_in_subfolders(root_folder)

if results_df is not None:
    #print(results_df)
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d-%H-%M")

    # Save the results to a CSV file
    results_df.to_csv(f"/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/{date_time_string}_image_analysis_results_new-features_new-names.csv", index=False)
else:
    print("No images found or processed successfully.")


# In[ ]:




