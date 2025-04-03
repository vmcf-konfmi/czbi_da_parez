#!/usr/bin/env python
# coding: utf-8

# ## Connect Google Drive

# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


# prompt: append path with code in quality.py and import all functions

import sys
sys.path.append('/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Python_code/code') # Replace with the actual path
# Import all functions from quality.py
from quality import *


# # Quality Check

# In[ ]:


# Mount Google Drive
# drive.mount('/content/gdrive')

# Example usage (assuming you've mounted your Google Drive):
# Replace '/content/gdrive/My Drive' with the actual path to your directory
print_directory_tree('/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images')

# # Initialize an empty list to store image information
# image_data = []

# # Replace with the actual path to your image directory
# image_dir = '/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images'

# # Walk through the directory and collect image information
# for root, _, files in os.walk(image_dir):
#     for file in files:
#         if file.lower().endswith(('.tiff', '.tif')): # Add more extensions if needed
#             image_path = os.path.join(root, file)
#             image_name = file
#             image_size, num_channels, pixel_size = image_info(image_path)
#             if image_size and num_channels > 1:  # Check if image_info returned valid data
#                 image_data.append([image_path, image_name, image_size, num_channels, pixel_size])


# # Create a Pandas DataFrame
# df = pd.DataFrame(image_data, columns=['Image Path', 'Image Name', 'Image Size', 'Number of Channels', 'Pixel Size (nm)'])

# # Print the DataFrame
# df
# df.to_csv("/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/quality_check.csv")

# Example usage (assuming your DataFrame is named 'df'):
# Replace 'Image Path' with the actual column name if different.
# Assuming df is already created from the previous code block
df = pd.read_csv("/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/quality_check.csv")
check_image_paths(df, 'Image Path')


# In[ ]:


x


# ## Details

# In[ ]:


# prompt: explore subfodlers and print out paht tree

import os

def print_directory_tree(root_dir, indent=0):
    """Prints a directory tree starting from the given root directory."""

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        print("  " * indent + item)  # Print with indentation
        if os.path.isdir(item_path):
            print_directory_tree(item_path, indent + 1)


# Example usage (assuming you've mounted your Google Drive):
# Replace '/content/gdrive/My Drive' with the actual path to your directory
print_directory_tree('/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images')


# In[ ]:


# prompt: for each image get image size, number of channels and pixel size in nanometers. Use from skimage.io import imread. Write everything in pandas table with image path and image name

from google.colab import drive
import os
from skimage.io import imread
import pandas as pd

# Mount Google Drive
# drive.mount('/content/gdrive')

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


# Initialize an empty list to store image information
image_data = []

# Replace with the actual path to your image directory
image_dir = '/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/Images'

# Walk through the directory and collect image information
for root, _, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith(('.tiff', '.tif')): # Add more extensions if needed
            image_path = os.path.join(root, file)
            image_name = file
            image_size, num_channels, pixel_size = image_info(image_path)
            if image_size and num_channels > 1:  # Check if image_info returned valid data
                image_data.append([image_path, image_name, image_size, num_channels, pixel_size])


# Create a Pandas DataFrame
df = pd.DataFrame(image_data, columns=['Image Path', 'Image Name', 'Image Size', 'Number of Channels', 'Pixel Size (nm)'])

# Print the DataFrame
df


# In[ ]:


df.to_csv("/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/quality_check.csv")


# In[ ]:


# prompt: write a function that checks all the images in Image Path column if they exists. Print any missing.

import os
import pandas as pd
from skimage.io import imread

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


# Example usage (assuming your DataFrame is named 'df'):
# Replace 'Image Path' with the actual column name if different.
# Assuming df is already created from the previous code block
df = pd.read_csv("/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/quality_check.csv")
check_image_paths(df, 'Image Path')


# In[ ]:


# quality.py

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



