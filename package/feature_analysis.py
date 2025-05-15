#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)

# Assign type, image ID, and source based on image name
def assign_columns(df):
    """
    Assign type, image ID, source, and person ID based on the 'image_name' column.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new columns added.
    """
    df["type"] = df["image_name"].apply(assign_type)
    df["image-ID"] = df["image_name"].apply(assign_imageID)
    df["source"] = df["image_name"].apply(assign_source)
    df['person-ID'] = df['image_name'].str.split('_').str[0]
    return df

def assign_type(image_name):
    """
    Determine the type of the image based on its name.

    Args:
        image_name (str): Name of the image.

    Returns:
        str: Type of the image (e.g., 'Filaggrin', 'Loricrin', 'Involucrin', or 'Other').
    """
    if "Filaggrin" in image_name:
        return "Filaggrin"
    elif "Loricrin" in image_name:
        return "Loricrin"
    elif "Involucrin" in image_name:
        return "Involucrin"
    else:
        return "Other"

def assign_imageID(image_name):
    """
    Extract the image ID from the image name.

    Args:
        image_name (str): Name of the image.

    Returns:
        str: Image ID (e.g., '1', '2', '3', or 'Other').
    """
    if "1" in image_name:
        return "1"
    elif "2" in image_name:
        return "2"
    elif "3" in image_name:
        return "3"
    else:
        return "Other"

def assign_source(image_name):
    """
    Determine the source of the image based on its name.

    Args:
        image_name (str): Name of the image.

    Returns:
        str: Source of the image (e.g., 'Healthy', 'Patient', or 'Other').
    """
    if "Healthy" in image_name:
        return "Healthy"
    elif "P" in image_name:
        return "Patient"
    else:
        return "Other"

# Group and pivot data
def group_and_pivot(df):
    """
    Group data by 'person-ID' and 'type', calculate the mean of numerical columns,
    and pivot the table to create a wide-format DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Pivoted DataFrame with aggregated values.
    """
    numerical_cols = df.select_dtypes(include=np.number).columns
    mean_df = df.groupby(['person-ID', 'type'])[numerical_cols].mean().reset_index()
    pivot_df = mean_df.pivot(index='person-ID', columns='type', values=numerical_cols)
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df.reset_index()
    pivot_df.insert(1, 'source', df.groupby('person-ID')['source'].first().values)
    return pivot_df

# Drop specific columns
def drop_columns(df, substrings):
    """
    Drop columns from the DataFrame that contain specific substrings.

    Args:
        df (pd.DataFrame): Input DataFrame.
        substrings (list): List of substrings to check for in column names.

    Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
    """

    return df.loc[:, ~df.columns.str.contains('nuclei')]

# Correlation analysis
def correlation_analysis(df):
    """
    Perform correlation analysis on the DataFrame and select the most unique features
    based on the variance of the correlation matrix.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of the most unique features (column names).
    """
    correlation_matrix = df.iloc[:, 2:].corr()
    variances = correlation_matrix.var()
    return variances.nlargest(505).index

# t-SNE clustering
def tsne_clustering(df, features, path_4_png, input_type, n_components=2, random_state=42, perplexity=30, n_iter=1000):
    """
    Perform t-SNE clustering on the selected features and plot the results.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of features to use for clustering.
        path_4_png (str): Path to save the plot.
        input_type (str): Input type for the plot title.
        n_components (int): Number of dimensions for t-SNE (default: 2).
        random_state (int): Random seed for reproducibility (default: 42).
        perplexity (int): Perplexity parameter for t-SNE (default: 30).
        n_iter (int): Number of iterations for optimization (default: 1000).

    Returns:
        np.ndarray: t-SNE embedding of the data.
    """
    X = df[features]
    source = df['source']
    tsne = TSNE(n_components, random_state, perplexity, n_iter)
    X_embedded = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    for s in source.unique():
        plt.scatter(X_embedded[source == s, 0], X_embedded[source == s, 1], label=s)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Clustering based on Source")
    plt.legend()
    plt.savefig(f'{path_4_png}/Tsne_projection_{input_type}_source.png' if path_4_png else f'Tsne_projection_{input_type}_source.png')
    plt.show()
    return X_embedded

# UMAP clustering
def umap_clustering(df, features, path_4_png, input_type, n_neighbors=35, min_dist=0.15, random_state=42):
    """
    Perform UMAP clustering on the selected features and plot the results.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of features to use for clustering.
        path_4_png (str): Path to save the plot.
        input_type (str): Input type for the plot title.
        n_neighbors (int): Number of neighbors for UMAP (default: 35).
        min_dist (float): Minimum distance parameter for UMAP (default: 0.15).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        np.ndarray: UMAP embedding of the data.
    """
    reducer = umap.UMAP(n_neighbors, min_dist, random_state)
    embedding = reducer.fit_transform(df[features].values)
    df['umap_x'] = embedding[:, 0]
    df['umap_y'] = embedding[:, 1]
    plt.figure(figsize=(8, 6))
    for source in df['source'].unique():
        subset = df[df['source'] == source]
        plt.scatter(subset['umap_x'], subset['umap_y'], label=source, alpha=0.7)
    plt.xlabel('UMAP X')
    plt.ylabel('UMAP Y')
    plt.title(f'UMAP Projection of {input_type} Data Colored by Source')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path_4_png}/umap_projection_{input_type}_source.png' if path_4_png else f'umap_projection_{input_type}_source.png')
    plt.show()
    return embedding

# PCA analysis
def pca_analysis(df, features, n_components=5):
    """
    Perform PCA analysis on the selected features and print the explained variance ratio
    and top contributing features for each principal component.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of features to use for PCA.
        n_components (int): Number of principal components to compute (default: 5).

    Returns:
        pd.DataFrame: DataFrame with PCA results.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    pca = PCA(n_components)
    pca_result = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained variance ratio for each principal component:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.4f}")
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=features)
    for i in range(5):
        print(f"\nTop 5 features for PC{i+1}:")
        top_features = loadings.iloc[:, i].abs().nlargest(5).index
        for feature in top_features:
            print(f"{feature}: {loadings.loc[feature, f'PC{i+1}']:.4f}")
    return pca_result

# PCA visualization
def pca_visualization(df, pca_df, path_4_png, input_type):
    """
    Visualize the PCA results by plotting the first two principal components.

    Args:
        df (pd.DataFrame): Input DataFrame.
        pca_df (pd.DataFrame): DataFrame with PCA results.
        path_4_png (str): Path to save the plot.
        input_type (str): Input type for the plot title.
    """
    pca_df['source'] = df['source']
    plt.figure(figsize=(8, 6))
    for source in pca_df['source'].unique():
        subset = pca_df[pca_df['source'] == source]
        plt.scatter(subset['PC1'], subset['PC2'], label=source, alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection of Data Colored by Source')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path_4_png}/PCA_projection_{input_type}_source.png' if path_4_png else f'PCA_projection_{input_type}_source.png')
    plt.show()

# Main function to run the analysis
def run_feature_analysis():
    """
    Main function to execute the feature analysis pipeline. It performs the following steps:
    1. Load the dataset.
    2. Assign new columns based on the 'image_name' column.
    3. Group and pivot the data.
    4. Drop specific columns based on substrings.
    5. Perform correlation analysis to select unique features.
    6. Apply t-SNE, UMAP, and PCA for dimensionality reduction and visualization.

    Note: File paths and input types are hardcoded for this example.
    """
    file_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/2024-11-28-16-10_image_analysis_results_new-features.csv"
    path_4_png = '/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez'
    input_type = 'All-Combined'

    df = load_data(file_path)
    df = assign_columns(df)
    pivot_df = group_and_pivot(df)
    pivot_df = drop_columns(pivot_df, ['_channel1_Involucrin', '_channel1_Loricrin', '_channel1_Filaggrin'])
    most_unique_features = correlation_analysis(pivot_df)
    new_df = pivot_df[['person-ID', 'source'] + list(most_unique_features)].dropna(subset=list(most_unique_features))

    tsne_df = tsne_clustering(new_df, most_unique_features, path_4_png, input_type)
    umap_df = umap_clustering(new_df, most_unique_features, path_4_png, input_type)
    pca_df = pca_analysis(new_df, most_unique_features)
    pca_visualization(new_df, pca_df, path_4_png, input_type)

if __name__ == "__main__":
    run_feature_analysis()