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
    Assign type, image ID, and source based on image name.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new columns.
    """
    def assign_type(image_name):
        if "Filaggrin" in image_name:
            return "Filaggrin"
        elif "Loricrin" in image_name:
            return "Loricrin"
        elif "Involucrin" in image_name:
            return "Involucrin"
        else:
            return "Other"

    def assign_imageID(image_name):
        if "1" in image_name:
            return "1"
        elif "2" in image_name:
            return "2"
        elif "3" in image_name:
            return "3"
        else:
            return "Other"

    def assign_source(image_name):
        if "Healthy" in image_name:
            return "Healthy"
        elif "P" in image_name:
            return "Patient"
        else:
            return "Other"

    df["type"] = df["image_name"].apply(assign_type)
    df["image-ID"] = df["image_name"].apply(assign_imageID)
    df["source"] = df["image_name"].apply(assign_source)
    df['person-ID'] = df['image_name'].str.split('_').str[0]
    return df

# Group and pivot data
def group_and_pivot(df):
    """
    Group data by 'person-ID' and 'type', then pivot the table.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Pivoted DataFrame.
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
    Drop columns containing specific substrings.

    Args:
        df (pd.DataFrame): Input DataFrame.
        substrings (list): List of substrings to check.

    Returns:
        pd.DataFrame: DataFrame with columns dropped.
    """
    columns_to_drop = [col for col in df.columns if any(sub in col for sub in substrings)]
    return df.drop(columns=columns_to_drop)

# Correlation analysis
def correlation_analysis(df):
    """
    Perform correlation analysis and select the most unique features.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        list: List of most unique features.
    """
    correlation_matrix = df.iloc[:, 2:].corr()
    variances = correlation_matrix.var()
    return variances.nlargest(505).index

# t-SNE clustering
def tsne_clustering(df, features, path_4_png, input_type):
    """
    Perform t-SNE clustering and plot the results.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of features to use.
        path_4_png (str): Path to save the plot.
        input_type (str): Input type for the plot title.
    """
    X = df[features]
    source = df['source']
    tsne = TSNE(n_components=2, random_state=42)
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

# UMAP clustering
def umap_clustering(df, features, path_4_png, input_type):
    """
    Perform UMAP clustering and plot the results.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of features to use.
        path_4_png (str): Path to save the plot.
        input_type (str): Input type for the plot title.
    """
    reducer = umap.UMAP(n_neighbors=35, min_dist=0.15, random_state=42)
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

# PCA analysis
def pca_analysis(df, features):
    """
    Perform PCA analysis and print the explained variance ratio and top features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of features to use.

    Returns:
        pd.DataFrame: DataFrame with PCA results.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    pca = PCA(n_components=5)
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
    return pca_df

# PCA visualization
def pca_visualization(df, pca_df, path_4_png, input_type):
    """
    Visualize PCA results and save the plot.

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
def main():
    file_path = "/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/2024-11-28-16-10_image_analysis_results_new-features.csv"
    path_4_png = '/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez'
    input_type = 'All-Combined'

    df = load_data(file_path)
    df = assign_columns(df)
    pivot_df = group_and_pivot(df)
    pivot_df = drop_columns(pivot_df, ['_channel1_Involucrin', '_channel1_Loricrin', '_channel1_Filaggrin'])
    most_unique_features = correlation_analysis(pivot_df)
    new_df = pivot_df[['person-ID', 'source'] + list(most_unique_features)].dropna(subset=list(most_unique_features))

    tsne_clustering(new_df, most_unique_features, path_4_png, input_type)
    umap_clustering(new_df, most_unique_features, path_4_png, input_type)
    pca_df = pca_analysis(new_df, most_unique_features)
    pca_visualization(new_df, pca_df, path_4_png, input_type)

if __name__ == "__main__":
    main()