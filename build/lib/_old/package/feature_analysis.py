#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import umap
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
def group_and_pivot_median(df):
    """
    Group data by 'person-ID' and 'type', calculate the median of numerical columns,
    and pivot the table to create a wide-format DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Pivoted DataFrame with aggregated values.
    """
    numerical_cols = df.select_dtypes(include=np.number).columns
    median_df = df.groupby(['person-ID', 'type'])[numerical_cols].median().reset_index()
    pivot_df = median_df.pivot(index='person-ID', columns='type', values=numerical_cols)
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df.reset_index()
    pivot_df.insert(1, 'source', df.groupby('person-ID')['source'].first().values)
    return pivot_df

def group_and_pivot_mean(df):
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

# def group_and_pivot(df):
#     """
#     Group data by 'person-ID' and 'type', and pivot the table to create a wide-format DataFrame.
#
#     This function reshapes the input DataFrame by grouping it based on 'person-ID' and 'type',
#     then pivots the data to create a wide-format table where each unique 'type' becomes a column.
#     It also adds a 'source' column to the resulting DataFrame, which contains the first 'source'
#     value for each 'person-ID'.
#
#     Args:
#         df (pd.DataFrame): Input DataFrame containing the data to be grouped and pivoted.
#
#     Returns:
#         pd.DataFrame: A wide-format DataFrame with aggregated values and a 'source' column.
#     """
#
#     pivot_df = df.pivot(index='person-ID', columns='type')
#     pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
#     pivot_df = pivot_df.reset_index()
#     pivot_df.insert(1, 'source', df.groupby('person-ID')['source'].first().values)
#     return pivot_df

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
def correlation_analysis(df, n_largest=505):
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
    return variances.nlargest(n_largest).index

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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, n_iter=n_iter)
    X_embedded = tsne.fit_transform(X_scaled)
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

def explain_tSne(X, X_embedded, verbose = False):
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Assuming you have X_scaled and X_embedded

  embedding_df = pd.DataFrame(X_embedded, columns=['tsne_dim_1', 'tsne_dim_2'])
  scaled_df = pd.DataFrame(X_scaled, columns=X.columns) # Assuming new_df has original column names

  # Calculate correlations
  correlations = pd.concat([scaled_df, embedding_df], axis=1).corr()
  tsne_1 =correlations.loc[scaled_df.columns, 'tsne_dim_1'].sort_values(ascending=False)
  tsne_2 =correlations.loc[scaled_df.columns, 'tsne_dim_2'].sort_values(ascending=False)


  # Look at correlations between original features and t-SNE dimensions
  if verbose:
    print("Correlation with t-SNE Dimension 1:")
    print(tsne_1)

    print("\nCorrelation with t-SNE Dimension 2:")
    print(tsne_2)

  return tsne_1, tsne_2

def absolute_scale_difference(df):
    """
    Calculates the absolute difference between the maximum and minimum values
    for each numerical column in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: A Series where the index is the column name and the values
                   are the absolute difference between the maximum and minimum
                   values for that column. Only numerical columns are included.
    """
    scale_diff = {}
    for col in df.select_dtypes(include=np.number).columns:
        min_val = df[col].min()
        max_val = df[col].max()
        scale_diff[col] = abs(max_val - min_val)
    return pd.Series(scale_diff)

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
    # Ensure only numerical features are selected
    numerical_features = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    # Use only numerical features for fitting UMAP
    scaler = StandardScaler()
    scaled_features_all = scaler.fit_transform(df[numerical_features].values)
    embedding = reducer.fit_transform(scaled_features_all)
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

def hdbscan_clustering(umap_embedding, min_cluster_size=5, min_samples=None):
    """
    Perform HDBSCAN clustering on UMAP embedding.

    Args:
        umap_embedding (np.ndarray): UMAP embedding of the data.
        min_cluster_size (int): Minimum size of clusters (default: 5).
        min_samples (int or None): Minimum samples in a cluster (default: None).

    Returns:
        np.ndarray: Cluster labels for each data point.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(umap_embedding)
    return cluster_labels

def plot_umap_embedding(umap_embedding, clusters, source, path_4_png, input_type):
    """
    Plots the UMAP embedding colored by clusters and uses symbols for sources.

    Args:
        umap_embedding (np.ndarray): The 2D UMAP embedding coordinates.
                                    Shape should be (n_samples, 2).
        clusters (np.ndarray): 1D array of cluster labels for each data point.
        source (pd.Series): Pandas Series containing source labels for each data point.
        path_4_png (str): Path to save the plot.
        input_type (str): Input type for the plot title.
    """
    if umap_embedding.shape[1] != 2:
        raise ValueError("umap_embedding must be a 2D array for plotting.")

    df_plot = pd.DataFrame({
        'umap_x': umap_embedding[:, 0],
        'umap_y': umap_embedding[:, 1],
        'cluster': clusters,
        'source': source
    })

    plt.figure(figsize=(10, 8))

    # Use seaborn for scatter plot with hue (cluster) and style (source)
    sns.scatterplot(data=df_plot, x='umap_x', y='umap_y', hue='cluster', style='source', palette='viridis', s=50)

    plt.xlabel('UMAP X')
    plt.ylabel('UMAP Y')
    plt.title(f'UMAP Projection of {input_type} Data Colored by Cluster and Styled by Source')
    plt.legend(title='Cluster', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True)

    if path_4_png:
        plt.savefig(f'{path_4_png}/umap_projection_{input_type}_cluster_source.png')
    else:
        plt.savefig(f'umap_projection_{input_type}_cluster_source.png')

    plt.show()

# Feature importance analysis
def get_umap_feature_importance(
  original_df: pd.DataFrame,
  umap_embedding: np.ndarray,
  feature_columns: list,
  test_size: float = 0.2,
  random_state: int = 42,
  n_top_features: int = 10
  ) -> dict:
  """
  Calculates the feature importance of original features in predicting UMAP embedding dimensions
  using a Random Forest Regressor.

  Ensures that UMAP coordinates themselves are NOT used as features for prediction.

  Args:
      original_df (pd.DataFrame): The original DataFrame containing all features.
      umap_embedding (np.ndarray): The 2D (or 3D) UMAP embedding coordinates.
                                  Shape should be (n_samples, n_components).
      feature_columns (list): A list of strings, containing the names of the
                              original columns in `original_df` that were used
                              to generate the UMAP embedding. These will be
                              used as predictors for the Random Forest.
      test_size (float): The proportion of the dataset to include in the test split.
      random_state (int): Controls the randomness of the data splitting and
                          Random Forest.
      n_top_features (int): The number of top features to display for each UMAP dimension.

  Returns:
      dict: A dictionary containing pandas Series of feature importances for each
            UMAP dimension (e.g., 'UMAP1_importance', 'UMAP2_importance').
  """

  if not isinstance(original_df, pd.DataFrame):
      raise TypeError("original_df must be a pandas DataFrame.")
  if not isinstance(umap_embedding, np.ndarray):
      raise TypeError("umap_embedding must be a numpy array.")
  if umap_embedding.ndim != 2:
      raise ValueError("umap_embedding must be a 2D array (n_samples, n_components).")
  if not all(col in original_df.columns for col in feature_columns):
      raise ValueError("Some feature_columns are not found in original_df.")
  if umap_embedding.shape[0] != original_df.shape[0]:
      raise ValueError("Number of samples in umap_embedding must match original_df.")
  if umap_embedding.shape[1] < 1:
      raise ValueError("UMAP embedding must have at least one dimension.")


  # --- CRITICAL STEP: Select ONLY the original features for X ---
  # This ensures that umap_x and umap_y (if they exist in original_df) are not used as predictors.
  X = original_df[feature_columns]

  # The UMAP embedding coordinates are the target (y)
  y = umap_embedding

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state
  )

  importance_results = {}

  # Iterate through each UMAP dimension
  for i in range(y.shape[1]):
      umap_dim_name = f"UMAP{i+1}"
      print(f"\n--- Feature Importance for predicting {umap_dim_name} ---")

      # Train Random Forest Regressor for the current UMAP dimension
      rf_model = RandomForestRegressor(random_state=random_state)
      rf_model.fit(X_train, y_train[:, i]) # Predict the i-th UMAP dimension

      # Get feature importances and create a pandas Series
      # The index is correctly set to the names of the original feature columns
      feature_importances = pd.Series(
          rf_model.feature_importances_, index=feature_columns
      )

      # Print top features
      print(feature_importances.sort_values(ascending=False).head(n_top_features))

      importance_results[f"{umap_dim_name}_importance"] = feature_importances

  return importance_results

# PCA analysis
def pca_analysis(df, features, n_components=2):
    """
    Perform PCA on the selected features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (list): List of features to use for PCA.
        n_components (int): Number of principal components to keep (default: 2).

    Returns:
        pd.DataFrame: DataFrame with PCA results.
    """
    X = df[features]
    # Ensure only numerical features are selected
    numerical_features = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]

    X_numerical = df[numerical_features]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numerical)

    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Create a DataFrame for the principal components
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

    # You might want to add back identifying columns from the original df
    # Assuming 'person-ID' and 'source' are important identifiers and are in the original df
    if 'person-ID' in df.columns and 'source' in df.columns:
        pca_df['person-ID'] = df['person-ID'].values
        pca_df['source'] = df['source'].values
    elif 'person-ID' in df.columns:
         pca_df['person-ID'] = df['person-ID'].values
    elif 'source' in df.columns:
        pca_df['source'] = df['source'].values

    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Explained variance ratio by components: {explained_variance_ratio}")
    print(f"Total explained variance: {explained_variance_ratio.sum()}")

    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained variance ratio for each principal component:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.4f}")

    # Dynamically create column names based on n_components for loadings DataFrame
    loading_columns = [f'PC{i+1}' for i in range(n_components)]
    # Ensure the index for loadings corresponds to the features used in PCA
    loadings = pd.DataFrame(pca.components_.T, columns=loading_columns, index=numerical_features)

    # Print top features based on the actual number of components
    for i in range(n_components):
        print(f"\nTop 5 features for PC{i+1}:")
        # Ensure we access columns based on the generated loading_columns
        top_features = loadings.iloc[:, i].abs().nlargest(5).index
        for feature in top_features:
            print(f"{feature}: {loadings.loc[feature, f'PC{i+1}']:.4f}")


    return pca_df

# PCA visualization
def pca_visualization(pca_df, path_4_png, input_type):
    """
    Visualize the PCA results by plotting the first two principal components.

    Args:
        df (pd.DataFrame): Input DataFrame.
        pca_df (pd.DataFrame): DataFrame with PCA results (should contain 'source' column).
        path_4_png (str): Path to save the plot.
        input_type (str): Input type for the plot title.
    """
    if 'source' not in pca_df.columns:
        print("Error: 'source' column not found in the PCA DataFrame. Cannot visualize by source.")
        return # Exit the function if 'source' is missing

    plt.figure(figsize=(8, 6))
    for source in pca_df['source'].unique():
        subset = pca_df[pca_df['source'] == source]
        plt.scatter(subset['PC1'], subset['PC2'], label=source, alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA Projection of Data ({input_type}) Colored by Source') # Added input_type to title
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path_4_png}/PCA_projection_{input_type}_source.png' if path_4_png else f'PCA_projection_{input_type}_source.png')
    plt.show()

def variance_analysis(df, most_unique_features, path_4_png, input_type):
    """
    Perform variance analysis and visualize the projection of data colored by source.

    Args:
        df (pd.DataFrame): DataFrame containing the data to analyze.
        most_unique_features (list): List of the most unique features for analysis.
        path_4_png (str): Path to save the plot.
        input_type (str): Input type for the plot title.
    """
    for source in df['source'].unique():
        subset = df[df['source'] == source]
        plt.scatter(subset[most_unique_features[0]], subset[most_unique_features[1]], label=source, alpha=0.7)

    plt.xlabel(most_unique_features[0])
    plt.ylabel(most_unique_features[1])
    plt.title(f'Variance Projection of {input_type} Data Colored by Source')
    plt.legend()
    plt.grid(True)
    if len(path_4_png) > 0:
        plt.savefig(f'{path_4_png}/var_projection_{input_type}_source.png')
    else:
        plt.savefig(f'var_projection_{input_type}_source.png')
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