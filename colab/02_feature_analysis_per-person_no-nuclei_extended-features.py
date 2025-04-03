#!/usr/bin/env python
# coding: utf-8

# ## Data Loading

# In[ ]:


get_ipython().system('pip install umap-learn')


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv("/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez/2024-11-28-16-10_image_analysis_results_new-features.csv")
path_4_png='/content/gdrive/Shareddrives/CzechBioImaging-DA-Parez'


# In[ ]:


df.head()


# In[ ]:


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


# In[ ]:


df.tail()


# In[ ]:


# Group data by 'person-ID' and 'type', then calculate the mean of numerical columns
numerical_cols = df.select_dtypes(include=np.number).columns
mean_df = df.groupby(['person-ID', 'type'])[numerical_cols].mean().reset_index()

# Display the resulting DataFrame
mean_df

# Pivot the table to have 'type' as columns
pivot_df = mean_df.pivot(index='person-ID', columns='type', values=numerical_cols)

# Flatten the MultiIndex columns
pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

# Reset the index to make 'person-ID' a regular column
pivot_df = pivot_df.reset_index()

# Display the resulting DataFrame
pivot_df


# In[ ]:


# Insert the 'source' column into the pivot_df DataFrame
pivot_df.insert(1, 'source', df.groupby('person-ID')['source'].first().values)
pivot_df


# In[ ]:


for column in pivot_df.columns:
  print(column)


# In[ ]:


# prompt: I want to drop any columns containing name _channel1_Filaggrin and _channel1_Loricrin

# Drop columns containing specific substrings
# columns_to_drop = [col for col in pivot_df.columns if '_channel1_Involucrin' in col]# or '_channel1_Loricrin' in col]
columns_to_drop = [col for col in pivot_df.columns if '_channel1_Involucrin' in col or '_channel1_Loricrin' in col or '_channel1_Filaggrin' in col]
pivot_df = pivot_df.drop(columns=columns_to_drop)
pivot_df
path_4_png=''


# In[ ]:


# @title
import umap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# def do_the_thing_UMAP(df,input_type,path_4_png):
# Select Type
# if input_type == 'Loricrin':
#   df_type = df[df['type'] == input_type]
# elif input_type == 'Filaggrin':
#   df_type = df[df['type'] == input_type]
# elif input_type == 'Involucrin':
#   df_type = df[df['type'] == input_type]
# else: #all
#   df_type = df#[df['type'] == input_type]

input_type='All-Combined'

df_type = pivot_df

# Correlation analysis
correlation_matrix = df_type.iloc[:, 2:].corr()

# Select the x most unique features based on variance (you can use other metrics as well)
# variances = df_type.iloc[:, 3:].var()
variances = correlation_matrix.var()
most_unique_features = variances.nlargest(505).index #.nsmallest(25).index

# Create a new dataframe with the selected features
new_df = df_type[['person-ID','source'] + list(most_unique_features)]
# Drop rows with NaN values in the specified columns
new_df = new_df.dropna(subset=list(most_unique_features))

new_df


# UMAP vs T-sne: https://pair-code.github.io/understanding-umap/

# In[ ]:


# prompt: use t-sne cluster based on source column

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming 'new_df' is already defined as in the previous code

# Prepare the data for t-SNE
X = new_df.iloc[:, 2:]  # Exclude 'person-ID' and 'source' columns
source = new_df['source']

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42) # You can adjust perplexity, early_exaggeration, etc.
X_embedded = tsne.fit_transform(X)

# Create the scatter plot
plt.figure(figsize=(10, 8))
for s in source.unique():
  plt.scatter(X_embedded[source == s, 0], X_embedded[source == s, 1], label=s)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Clustering based on Source")
plt.legend()
# plt.show()

if len(path_4_png) > 0:
  plt.savefig(f'{path_4_png}/Tsne_projection_{input_type}_source.png')
else:
  plt.savefig(f'Tsne_projection__projection_{input_type}_source.png')
plt.show()


# In[ ]:


# prompt: use umap cluster based on source column

# Assuming the previous code is executed and 'new_df' is available

import umap

# Create a UMAP reducer with specified parameters
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

# Fit and transform the data using the selected features, excluding 'person-ID' and 'source'
embedding = reducer.fit_transform(new_df.iloc[:, 2:])

# Create a new DataFrame with the embedding results
embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])

# Concatenate the embedding results with 'person-ID' and 'source'
embedding_df = pd.concat([new_df[['person-ID', 'source']], embedding_df], axis=1)

# Now you can use the 'embedding_df' for visualization or further analysis
# For example, plot the UMAP embedding colored by 'source':
plt.figure(figsize=(10, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='source', data=embedding_df)
plt.title('UMAP Embedding Colored by Source')
plt.show()


# In[ ]:


#Use UMAP to cluster features in df_new

# Select the feature columns
# feature_cols = most_unique_features
feature_cols = new_df.columns[2:]
X = new_df[feature_cols].values


# Apply UMAP
# reducer = umap.UMAP(n_neighbors=7, min_dist=0.5, random_state=42) # Adjust parameters as needed
reducer = umap.UMAP(n_neighbors=35, min_dist=0.15, random_state=42) # Adjust parameters as needed
embedding = reducer.fit_transform(X)

# Add UMAP coordinates to the DataFrame
new_df['umap_x'] = embedding[:, 0]
new_df['umap_y'] = embedding[:, 1]



# Now you can use df_new['umap_x'] and df_new['umap_y'] for plotting or further analysis

# Example plot (optional)
plt.scatter(new_df['umap_x'], new_df['umap_y'])
plt.show()


# In[ ]:


# path_4_png=''

# Create the scatter plot with different symbols based on the 'source' column
plt.figure(figsize=(8, 6))

for source in new_df['source'].unique():
    subset = new_df[new_df['source'] == source]
    plt.scatter(subset['umap_x'], subset['umap_y'], label=source, alpha=0.7)

plt.xlabel('UMAP X')
plt.ylabel('UMAP Y')
plt.title(f'UMAP Projection of {input_type} Data Colored by Source')
plt.legend()
plt.grid(True)
if len(path_4_png) > 0:
  plt.savefig(f'{path_4_png}/umap_projection_{input_type}_source.png')
  print(f'{path_4_png}/umap_projection_{input_type}_source.png')
else:
  plt.savefig(f'umap_projection_{input_type}_source.png')
plt.show()


# In[ ]:


# Now you can use df_new['umap_x'] and df_new['umap_y'] for plotting or further analysis

# Example plot (optional)
plt.scatter(new_df[most_unique_features[0]], new_df[most_unique_features[1]])
plt.show()

# Create the scatter plot with different symbols based on the 'source' column
plt.figure(figsize=(8, 6))

for source in new_df['source'].unique():
    subset = new_df[new_df['source'] == source]
    plt.scatter(subset[most_unique_features[0]], subset[most_unique_features[1]], label=source, alpha=0.7)

plt.xlabel(most_unique_features[0])
plt.ylabel(most_unique_features[1])
plt.title(f'var Projection of {input_type} Data Colored by Source')
plt.legend()
plt.grid(True)
if len(path_4_png) > 0:
  plt.savefig(f'{path_4_png}/var_projection_{input_type}_source.png')
else:
  plt.savefig(f'var_projection_{input_type}_source.png')
plt.show()


# In[ ]:


# prompt: do a PCA analysis and explain first 5 most meaningful features

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming 'new_df' and 'most_unique_features' are defined from the previous code

# Select numerical features for PCA
features = new_df[most_unique_features]#.dropna()

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=5)  # Reduce to 5 principal components
pca_result = pca.fit_transform(scaled_features)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Print explained variance for each principal component
print("Explained variance ratio for each principal component:")
for i, ratio in enumerate(explained_variance_ratio):
  print(f"PC{i+1}: {ratio:.4f}")


# Analyze loadings to interpret principal components
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=most_unique_features)

# Print the top 5 features with highest absolute loading for each component
for i in range(5):  # Iterate through each PC
  print(f"\nTop 5 features for PC{i+1}:")
  top_features = loadings.iloc[:,i].abs().nlargest(5).index
  for feature in top_features:
      print(f"{feature}: {loadings.loc[feature,f'PC{i+1}']:.4f}")


# In[ ]:


# Select numerical features for PCA
features = new_df[most_unique_features]#.dropna()

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 5 principal components
pca_result = pca.fit_transform(scaled_features)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Print explained variance for each principal component
print("Explained variance ratio for each principal component:")
for i, ratio in enumerate(explained_variance_ratio):
  print(f"PC{i+1}: {ratio:.4f}")


# Analyze loadings to interpret principal components
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=most_unique_features)

# Print the top 5 features with highest absolute loading for each component
for i in range(2):  # Iterate through each PC
  print(f"\nTop 2 features for PC{i+1}:")
  top_features = loadings.iloc[:,i].abs().nlargest(5).index
  for feature in top_features:
      print(f"{feature}: {loadings.loc[feature,f'PC{i+1}']:.4f}")


# ## Explained variance ratio for each principal component:
# PC1: 0.2054
# PC2: 0.1178
# PC3: 0.1005
# PC4: 0.0666
# PC5: 0.0573
# 
# ### Top 5 features for PC1:
#  * minor_axis_length_mean_channel1_Loricrin: 0.0903
#  * minor_axis_length_mean_channel1_Filaggrin: 0.0903
#  * minor_axis_length_median_channel1_Loricrin: 0.0900
#  * minor_axis_length_median_channel1_Filaggrin: 0.0900
#  * solidity_median_channel1_Filaggrin: 0.0899
# 
# ### Top 5 features for PC2:
#  * solidity_std_channel2_Loricrin: 0.0950
#  * major_axis_length_mean_channel2_Loricrin: 0.0911
#  * solidity_mean_channel2_Loricrin: -0.0910
#  * solidity_mean_channel2_Filaggrin: -0.0882
#  * major_axis_length_mean_channel2_Involucrin: 0.0876
# 
# ### Top 5 features for PC3:
#  * centroid-0_count_Loricrin: 0.0985
#  * perimeter_count_channel2_Loricrin: 0.0985
#  * bbox-3_count_Loricrin: 0.0985
#  * bbox-0_count_Loricrin: 0.0985
#  * area_count_channel2_Loricrin: 0.0985
# 
# ### Top 5 features for PC4:
#  * min_intensity_mean_Filaggrin: 0.1102
#  * min_intensity_mean_Loricrin: 0.1102
#  * min_intensity_median_Loricrin: 0.1099
#  * min_intensity_median_Filaggrin: 0.1099
#  * quartiles-1_max_channel2_Involucrin: 0.0928
# 
# ### Top 5 features for PC5:
#  * quartiles-0_mean_channel1_Involucrin: 0.1338
#  * quartiles-0_median_channel1_Involucrin: 0.1332
#  * quartiles-1_median_channel1_Involucrin: 0.1260
#  * quartiles-1_mean_channel1_Involucrin: 0.1255
#  * mean_intensity_mean_channel1_Involucrin: 0.1254

# In[ ]:


new_df = new_df.reset_index(drop=True)


# In[ ]:


# Select numerical features for PCA
# features = new_df[most_unique_features]#.dropna()

# # Apply PCA with 2 components for visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(features)

# Create a DataFrame for the PCA results
# pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df = pca_df[['PC1', 'PC2']]
pca_df['source'] = new_df['source']


# Plot the PCA results
plt.figure(figsize=(8, 6))
for source in pca_df['source'].unique():
    subset = pca_df[pca_df['source'] == source]
    plt.scatter(subset['PC1'], subset['PC2'], label=source, alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Data Colored by Source')
plt.legend()
plt.grid(True)

if len(path_4_png) > 0:
  plt.savefig(f'{path_4_png}/PCA_projection_{input_type}_source.png')
  print(f'{path_4_png}/PCA_projection_{input_type}_source.png')
else:
  plt.savefig(f'PCA_projection_{input_type}_source.png')
plt.show()


# In[ ]:


x


# ## Details

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select only one Type
df_fillagrin = df[df['type'] == 'Loricrin']

# Correlation analysis
correlation_matrix = df_fillagrin.iloc[:, :-4].corr()


# In[ ]:


for column in correlation_matrix.columns:
  print(column)


# In[ ]:


# Plot the correlation matrix using seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features for Filaggrin')
plt.show()


# In[ ]:


# Select the 5 most unique features based on variance (you can use other metrics as well)
# variances = df_fillagrin.iloc[:, 3:].var()
variances = correlation_matrix.var()
most_unique_features = variances.nlargest(55).index #.nsmallest(25).index

# Create a new dataframe with the selected features
new_df = df_fillagrin[['source','image-ID', 'type', 'image_name'] + list(most_unique_features)]

new_df.head()


# In[ ]:


most_unique_features


# In[ ]:


# Drop rows with NaN values in the specified columns
new_df = new_df.dropna(subset=list(most_unique_features))


# In[ ]:


#Use UMAP to cluster features in df_new

import umap
import pandas as pd


# Select the feature columns
feature_cols = most_unique_features
X = new_df[feature_cols].values


# Apply UMAP
reducer = umap.UMAP(n_neighbors=7, min_dist=0.5, random_state=42) # Adjust parameters as needed
embedding = reducer.fit_transform(X)

# Add UMAP coordinates to the DataFrame
new_df['umap_x'] = embedding[:, 0]
new_df['umap_y'] = embedding[:, 1]

# Now you can use df_new['umap_x'] and df_new['umap_y'] for plotting or further analysis

# Example plot (optional)
plt.scatter(new_df['umap_x'], new_df['umap_y'])
plt.show()


# In[ ]:


# prompt: plot scatter plot of umap_x and umap_y with different symbols based on source column

import matplotlib.pyplot as plt

# Create the scatter plot with different symbols based on the 'source' column
plt.figure(figsize=(8, 6))

for source in new_df['source'].unique():
    subset = new_df[new_df['source'] == source]
    plt.scatter(subset['umap_x'], subset['umap_y'], label=source, alpha=0.7)

plt.xlabel('UMAP X')
plt.ylabel('UMAP Y')
plt.title('UMAP Projection of Filaggrin Data Colored by Source')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# Create the scatter plot with different symbols based on the 'source' column
plt.figure(figsize=(8, 6))

for source in new_df['type'].unique():
    subset = new_df[new_df['type'] == source]
    plt.scatter(subset['umap_x'], subset['umap_y'], label=source, alpha=0.7)

plt.xlabel('UMAP X')
plt.ylabel('UMAP Y')
plt.title('UMAP Projection of Filaggrin Data Colored by Type')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




