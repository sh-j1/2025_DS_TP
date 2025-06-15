"""
train_cluster.py

This script performs KMeans clustering on a dataset, excluding the 'stress_level' column as the target variable.
It preprocesses the data by removing rows with missing values in the 'stress_level' column, then applies KMeans clustering to the remaining features.
The resulting cluster labels are saved alongside the original data, and a 2D PCA projection of the features with cluster assignments is also saved for visualization.

Dependencies:
- pandas
- scikit-learn (sklearn)
- os

Preprocessing steps:
- Loads data from a CSV file.
- Drops rows where 'stress_level' is missing.
- Excludes 'stress_level' from clustering features.

Outputs:
- 'kmeans_cluster_labels.csv': Original data with cluster labels.
- 'kmeans_pca_2d.csv': 2D PCA projection with cluster labels.

Notes/Assumptions:
- Assumes the input CSV contains a 'stress_level' column.
- Assumes all other columns are numeric and suitable for clustering.
- The output directory will be created if it does not exist.
"""

import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def train_kmeans(data_path: str, output_dir: str, n_clusters: int = 3):
    # Load the dataset from the given CSV file path
    df = pd.read_csv(data_path)
    # Drop rows where 'stress_level' is missing (required preprocessing)
    df = df.dropna(subset=['stress_level'])  # 필수 결측 제거

    # Exclude the target column 'stress_level' from clustering features
    X = df.drop(columns=['stress_level'])

    # Fit KMeans clustering to the features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Save the original data with the assigned cluster labels
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    # Save the clustered data to a CSV file
    df_clustered.to_csv(os.path.join(output_dir, 'kmeans_cluster_labels.csv'), index=False)

    # Optionally, perform PCA for 2D visualization and save the result
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels
    pca_df.to_csv(os.path.join(output_dir, 'kmeans_pca_2d.csv'), index=False)

    # Print completion message
    print("[OK] KMeans clustering completed and results saved.")
