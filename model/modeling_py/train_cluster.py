"""
train_cluster.py

This script performs KMeans clustering on a dataset with a 'stress_level' column.
It preprocesses the data by removing rows with missing 'stress_level' values,
excludes the 'stress_level' column from clustering features, and applies KMeans clustering.
The script saves the resulting cluster labels and a 2D PCA projection (for visualization) to the specified output directory.

Dependencies:
- pandas
- scikit-learn

Preprocessing steps:
- Removes rows with missing 'stress_level' values.
- Drops the 'stress_level' column before clustering.

Notes/Assumptions:
- The input CSV must contain a 'stress_level' column.
- All other columns are used as features for clustering.
- The script assumes numeric features (non-numeric columns may cause errors).
"""

import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def train_kmeans(data_path: str, output_dir: str, n_clusters: int = 3):
    # Load the dataset from the given CSV file path
    df = pd.read_csv(data_path)
    # Remove rows where 'stress_level' is missing (NaN)
    df = df.dropna(subset=['stress_level'])  # Remove essential missing values

    # Exclude the target column ('stress_level') from clustering features
    X = df.drop(columns=['stress_level'])

    # Fit KMeans clustering on the feature set
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Save the original data with the assigned cluster labels
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels

    # Create the output directory if it doesn't exist
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
