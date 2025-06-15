"""
model_eval_plot.py

This script provides visualization utilities for evaluating machine learning models, specifically for classification and clustering tasks. It includes functions to:

- Visualize confusion matrices from CSV prediction files.
- Batch-generate confusion matrix plots for all prediction CSVs in a directory.
- Visualize the structure of a trained Decision Tree model.
- Visualize KMeans clustering results in 2D PCA space.

Preprocessing Steps & Assumptions:
- Prediction CSVs must contain 'y_true' and 'y_pred' columns with integer class labels (0, 1, 2).
- For Decision Tree visualization, a pickled model file and a list of feature names are required.
- For KMeans PCA visualization, the CSV must contain 'PC1', 'PC2', and 'cluster' columns.
- The script assumes a 3-class classification problem (labels: 0, 1, 2).

Dependencies:
- pandas
- matplotlib
- seaborn
- scikit-learn
- pickle
- os

Notes:
- All plots are saved to disk as PNG files.
- Designed for use in EDA and model evaluation pipelines.
"""

import os  # For directory and file operations
import pandas as pd  # For data manipulation and CSV reading
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced plotting
from sklearn.metrics import confusion_matrix, accuracy_score  # For evaluation metrics
from sklearn.tree import plot_tree  # For decision tree visualization
import pickle  # For loading serialized models

# 🔹 Confusion Matrix Visualization
def plot_confusion_from_csv(csv_path: str, output_path: str):
    # Read the CSV file containing predictions
    df = pd.read_csv(csv_path)
    # Extract true and predicted labels as integers
    y_true = df['y_true'].astype(int)
    y_pred = df['y_pred'].astype(int)

    # Compute the confusion matrix for 3 classes (0, 1, 2)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # Calculate accuracy score
    acc = accuracy_score(y_true, y_pred)

    # Create a new figure for the plot
    plt.figure(figsize=(6, 5))
    # Plot the confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    # Set axis labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.2f}')
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 🔹 Save Confusion Matrix Images for All CSVs
def generate_all_confusion_plots(pred_dir: str, output_dir: str):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Iterate over all files in the prediction directory
    for filename in os.listdir(pred_dir):
        # Process only CSV files
        if filename.endswith('.csv'):
            # Extract model name from filename
            model_name = filename.replace('_predictions.csv', '')
            # Build full paths for input and output
            csv_path = os.path.join(pred_dir, filename)
            output_path = os.path.join(output_dir, f'{model_name}_confusion.png')
            # Generate and save the confusion matrix plot
            plot_confusion_from_csv(csv_path, output_path)
            print(f"[OK] {model_name} confusion matrix saved.")

# 🔹 Decision Tree Structure Visualization (Example: Fold 1)
def visualize_decision_tree(model_path: str, feature_names: list, output_path: str):
    # Load the pickled decision tree model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Create a new figure for the tree plot
    plt.figure(figsize=(18, 10))
    # Plot the decision tree with feature and class names
    plot_tree(model, feature_names=feature_names, class_names=['0', '1', '2'], filled=True, fontsize=8)
    # Set the plot title
    plt.title('Decision Tree Structure (Fold 1)')
    # Save the plot to file
    plt.savefig(output_path)
    plt.close()
    print("[OK] Decision Tree 구조 시각화 완료")

# 🔹 KMeans PCA Visualization
def visualize_kmeans_pca(pca_csv_path: str, output_path: str):
    # Read the CSV file containing PCA and cluster assignments
    df = pd.read_csv(pca_csv_path)

    # Create a new figure for the scatter plot
    plt.figure(figsize=(7, 6))
    # Plot the PCA components colored by cluster label
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='Set2', data=df, s=50, alpha=0.8)
    # Set plot title and axis labels
    plt.title('KMeans Clustering (PCA 2D View)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("[OK] KMeans 2D 시각화 저장 완료")
