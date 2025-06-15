"""
model_eval_plot.py
This script provides utility functions for visualizing and evaluating machine learning model results, specifically for classification and clustering tasks. It includes functions to plot confusion matrices from prediction CSV files, visualize decision tree structures, and display KMeans clustering results in PCA-reduced 2D space.
Dependencies:
    - os: For file and directory operations.
    - pandas: For reading and handling CSV data.
    - matplotlib.pyplot: For plotting and saving figures.
    - seaborn: For enhanced data visualization.
    - sklearn.metrics: For computing confusion matrices and accuracy scores.
    - sklearn.tree: For visualizing decision tree models.
    - pickle: For loading serialized model objects.
Functions:
    - plot_confusion_from_csv: Reads a CSV file containing true and predicted labels, computes the confusion matrix and accuracy, and saves a heatmap plot.
    - generate_all_confusion_plots: Iterates over all prediction CSV files in a directory, generating and saving confusion matrix plots for each.
    - visualize_decision_tree: Loads a pickled decision tree model and visualizes its structure, saving the plot as an image.
    - visualize_kmeans_pca: Reads a CSV file with PCA-reduced features and cluster assignments, and visualizes the clustering in 2D.
Preprocessing Steps & Assumptions:
    - Prediction CSV files must contain 'y_true' and 'y_pred' columns for confusion matrix plotting.
    - Decision tree models must be serialized (pickled) and compatible with sklearn's plot_tree.
    - For KMeans PCA visualization, the CSV must include 'PC1', 'PC2', and 'cluster' columns.
    - Class labels are assumed to be [0, 1, 2] for classification tasks.
    - Output directories are created if they do not exist.
Notes:
    - The script is modular and can be imported as a utility or run as part of a larger evaluation pipeline.
    - All plots are saved to disk; no interactive display is provided.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import plot_tree
import pickle

# 🔹 Visualize Confusion Matrix from CSV
def plot_confusion_from_csv(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)  # Read the CSV file containing predictions
    y_true = df['y_true']       # Extract true labels
    y_pred = df['y_pred']       # Extract predicted labels

    cm = confusion_matrix(y_true, y_pred, labels=[0.0, 1.0, 2.0])  # Compute confusion matrix
    acc = accuracy_score(y_true, y_pred)                           # Compute accuracy

    plt.figure(figsize=(6, 5))  # Set figure size
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])  # Plot heatmap
    plt.xlabel('Predicted Label')      # X-axis label
    plt.ylabel('True Label')           # Y-axis label
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.2f}')  # Plot title with accuracy
    plt.tight_layout()                 # Adjust layout
    plt.savefig(output_path)           # Save plot to file
    plt.close()                        # Close the plot

# 🔹 Save Confusion Matrix images for all CSVs in a directory
def generate_all_confusion_plots(pred_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    for filename in os.listdir(pred_dir):   # Iterate over files in prediction directory
        if filename.endswith('.csv'):       # Process only CSV files
            model_name = filename.replace('_predictions.csv', '')  # Extract model name
            csv_path = os.path.join(pred_dir, filename)            # Full path to CSV
            output_path = os.path.join(output_dir, f'{model_name}_confusion.png')  # Output image path
            plot_confusion_from_csv(csv_path, output_path)         # Generate and save confusion matrix plot
            print(f"[OK] {model_name} confusion matrix saved.")    # Print status

# 🔹 Visualize Decision Tree structure (example for Fold 1)
def visualize_decision_tree(model_path: str, feature_names: list, output_path: str):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)    # Load the pickled decision tree model

    plt.figure(figsize=(18, 10))  # Set figure size
    plot_tree(model, feature_names=feature_names, class_names=['0', '1', '2'], filled=True, fontsize=8)  # Plot tree
    plt.title('Decision Tree Structure (Fold 1)')  # Set plot title
    plt.savefig(output_path)      # Save plot to file
    plt.close()                   # Close the plot
    print("[OK] Decision Tree 구조 시각화 완료")  # Print status

# 🔹 Visualize KMeans clustering in PCA-reduced 2D space
def visualize_kmeans_pca(pca_csv_path: str, output_path: str):
    df = pd.read_csv(pca_csv_path)  # Read the CSV file with PCA and cluster info

    plt.figure(figsize=(7, 6))      # Set figure size
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='Set2', data=df, s=50, alpha=0.8)  # Scatter plot
    plt.title('KMeans Clustering (PCA 2D View)')  # Set plot title
    plt.xlabel('Principal Component 1')           # X-axis label
    plt.ylabel('Principal Component 2')           # Y-axis label
    plt.legend(title='Cluster')                   # Add legend
    plt.tight_layout()                            # Adjust layout
    plt.savefig(output_path)                      # Save plot to file
    plt.close()                                   # Close the plot
    print("[OK] KMeans 2D 시각화 저장 완료")        # Print status
