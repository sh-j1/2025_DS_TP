import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import plot_tree
import pickle

# 🔹 Confusion Matrix 시각화
def plot_confusion_from_csv(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    y_true = df['y_true']
    y_pred = df['y_pred']

    cm = confusion_matrix(y_true, y_pred, labels=[0.0, 1.0, 2.0])
    acc = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix\nAccuracy: {acc:.2f}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 🔹 모든 CSV에 대해 Confusion Matrix 이미지 저장
def generate_all_confusion_plots(pred_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(pred_dir):
        if filename.endswith('.csv'):
            model_name = filename.replace('_predictions.csv', '')
            csv_path = os.path.join(pred_dir, filename)
            output_path = os.path.join(output_dir, f'{model_name}_confusion.png')
            plot_confusion_from_csv(csv_path, output_path)
            print(f"[OK] {model_name} confusion matrix saved.")

# 🔹 Decision Tree 구조 시각화 (Fold 1만 예시)
def visualize_decision_tree(model_path: str, feature_names: list, output_path: str):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    plt.figure(figsize=(18, 10))
    plot_tree(model, feature_names=feature_names, class_names=['0', '1', '2'], filled=True, fontsize=8)
    plt.title('Decision Tree Structure (Fold 1)')
    plt.savefig(output_path)
    plt.close()
    print("[OK] Decision Tree 구조 시각화 완료")

# 🔹 KMeans PCA 시각화
def visualize_kmeans_pca(pca_csv_path: str, output_path: str):
    df = pd.read_csv(pca_csv_path)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='Set2', data=df, s=50, alpha=0.8)
    plt.title('KMeans Clustering (PCA 2D View)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print("[OK] KMeans 2D 시각화 저장 완료")
