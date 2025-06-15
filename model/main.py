"""
Main script for training, evaluating, and visualizing machine learning models on stress level data.
This script performs the following steps:
1. Loads a dataset from 'data/merged_by_index_sample.csv'.
2. Removes rows with missing values in the 'stress_level' column.
3. Splits the data into features (X) and target (y).
4. Performs stratified K-Fold cross-validation (default: 5 folds) to ensure balanced class distribution in each fold.
5. For each fold:
    - Trains multiple classification models using the training data.
    - Saves each trained model to disk.
6. Generates and saves predictions for all folds using the trained models.
7. Performs KMeans clustering on the dataset and saves clustering results.
8. Generates and saves confusion matrix plots for model evaluation.
9. Visualizes the structure of the decision tree model for the first fold.
10. Visualizes KMeans clustering results using 2D PCA projection.
Dependencies:
- pandas
- modeling.utils_split
- modeling.train_classifier
- modeling.predict_and_save
- modeling.train_cluster
- modeling.model_eval_plot
Assumptions/Notes:
- The input CSV file must contain a 'stress_level' column.
- All intermediate and output files are saved under the 'artifacts' directory.
- The script expects specific directory structures for models, predictions, clusters, and plots.
- Custom utility modules (under 'modeling') must be implemented and available in the Python path.
"""

import pandas as pd
from modeling.utils_split import get_stratified_kfold_splits
from modeling.train_classifier import train_models, save_model
from modeling.predict_and_save import predict_all_folds
from modeling.train_cluster import train_kmeans
from modeling.model_eval_plot import generate_all_confusion_plots
from modeling.model_eval_plot import visualize_decision_tree
from modeling.model_eval_plot import visualize_kmeans_pca

# 1. 데이터 로드
df = pd.read_csv('data/merged_by_index_sample.csv')

# NaN 있는 행 제거
df = df.dropna(subset=['stress_level'])

# 독립변수(X), 종속변수(y) 분리
X = df.drop(columns=['stress_level'])
y = df['stress_level']

# 2. Stratified K-Fold 분할
splits = get_stratified_kfold_splits(X, y, n_splits=5)

# 3. Fold별 모델 학습 및 저장
for fold_idx, (train_idx, val_idx) in enumerate(splits):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

    print(f"K-Fold {fold_idx+1} Leaning Model...")
    models = train_models(X_train, y_train)

    for name, model in models.items():
        model_name = f"{name}_fold{fold_idx+1}"
        save_model(model, model_name)
        print(f" {model_name} Save Complete!")

# 예측 결과 저장
predict_all_folds(
    model_dir='artifacts/models',
    data_path='data/merged_by_index_sample.csv',
    output_dir='artifacts/predictions'
)

# KMeans 클러스터링 실행
train_kmeans(
    data_path='data/merged_by_index_sample.csv',
    output_dir='artifacts/clusters',
    n_clusters=3
)

# Confusion Matrix 이미지 저장
generate_all_confusion_plots(
    pred_dir='artifacts/predictions',
    output_dir='artifacts/plots'
)

# Decision Tree 구조 시각화 (Fold 1만)
df = pd.read_csv('data/merged_by_index_sample.csv').dropna(subset=['stress_level'])
feature_names = df.drop(columns=['stress_level']).columns.tolist()

visualize_decision_tree(
    model_path='artifacts/models/decision_tree_fold1.pkl',
    feature_names=feature_names,
    output_path='artifacts/plots/decision_tree_structure.png'
)

# KMeans 2D PCA 시각화
visualize_kmeans_pca(
    pca_csv_path='artifacts/clusters/kmeans_pca_2d.csv',
    output_path='artifacts/plots/kmeans_pca_2d.png'
)
