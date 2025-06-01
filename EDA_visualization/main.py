import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from preprocessing.preprocessing import data_inspection
from preprocessing.preprocessing import data_cleaning
from preprocessing.preprocessing import preprocessing
from visualization.vclass import EDA_relavant_statics
from modeling_py.utils_split import get_stratified_kfold_splits
from modeling_py.train_classifier import train_models, save_model
from modeling_py.predict_and_save import predict_all_folds_Train_Test
from modeling_py.train_cluster import train_kmeans
from modeling_py.model_eval_plot import generate_all_confusion_plots
from modeling_py.model_eval_plot import visualize_decision_tree
from modeling_py.model_eval_plot import visualize_kmeans_pca
from visualization.vclass import visualize_compare_variable_growingstress

model_params={
    'logistic_regression':{
        0:{'max_iter':500},
        1:{'max_iter':1000},
        2:{'max_iter':1500},
        3:{'max_iter':2000},
        4:{'max_iter':2500}
    },
    'decision_tree':{
        0:{'max_depth':3},
        1:{'max_depth':4},
        2:{'max_depth':5},
        3:{'max_depth':6},
        4:{'max_depth':7}
    }
}

data_inspection(
    data_path="StressLevelDataset(original).csv",
    output_dir="artifacts/plots"
)
data_cleaning(
    data_path="StressLevelDataset(original).csv",
    output_dir="artifacts/csv_data"
)
preprocessing(
    data_path="StressLevelDataset(cleaned).csv",
    output_dir="artifacts/csv_data"
)
EDA_relavant_statics(
    data_path="StressLevelDataset(final).csv",
    output_dir = "artifacts/plot"
)
print("------------------------------------------------ 전처리 끝 ------------------------------------------")
# 데이터 로드
df = pd.read_csv('StressLevelDataset(final).csv')
# NaN 있는 행 제거
df = df.dropna(subset=['stress_level'])
# 독립변수(X), 종속변수(y) 분리
X = df.drop(columns=['stress_level'])
y = df['stress_level']

# 1. train set과 test 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2, random_state=42, stratify=y
)
# 1-1. test 셋은 따로 저장
df_test = pd.concat([X_test,y_test.rename('stress_level')],axis=1)
output_dir_to_csv = "artifacts/csv_data"
output_path = os.path.join(output_dir_to_csv, 'TestDataset.csv')
df_test.to_csv(output_path,index=False)
# 2. Stratified K-Fold 분할
splits = get_stratified_kfold_splits(X_train, y_train, n_splits=5)

os.makedirs(output_dir_to_csv, exist_ok=True)
# 3. Fold별로 훈련 데이터 csv에 저장, 모델 학습 및 저장
for fold_idx, (train_idx, val_idx) in enumerate(splits):
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]

    print(f"K-Fold {fold_idx+1} TrainData saving...")
    df_train_fold = pd.concat([X_train_fold, y_train_fold.rename('stress_level')], axis=1)
    
    output_path = os.path.join(output_dir_to_csv, f'TrainDataset_fold{fold_idx+1}.csv')
    df_train_fold.to_csv(output_path,index=False)

    print(f"K-Fold {fold_idx+1} Learning Model...")
    models = train_models(X_train_fold, y_train_fold, fold_idx)

    for name, model in models.items():
        model_name = f"{name}_fold{fold_idx+1}"
        save_model(model, model_name)
        print(f" {model_name} Save Complete!")

# 예측 결과 저장
all=[]
for i in range(5):
    result_df = predict_all_folds_Train_Test(
        model_dir='artifacts/models',
        Train_data_path = f'TrainDataset_fold{i+1}.csv',
        Test_data_path = 'TestDataset.csv',
        output_dir='artifacts/predictions',
        fold = i+1
    )
    all.append(result_df)
all_df = pd.concat(all,ignore_index=True)
output_path = os.path.join(output_dir_to_csv, 'Train & Test data Score comparison.csv')
all_df.to_csv(output_path)

# plot으로 최적 iter or depth 보여주기
df_lr = all_df[all_df['model_type'] == 'logistic_regression'].copy()
df_dt = all_df[all_df['model_type'] == 'decision_tree'].copy()

df_lr['iteration'] = df_lr['fold'].apply(lambda x: model_params['logistic_regression'][x-1]['max_iter'])
df_dt['max_depth'] = df_dt['fold'].apply(lambda x: model_params['decision_tree'][x-1]['max_depth'])

df_lr['score_diff'] = df_lr['train_score'] - df_lr['test_score']
df_dt['score_diff'] = df_dt['train_score'] - df_dt['test_score']
output_dir='artifacts/plot'
output_path = os.path.join(output_dir, 'LogisticRegression Train & Test data Score comparison.png')
# 로지스틱 리그레션 plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df_lr['iteration'], df_lr['train_score'], 'bo-', label='Train Score')
plt.plot(df_lr['iteration'], df_lr['test_score'], 'ro-', label='Test Score')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Logistic Regression Scores by Iteration')
plt.legend()
plt.subplot(1, 2, 2)
sns.barplot(x='iteration', y='score_diff', data=df_lr, color='lightblue')
plt.xlabel('Iteration')
plt.ylabel('Train - Test Score Difference')
plt.title('Logistic Regression Train-Test Difference')
plt.tight_layout()
plt.savefig(output_path)

output_path = os.path.join(output_dir, 'SelectionTree Train & Test data Score comparison.png')
# 셀렉션 트리 plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df_dt['max_depth'], df_dt['train_score'], 'bo-', label='Train Score')
plt.plot(df_dt['max_depth'], df_dt['test_score'], 'ro-', label='Test Score')
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Decision Tree Scores by Max Depth')
plt.legend()
plt.subplot(1, 2, 2)
sns.barplot(x='max_depth', y='score_diff', data=df_dt, color='lightgreen')
plt.xlabel('Max Depth')
plt.ylabel('Train - Test Score Difference')
plt.title('Decision Tree Train-Test Difference')
plt.tight_layout()
plt.savefig(output_path)

# KMeans 클러스터링 실행
train_kmeans(
    data_path='StressLevelDataset(final).csv',
    output_dir='artifacts/clusters',
    n_clusters=3
)

# Confusion Matrix 이미지 저장
generate_all_confusion_plots(
    pred_dir='artifacts/predictions',
    output_dir='artifacts/plots'
)

# Decision Tree 구조 시각화 (Fold 1만)
df = pd.read_csv('StressLevelDataset(final).csv').dropna(subset=['stress_level'])
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
#----------------------------------------------------EDA, Model-predicition comparison---------------------------------------------------------------------------
"""visualize_compare_variable_growingstress(#여긴 나중에 시간 나면 추가할게요...하다보니까 끝이 없어서
    data_path='StressLevelDataset(final).csv',
    output_dir='artifacts/eda',
    variables=['days_indoors','mood_swings','coping_struggles','work_interest','social_weakness']
)"""
