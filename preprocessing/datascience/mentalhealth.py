"""
This script preprocesses a mental health dataset by performing the following steps:
1. Loads the dataset from a CSV file and standardizes column names to lowercase with underscores.
2. Handles missing values in the 'self_employed' column by filling them with 'No' if the column exists.
3. Encodes all categorical columns using Label Encoding.
4. Scales all numerical columns (except the target column 'treatment') using StandardScaler, then restores the target column.
5. Detects and removes outliers from numerical columns based on a z-score threshold of 3.
6. Applies One-Hot Encoding to selected categorical columns ('gender', 'country', 'occupation') if they exist in the dataset.
7. Saves intermediate and final processed datasets to CSV files and prints status messages for each major step.
Dependencies:
    - pandas
    - numpy
    - scikit-learn
    - scipy
Note:
    - File paths and column names should be adjusted as needed for different datasets or environments.
    - The script assumes the presence of specific columns for encoding; check and modify the column list as necessary.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import zscore

# 1. 데이터 로드
df = pd.read_csv("C:/Users/ATIV/Desktop/vscode/vscode/Mental Health Dataset.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")  # 컬럼명 표준화

# 2. 결측치 처리
if 'self_employed' in df.columns:
    df['self_employed'] = df['self_employed'].fillna('No')  # 결측값 'No'로 대체

# 3. 범주형 컬럼 인코딩 (Label Encoding)
categorical_cols = df.select_dtypes(include='object').columns.tolist()  # object 타입 컬럼 추출
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))  # 각 범주형 컬럼에 라벨 인코딩 적용

# 4. 수치형 변수 스케일링 (모든 수치형 컬럼에 대해, 'treatment' 제외)
target_col = 'treatment'
target = df[target_col]  # 타겟 컬럼 값 백업

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  # 수치형 컬럼 추출
if target_col in numerical_cols:
    numerical_cols.remove(target_col)  # 타겟 컬럼 제외

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])  # 표준화 적용
df[target_col] = target  # 타겟 컬럼 복구

df.to_csv("mental_health_scaled.csv", index=False)
print("✅ 모든 수치형 변수 스케일링 완료: mental_health_scaled.csv 저장됨")

# 5. 이상치 탐지 및 제거 (z-score 기준)
df_outlier = df.copy()
numeric_cols = df_outlier.select_dtypes(include=['float64', 'int64']).columns.tolist()  # 수치형 컬럼 추출
if target_col in numeric_cols:
    numeric_cols.remove(target_col)  # 타겟 컬럼 제외

z_scores = np.abs(zscore(df_outlier[numeric_cols]))  # z-score 계산
threshold = 3
outlier_mask = (z_scores > threshold).any(axis=1)  # 임계값 초과 여부
print(f"❗ 이상치 행 수: {outlier_mask.sum()}개")

df_no_outlier = df_outlier[~outlier_mask]  # 이상치 제거

# 6. One-Hot Encoding 적용 (보완 조건 포함)
df = df_no_outlier.copy()

# 인코딩할 컬럼 리스트 설정
onehot_cols = ['gender', 'country', 'occupation']  # 실제 컬럼 이름이 존재하는지 확인 필요
existing_onehot_cols = [col for col in onehot_cols if col in df.columns]  # 존재하는 컬럼만 추출

if existing_onehot_cols:
    df_encoded = pd.get_dummies(df, columns=existing_onehot_cols, drop_first=True)  # 원-핫 인코딩 적용
    df_encoded.to_csv("mental_health_final_encoded.csv", index=False)
    print("✅ One-Hot Encoding 완료: mental_health_final_encoded.csv 저장됨")
else:
    print("⚠️ One-Hot Encoding 대상 컬럼이 존재하지 않음. 인코딩 생략됨.")
