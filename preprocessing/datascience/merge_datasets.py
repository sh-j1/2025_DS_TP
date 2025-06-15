import pandas as pd
"""
This script merges two preprocessed datasets related to mental health and student stress into a single CSV file.
Dependencies:
    - pandas
Preprocessing Steps:
    1. Loads the preprocessed Mental Health Dataset from a specified CSV file.
    2. Loads the Student Stress Dataset from another CSV file and standardizes its column names by:
        - Stripping whitespace
        - Converting to lowercase
        - Replacing spaces with underscores
    3. Prints the shapes of both datasets for verification.
    4. Checks if both datasets have the same number of rows; raises an error if not.
    5. Merges the datasets horizontally (column-wise) based on their index.
    6. Saves the merged dataset to 'merged_dataset.csv'.
Notes/Assumptions:
    - The script assumes both datasets are preprocessed and aligned such that their rows correspond to the same entities.
    - File paths are hardcoded and should be updated as needed.
    - The merged dataset will only be created if the row counts match exactly.
"""

# 1. 전처리된 Mental Health Dataset 로드
mental_df = pd.read_csv("C:/Users/ATIV/Desktop/vscode/vscode/datascience/StressLevelDataset.csv")

# 2. Student Stress Dataset 로드 및 컬럼 정제
stress_df = pd.read_csv("C:/Users/ATIV/Desktop/vscode/vscode/mental_health_final_encoded.csv")
stress_df.columns = stress_df.columns.str.strip().str.lower().str.replace(" ", "_")

# 3. 기본 정보 출력
print("🧠 Mental Health Dataset shape:", mental_df.shape)
print("📚 Student Stress Dataset shape:", stress_df.shape)

# 4. 병합 가능 여부 확인
if mental_df.shape[0] != stress_df.shape[0]:
    raise ValueError("❌ 두 데이터셋의 행 수가 일치하지 않아 병합할 수 없습니다.")
else:
    print("✅ 행 수 일치. 병합 진행합니다.")

# 5. 열 방향 병합 (index 기준)
merged_df = pd.concat([mental_df.reset_index(drop=True), stress_df.reset_index(drop=True)], axis=1)

# 6. 결과 저장
merged_df.to_csv("merged_dataset.csv", index=False)
print("✅ 병합 완료: merged_dataset.csv 저장됨")
