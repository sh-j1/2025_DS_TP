import pandas as pd
"""
This script merges two datasets—one containing mental health data and another containing stress level data—by aligning their rows based on index order.
Dependencies:
    - pandas
Preprocessing Steps:
    1. Loads the mental health dataset from a specified CSV file.
    2. Loads the stress level dataset from a specified CSV file.
    3. Standardizes the column names of the stress dataset by stripping whitespace, converting to lowercase, and replacing spaces with underscores.
    4. Selects the top 1100 rows from the stress dataset to match the expected number of rows for merging.
    5. Resets the indices of both datasets and concatenates them column-wise.
    6. Saves the merged dataset to a new CSV file named 'merged_by_index_sample.csv'.
Notes/Assumptions:
    - Assumes that the number of rows in the mental health dataset matches or exceeds 1100.
    - Assumes that both input CSV files exist at the specified paths.
    - The merge is performed purely by row order, not by any key or identifier.
    - The script is intended for use in a preprocessing pipeline where aligned datasets are required for further analysis.
"""

# 파일 경로
mental_path = "C:/Users/ATIV/Desktop/vscode/vscode/mental_health_final_encoded.csv"
stress_path = "C:/Users/ATIV/Desktop/vscode/vscode/datascience/StressLevelDataset.csv"

# 1. Mental Health 데이터 로드
mental_df = pd.read_csv(mental_path)

# 2. Stress Dataset 로드
stress_df = pd.read_csv(stress_path)
stress_df.columns = stress_df.columns.str.strip().str.lower().str.replace(" ", "_")

# 3. Stress Dataset에서 상위 1100개만 추출
stress_df_sampled = stress_df.head(1100)

# 4. 병합 (행 수 맞춤 후 열 기준 병합)
merged_df = pd.concat([mental_df.reset_index(drop=True), stress_df_sampled.reset_index(drop=True)], axis=1)

# 5. 저장
merged_df.to_csv("merged_by_index_sample.csv", index=False)
print("✅ 병합 완료: merged_by_index_sample.csv 저장됨")
