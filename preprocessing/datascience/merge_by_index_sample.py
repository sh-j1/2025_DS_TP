import pandas as pd

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
