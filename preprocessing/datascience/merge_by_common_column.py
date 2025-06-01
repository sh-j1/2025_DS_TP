import pandas as pd

# 파일 경로
mental_path = "C:/Users/ATIV/Desktop/vscode/vscode/mental_health_final_encoded.csv"
stress_path = "C:/Users/ATIV/Desktop/vscode/vscode/datascience/StressLevelDataset.csv"

# 1. 데이터 로드
mental_df = pd.read_csv(mental_path)
stress_df = pd.read_csv(stress_path)
stress_df.columns = stress_df.columns.str.strip().str.lower().str.replace(" ", "_")

# 2. 공통 컬럼 확인
common_cols = set(mental_df.columns).intersection(set(stress_df.columns))
print("🔍 공통 컬럼:", common_cols)

# 3. 병합 시도
if common_cols:
    # 예시: 가장 먼저 나오는 컬럼으로 병합
    merge_key = list(common_cols)[0]
    print(f"📎 공통 키 '{merge_key}' 기준 병합 시도 중...")

    merged_df = pd.merge(mental_df, stress_df, on=merge_key, how='inner')
    merged_df.to_csv("merged_by_common_column.csv", index=False)
    print("✅ 공통 컬럼 기준 병합 완료: merged_by_common_column.csv 저장됨")
else:
    print("❌ 공통 컬럼이 없어 병합할 수 없습니다.")
