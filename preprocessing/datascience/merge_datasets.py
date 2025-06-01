import pandas as pd

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
