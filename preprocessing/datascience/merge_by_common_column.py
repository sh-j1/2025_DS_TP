"""
This script merges two CSV datasets based on their common columns using pandas.
Preprocessing Steps:
- Loads two datasets: a mental health dataset and a stress level dataset, from specified file paths.
- Standardizes the column names of the stress level dataset by stripping whitespace, converting to lowercase, and replacing spaces with underscores to facilitate matching.
- Identifies common columns between the two datasets.
- If common columns exist, merges the datasets using the first common column as the key with an inner join.
- Saves the merged DataFrame to 'merged_by_common_column.csv'.
Dependencies:
- pandas
Notes/Assumptions:
- The script assumes that both input CSV files exist at the specified paths.
- Only the first common column found is used as the merge key.
- The merge is performed as an inner join, so only rows with matching values in the merge key will be included in the output.
- The script prints informative messages about the merge process and the result.
"""
import pandas as pd

# File paths for the datasets
mental_path = "C:/Users/ATIV/Desktop/vscode/vscode/mental_health_final_encoded.csv"
stress_path = "C:/Users/ATIV/Desktop/vscode/vscode/datascience/StressLevelDataset.csv"

# 1. Load the datasets into DataFrames
mental_df = pd.read_csv(mental_path)
stress_df = pd.read_csv(stress_path)

# Standardize column names in stress_df: strip whitespace, lowercase, and replace spaces with underscores
stress_df.columns = stress_df.columns.str.strip().str.lower().str.replace(" ", "_")

# 2. Find common columns between the two DataFrames
common_cols = set(mental_df.columns).intersection(set(stress_df.columns))
print("🔍 Common columns:", common_cols)

# 3. Attempt to merge if there are common columns
if common_cols:
    # Use the first common column as the merge key
    merge_key = list(common_cols)[0]
    print(f"📎 Attempting to merge on common key '{merge_key}'...")

    # Merge the DataFrames on the selected key using an inner join
    merged_df = pd.merge(mental_df, stress_df, on=merge_key, how='inner')
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv("merged_by_common_column.csv", index=False)
    print("✅ Merge complete: merged_by_common_column.csv saved")
else:
    print("❌ No common columns found. Cannot merge.")
