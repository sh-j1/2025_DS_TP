import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns
import os

def data_inspection(data_path: str, output_dir: str):
    print("-------------------------[DATA INSPECTION]----------------------------\n")
    df = pd.read_csv(data_path)
    os.makedirs(output_dir,exist_ok=True)
    print(df.info())
    print(df.head())
    print(df.describe())

    num_cols = df.select_dtypes(include=['float64','int64']).columns
    fig = df[num_cols].hist(bins=10, figsize=(15,10))
    plt.tight_layout()
    
    for ax,col in zip(fig.flatten(), num_cols):
        ax.set_title(f"Histogram of {col}")
    output_path = os.path.join(output_dir,"BP_scatter_HIST.png")
    plt.savefig(output_path)
    plt.close()
    print("BP_scatter_HIST.png completely saved")

def data_cleaning(data_path: str, output_dir: str):
    print("-------------------------[DATA CLEANING]----------------------------\n")
    df = pd.read_csv(data_path)
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir, 'StressLevelDataset(cleaned).csv')
    print(df.isna().sum())
    missing_row = df.isna().mean(axis=1)
    df_drop = df[missing_row<=0.3]
    print(f"행 갯수:{len(df)},제거된 행 갯수:{(missing_row>0.3).sum()}")
    for col in df_drop.columns:
        if df_drop[col].dtype == 'object':
            mode_val = df_drop[col].mode()[0]
            df_drop[col] = df_drop[col].fillna(mode_val)
        else:
            df_drop[col]=df_drop[col].interpolate(method='linear').round(1)
            df_drop[col]=df_drop[col].fillna(df_drop[col].median())

    print(df_drop.info())
    print(df_drop.head())
    print(df_drop.describe())

    df_drop.to_csv(output_path, index=False)

def preprocessing(data_path:str, output_dir:str):
    print("-------------------------[DATA PREPROCESSING]----------------------------\n")
    df = pd.read_csv(data_path)
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir, 'StressLevelDataset(final).csv')
    num_df = df.select_dtypes(include=['float64','int64'])
    z_score = num_df.apply(zscore)
    df_z = df[(np.abs(z_score)<=3).all(axis=1)]
    category_cols = df_z.select_dtypes(include=['object','category']).columns

    for v in category_cols:
        uni_val = df_z[v].unique()
        if len(uni_val)==2:
            df_z[v]=df_z[v].map({'Yes':1,'No':0})
        else:
            encoder = OrdinalEncoder(categories=[['Low','Medium','High']])
            df_z[v] = encoder.fit_transform(df_z[[v]]).astype(int).ravel()
    scaler = StandardScaler()
    num_cols = df_z.select_dtypes(include=['float64','int64']).drop(columns=['stress_level']).columns
    df_z[num_cols]=scaler.fit_transform(df_z[num_cols])
    print(df_z.info())
    print(df_z.head())
    print(df_z.describe())
    df_z.to_csv(output_path, index=False)
    