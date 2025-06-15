"""
This script provides a set of functions for exploratory data analysis (EDA), data cleaning, and preprocessing
of a dataset, specifically tailored for stress level prediction tasks. It leverages popular Python libraries
such as pandas, numpy, matplotlib, seaborn, scipy, and scikit-learn.
Dependencies:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scipy
    - scikit-learn
    - os
Functions:
    - data_inspection(data_path: str, output_dir: str):
        Loads a dataset, prints basic information, and generates histograms for numerical columns.
        Saves the histogram plot to the specified output directory.
    - data_cleaning(data_path: str, output_dir: str):
        Handles missing data by removing rows with more than 30% missing values.
        For categorical columns, fills missing values with the mode.
        For numerical columns, interpolates missing values and fills any remaining with the median.
        Saves the cleaned dataset to the specified output directory.
    - preprocessing(data_path: str, output_dir: str):
        Removes outliers from numerical columns using z-score thresholding (|z| <= 3).
        Encodes binary categorical variables as 0/1 and ordinal variables using OrdinalEncoder.
        Standardizes numerical features (excluding the target 'stress_level') using StandardScaler.
        Saves the preprocessed dataset to the specified output directory.
Notes and Assumptions:
    - The script assumes the presence of a 'stress_level' column as the target variable.
    - Categorical columns with two unique values are mapped as 'Yes'->1 and 'No'->0.
    - Ordinal encoding is applied to columns with values ['Low', 'Medium', 'High'].
    - The script creates output directories if they do not exist.
    - All file paths must be provided as strings.
    - The script is designed for tabular data in CSV format.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns
import os
# Function for inspecting the dataset
def data_inspection(data_path: str, output_dir: str):
    print("-------------------------[DATA INSPECTION]----------------------------\n")
    df = pd.read_csv(data_path)  # Load the dataset from the given path
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    print(df.info())  # Print dataframe info (column types, non-null counts)
    print(df.head())  # Print the first few rows of the dataframe
    print(df.describe())  # Print summary statistics for numerical columns

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Get numerical columns
    fig = df[num_cols].hist(bins=10, figsize=(15, 10))  # Plot histograms for numerical columns
    plt.tight_layout()  # Adjust subplot spacing

    for ax, col in zip(fig.flatten(), num_cols):  # Set titles for each histogram
        ax.set_title(f"Histogram of {col}")
    output_path = os.path.join(output_dir, "BP_scatter_HIST.png")  # Output path for the plot
    plt.savefig(output_path)  # Save the histogram plot
    plt.close()  # Close the plot to free memory
    print("BP_scatter_HIST.png completely saved")  # Notify user of completion

# Function for cleaning the dataset
def data_cleaning(data_path: str, output_dir: str):
    print("-------------------------[DATA CLEANING]----------------------------\n")
    df = pd.read_csv(data_path)  # Load the dataset
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if needed
    output_path = os.path.join(output_dir, 'StressLevelDataset(cleaned).csv')  # Output path for cleaned data
    print(df.isna().sum())  # Print count of missing values per column
    missing_row = df.isna().mean(axis=1)  # Calculate fraction of missing values per row
    df_drop = df[missing_row <= 0.3]  # Drop rows with more than 30% missing values
    print(f"행 갯수:{len(df)},제거된 행 갯수:{(missing_row>0.3).sum()}")  # Print number of rows before and after dropping

    for col in df_drop.columns:  # Iterate over columns to handle missing values
        if df_drop[col].dtype == 'object':  # For categorical columns
            mode_val = df_drop[col].mode()[0]  # Get the mode (most frequent value)
            df_drop[col] = df_drop[col].fillna(mode_val)  # Fill missing with mode
        else:  # For numerical columns
            df_drop[col] = df_drop[col].interpolate(method='linear').round(1)  # Interpolate missing values
            df_drop[col] = df_drop[col].fillna(df_drop[col].median())  # Fill any remaining with median

    print(df_drop.info())  # Print info of cleaned dataframe
    print(df_drop.head())  # Print first few rows
    print(df_drop.describe())  # Print summary statistics

    df_drop.to_csv(output_path, index=False)  # Save cleaned dataframe to CSV

# Function for preprocessing the dataset
def preprocessing(data_path: str, output_dir: str):
    print("-------------------------[DATA PREPROCESSING]----------------------------\n")
    df = pd.read_csv(data_path)  # Load the dataset
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if needed
    output_path = os.path.join(output_dir, 'StressLevelDataset(final).csv')  # Output path for preprocessed data
    num_df = df.select_dtypes(include=['float64', 'int64'])  # Select numerical columns
    z_score = num_df.apply(zscore)  # Calculate z-scores for outlier detection
    df_z = df[(np.abs(z_score) <= 3).all(axis=1)]  # Remove rows with outliers (z-score > 3)
    category_cols = df_z.select_dtypes(include=['object', 'category']).columns  # Get categorical columns

    for v in category_cols:  # Encode categorical columns
        uni_val = df_z[v].unique()  # Get unique values
        if len(uni_val) == 2:  # If binary categorical
            df_z[v] = df_z[v].map({'Yes': 1, 'No': 0})  # Map 'Yes' to 1, 'No' to 0
        else:  # For ordinal categorical columns
            encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])  # Define ordinal encoder
            df_z[v] = encoder.fit_transform(df_z[[v]]).astype(int).ravel()  # Encode and flatten

    scaler = StandardScaler()  # Initialize standard scaler
    num_cols = df_z.select_dtypes(include=['float64', 'int64']).drop(columns=['stress_level']).columns  # Get numerical columns except target
    df_z[num_cols] = scaler.fit_transform(df_z[num_cols])  # Standardize numerical features

    print(df_z.info())  # Print info of preprocessed dataframe
    print(df_z.head())  # Print first few rows
    print(df_z.describe())  # Print summary statistics

    df_z.to_csv(output_path, index=False)  # Save preprocessed dataframe to CSV