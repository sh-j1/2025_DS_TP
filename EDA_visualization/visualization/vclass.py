"""
This script provides exploratory data analysis (EDA) and visualization utilities for examining the relationships between various factors and student stress levels.
Dependencies:
    - pandas
    - matplotlib
    - seaborn
    - os
Functions:
    - EDA_relavant_statics(data_path: str, output_dir: str):
        Reads a CSV dataset containing psychological, physical, environmental, academic, and social features, along with a target variable 'stress_level'.
        Computes and visualizes:
            - The correlation between each feature group and stress_level.
            - The number of students under negative influence for each group, based on feature directionality and correlation.
            - Scatter and regression plots for environmental and social features against stress_level.
        Saves all plots to the specified output directory.
    - visualize_compare_variable_growingstress(data_path: str, output_dir: str, variables: list):
        For a given list of variables, generates boxplots comparing their distributions across different levels of 'growing_stress'.
        Saves the resulting plot to the specified output directory.
Preprocessing Steps:
    - Reads input data from a CSV file.
    - Assumes specific column names for psychological, physical, environmental, academic, social, and target variables.
    - Handles output directory creation if it does not exist.
Notes/Assumptions:
    - The input CSV must contain all required columns as specified in the feature lists.
    - The script assumes that higher values in some features indicate negative influence, while lower values in others do, based on their correlation with stress_level.
    - Plots are saved as PNG files in the output directory.
    - The script prints intermediate DataFrames and plot save confirmations for transparency.
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os

def EDA_relavant_statics(data_path: str, output_dir: str):
    # Read the CSV data into a DataFrame
    df = pd.read_csv(data_path)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Set the output path for the correlation plot
    output_path = os.path.join(output_dir, "feature_correlation.png")
    # Define feature groups
    psycho = ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression']
    physical = ['headache', 'blood_pressure', 'sleep_quality', 'breathing_problem']
    environment = ['noise_level', 'living_conditions', 'safety', 'basic_needs']
    academic = ['academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns']
    social = ['social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']
    target = 'stress_level'
    # Calculate correlation between each feature group and the target
    corr_psycho = df[psycho + [target]].corr()[target].drop(target)
    corr_pyhsical = df[physical + [target]].corr()[target].drop(target)
    corr_environment = df[environment + [target]].corr()[target].drop(target)
    corr_academic = df[academic + [target]].corr()[target].drop(target)
    corr_social = df[social + [target]].corr()[target].drop(target)
    # Create DataFrames for each group with correlation values
    df_psycho = pd.DataFrame({
        'feature': corr_psycho.index,
        'corr': corr_psycho.values,
        'department': 'psycho'
    })
    df_pyhsical = pd.DataFrame({
        'feature': corr_pyhsical.index,
        'corr': corr_pyhsical.values,
        'department': 'pyhsical'
    })
    df_environment = pd.DataFrame({
        'feature': corr_environment.index,
        'corr': corr_environment.values,
        'department': 'environment'
    })
    df_academic = pd.DataFrame({
        'feature': corr_academic.index,
        'corr': corr_academic.values,
        'department': 'academic'
    })
    df_social = pd.DataFrame({
        'feature': corr_social.index,
        'corr': corr_social.values,
        'department': 'social'
    })
    # Concatenate all correlation DataFrames
    df_cor = pd.concat([df_psycho, df_pyhsical, df_environment, df_academic, df_social])
    # Print the combined correlation DataFrame
    print(df_cor)
    # Plot the correlation barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_cor, x='corr', y='feature', hue='department', orient='h')
    plt.tight_layout()
    plt.savefig(output_path)
    # Calculate the number of students under negative influence for each group
    output_path = os.path.join(output_dir, "Num_Negative_factor.png")
    # For psycho and physical, higher values are negative (correlation positive)
    num_student_undermean_psycho = (df[psycho] > 0).all(axis=1).sum()
    num_student_undermean_physical = (df[physical] > 0).all(axis=1).sum()
    # For environment and academic, lower values are negative (correlation negative)
    num_student_undermean_environment = (df[environment] < 0).all(axis=1).sum()
    num_student_undermean_academic = (df[academic] < 0).all(axis=1).sum()
    # For social, higher values are negative
    num_student_undermean_social = (df[social] > 0).all(axis=1).sum()
    # Collect the counts and labels
    Num_negative = [num_student_undermean_psycho, num_student_undermean_physical, num_student_undermean_environment, num_student_undermean_academic, num_student_undermean_social]
    x_label = ['psycho', 'physical', 'environment', 'academic', 'social']
    # Create a DataFrame for plotting
    df_p1 = pd.DataFrame({
        'Factor': x_label,
        'Num_student': Num_negative
    })
    # Plot the number of students under negative influence
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_p1, x='Factor', y='Num_student')
    plt.title("Number of Students Negative Factor")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    # Plot scatter and regression plots for environment features vs stress_level
    output_path = os.path.join(output_dir, "Environment to Stress.png")
    num_f = len(environment)
    n_col = 2
    n_row = (num_f + n_col - 1) // n_col
    plt.figure(figsize=(5 * n_col, 4 * n_row))
    for idx, feature in enumerate(environment, 1):
        plt.subplot(n_row, n_col, idx)
        sns.scatterplot(data=df, x=feature, y=target)
        sns.regplot(data=df, x=feature, y=target, scatter=False, color='red')
        plt.xlabel(feature)
        plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(output_path)
    # Plot scatter and regression plots for social features vs stress_level
    output_path = os.path.join(output_dir, "Social to Stress.png")
    plt.figure(figsize=(5 * n_col, 4 * n_row))
    for idx, feature in enumerate(social, 1):
        plt.subplot(n_row, n_col, idx)
        sns.scatterplot(data=df, x=feature, y=target)
        sns.regplot(data=df, x=feature, y=target, scatter=False, color='red')
        plt.xlabel(feature)
        plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(output_path)

def visualize_compare_variable_growingstress(data_path: str, output_dir: str, variables: list):
    # Read the CSV data into a DataFrame
    df = pd.read_csv(data_path)
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Set the output path for the boxplot
    output_path = os.path.join(output_dir, "growing_stress_comparison.png")
    # Create subplots for each variable
    fig, axs = plt.subplots(nrows=len(variables), ncols=1, figsize=(8, 4 * len(variables)))
    # If only one variable, wrap axs in a list
    if len(variables) == 1:
        axs = [axs]
    # Plot boxplots for each variable grouped by 'growing_stress'
    for i, var in enumerate(variables):
        sns.boxplot(x='growing_stress', y=var, data=df, ax=axs[i], width=0.5, fliersize=3, boxprops=dict(alpha=0.9), linewidth=1.2)
        axs[i].set_title(f'{var} by growing_stress')
        axs[i].set_xlabel('growing_stress')
        axs[i].set_ylabel(var)
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Ok] Plot saved to {output_path}")
    plt.close()
