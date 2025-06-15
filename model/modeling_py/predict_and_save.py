"""
This script loads trained machine learning models for each fold of a stratified K-fold cross-validation,
applies them to the corresponding validation splits, and saves the prediction results to CSV files.
Preprocessing Steps:
- Reads a CSV dataset from the specified path.
- Drops rows with missing values in the 'stress_level' column.
- Splits the data into features (X) and target (y).
- Generates stratified K-fold splits to ensure balanced class distribution across folds.
Dependencies:
- pandas for data manipulation and CSV I/O.
- pickle for loading serialized model objects.
- os for file and directory operations.
- modeling.utils_split.get_stratified_kfold_splits for generating stratified splits.
Notes/Assumptions:
- The dataset must contain a 'stress_level' column as the target variable.
- Pre-trained models for each fold and model type ('logistic_regression', 'decision_tree') must exist in the specified model directory, named as '{model_type}_fold{fold_number}.pkl'.
- The script saves prediction results for each fold and model type as '{model_type}_fold{fold_number}_predictions.csv' in the output directory.
"""

import os  # Import os for file and directory operations
import pickle  # Import pickle for loading serialized model objects
import pandas as pd  # Import pandas for data manipulation and CSV I/O

from modeling.utils_split import get_stratified_kfold_splits  # Import function to generate stratified K-fold splits

def load_model(path):
    # Open the model file in binary read mode
    with open(path, 'rb') as f:
        # Load and return the model object from the file
        return pickle.load(f)

def predict_all_folds(model_dir, data_path, output_dir):
    # Read the dataset from the specified CSV file path
    df = pd.read_csv(data_path)
    # Drop rows where the 'stress_level' column has missing values
    df = df.dropna(subset=['stress_level'])

    # Separate features (X) by dropping the target column
    X = df.drop(columns=['stress_level'])
    # Extract the target variable (y)
    y = df['stress_level']

    # Generate stratified K-fold splits (returns list of (train_idx, val_idx) tuples)
    splits = get_stratified_kfold_splits(X, y, n_splits=5)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each fold (i: fold index, val_idx: validation indices)
    for i, (_, val_idx) in enumerate(splits):
        # Select validation features and labels for this fold
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Iterate over each model type
        for model_type in ['logistic_regression', 'decision_tree']:
            # Construct the path to the saved model for this fold and type
            model_path = os.path.join(model_dir, f"{model_type}_fold{i+1}.pkl")
            # Load the trained model from file
            model = load_model(model_path)

            # Predict the labels for the validation set
            y_pred = model.predict(X_val)

            # Create a DataFrame with true and predicted labels
            result_df = pd.DataFrame({
                'y_true': y_val.values,
                'y_pred': y_pred
            })

            # Save the prediction results to a CSV file in the output directory
            result_df.to_csv(os.path.join(output_dir, f"{model_type}_fold{i+1}_predictions.csv"), index=False)
            # Print a confirmation message
            print(f"[OK] {model_type}_fold{i+1} prediction result saved.")
