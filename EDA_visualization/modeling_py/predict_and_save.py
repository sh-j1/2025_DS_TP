"""
This script provides functionality to load trained machine learning models, perform predictions on both training and test datasets, and save the prediction results and evaluation scores for each fold and model type.
Dependencies:
    - os: For directory and file path operations.
    - pickle: For loading serialized model objects.
    - pandas: For data manipulation and CSV I/O.
Functions:
    - load_model(path): Loads a pickled model from the specified file path.
    - predict_all_folds_Train_Test(model_dir, Train_data_path, Test_data_path, output_dir, fold):
        - Loads training and test datasets from CSV files.
        - Separates features and target variable ('stress_level') for both datasets.
        - Iterates over two model types ('logistic_regression', 'decision_tree'), loading the corresponding model for the specified fold.
        - Generates predictions on the training data, computes accuracy scores for both training and test sets.
        - Saves prediction results for each model and fold as CSV files in the output directory.
        - Aggregates and returns a DataFrame containing train and test scores for each model type and fold.
Preprocessing Steps:
    - Assumes that both training and test datasets contain a 'stress_level' column as the target variable.
    - Drops the 'stress_level' column to obtain feature matrices for prediction.
Notes/Assumptions:
    - Model files are expected to be named in the format '{model_type}_fold{fold}.pkl' and located in the specified model directory.
    - The script assumes that the input CSV files and model files exist and are accessible.
    - Output prediction CSVs are saved with the naming convention '{model_type}_fold{fold}_predictions.csv' in the specified output directory.
    - The script prints a confirmation message after saving each prediction result.
"""

import os  # Import the os module for directory and file path operations
import pickle  # Import pickle for loading serialized model objects
import pandas as pd  # Import pandas for data manipulation and CSV I/O

def load_model(path):
    # Open the model file in binary read mode
    with open(path, 'rb') as f:
        # Load and return the pickled model object
        return pickle.load(f)

def predict_all_folds_Train_Test(model_dir, Train_data_path, Test_data_path, output_dir, fold):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Load the training dataset from CSV
    df_train = pd.read_csv(Train_data_path)
    # Load the test dataset from CSV
    df_test = pd.read_csv(Test_data_path)
    # Separate features from the target variable in the training set
    X = df_train.drop(columns=['stress_level'])
    y = df_train['stress_level']
    # Separate features from the target variable in the test set
    X_t = df_test.drop(columns=['stress_level'])
    y_t = df_test['stress_level']

    fold_results=[]  # Initialize a list to store results for each model type
    # Iterate over the two model types
    for model_type in ['logistic_regression', 'decision_tree']:
        # Construct the path to the model file for the current fold and model type
        model_path = os.path.join(model_dir, f"{model_type}_fold{fold}.pkl")
        # Load the trained model
        model = load_model(model_path)
        # Generate predictions on the training data
        y_pred = model.predict(X)
        # Calculate the accuracy score on the training data
        Train_score = model.score(X,y)
        # Calculate the accuracy score on the test data
        Test_score = model.score(X_t,y_t)
        # Create a DataFrame with true and predicted labels for the training data
        result_df = pd.DataFrame({
            'y_true': y.values,
            'y_pred': y_pred
        })
        # Create a DataFrame with the fold, model type, and scores
        score_df = pd.DataFrame({
            'fold': [fold],
            'model_type': [model_type],
            'train_score': [Train_score],
            'test_score' : [Test_score]
        })
        # Append the score DataFrame to the results list
        fold_results.append(score_df)
        # Save the prediction results to a CSV file
        result_df.to_csv(os.path.join(output_dir, f"{model_type}_fold{fold}_predictions.csv"), index=False)
        # Print a confirmation message
        print(f"[OK] {model_type}_fold{fold} prediction result saved.")
    # Concatenate all score DataFrames and return as a single DataFrame
    return pd.concat(fold_results, ignore_index=True)