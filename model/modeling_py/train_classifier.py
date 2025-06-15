"""
train_classifier.py

This script defines functions to train and save classification models using scikit-learn.
It provides two classifiers: Logistic Regression and Decision Tree. The script assumes that
the input data (X_train, y_train) is preprocessed and ready for model training (e.g., missing
values handled, categorical variables encoded, and features scaled as needed).

Dependencies:
- scikit-learn
- pickle
- os

Functions:
- train_models(X_train, y_train): Trains Logistic Regression and Decision Tree classifiers.
- save_model(model, model_name, output_dir): Saves a trained model as a pickle file.

Notes/Assumptions:
- Input features and labels must be compatible with scikit-learn estimators.
- Models are saved in the 'artifacts/models' directory by default.
- No data preprocessing is performed in this script; preprocessing should be done beforehand.
"""

import os  # For directory and file path operations
import pickle  # For serializing and saving Python objects
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression classifier
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree classifier

def train_models(X_train, y_train):
    models = {}  # Dictionary to store trained models

    logreg = LogisticRegression(max_iter=1000)  # Initialize Logistic Regression with 1000 max iterations
    logreg.fit(X_train, y_train)  # Train Logistic Regression on the training data
    models['logistic_regression'] = logreg  # Store trained Logistic Regression model

    tree = DecisionTreeClassifier(max_depth=5)  # Initialize Decision Tree with max depth 5
    tree.fit(X_train, y_train)  # Train Decision Tree on the training data
    models['decision_tree'] = tree  # Store trained Decision Tree model

    return models  # Return dictionary of trained models

def save_model(model, model_name, output_dir='artifacts/models'):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    path = os.path.join(output_dir, f'{model_name}.pkl')  # Build file path for the model
    with open(path, 'wb') as f:  # Open the file in write-binary mode
        pickle.dump(model, f)  # Serialize and save the model to disk
