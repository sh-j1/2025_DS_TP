"""
train_classifier.py

This script defines functions to train and save classification models using scikit-learn.
It supports Logistic Regression and Decision Tree classifiers, with hyperparameters that can be varied by fold index (useful for cross-validation).
The script does not include data preprocessing steps; it assumes that the input data (X_train, y_train) is already preprocessed and ready for modeling.

Dependencies:
- scikit-learn (sklearn)
- pickle
- os

Notes/Assumptions:
- The input features (X_train) and labels (y_train) must be preprocessed and compatible with scikit-learn estimators.
- The fold_idx parameter is used to select hyperparameters for each fold, supporting cross-validation workflows.
- Models are saved as pickle files in the specified output directory.
"""

import os  # For directory and file operations
import pickle  # For saving models to disk
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression classifier
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree classifier

# Dictionary containing hyperparameters for each model and fold index
param_dict = {
    'logistic_regression': {
        0: {'max_iter': 500},
        1: {'max_iter': 1000},
        2: {'max_iter': 1500},
        3: {'max_iter': 2000},
        4: {'max_iter': 2500}
    },
    'decision_tree': {
        0: {'max_depth': 3},
        1: {'max_depth': 4},
        2: {'max_depth': 5},
        3: {'max_depth': 6},
        4: {'max_depth': 7}
    }
}

def train_models(X_train, y_train, fold_idx):
    """
    Train Logistic Regression and Decision Tree classifiers with parameters
    selected by fold index.

    Args:
        X_train: Training features (preprocessed).
        y_train: Training labels.
        fold_idx: Index to select hyperparameters for cross-validation.

    Returns:
        models: Dictionary with trained models.
    """
    models = {}  # Dictionary to store trained models

    # Get hyperparameters for Logistic Regression for this fold
    logreg_params = param_dict['logistic_regression'].get(fold_idx, {})
    # Initialize Logistic Regression with selected parameters
    logreg = LogisticRegression(**logreg_params)
    # Train Logistic Regression model
    logreg.fit(X_train, y_train)
    # Store trained Logistic Regression model
    models['logistic_regression'] = logreg

    # Get hyperparameters for Decision Tree for this fold
    tree_params = param_dict['decision_tree'].get(fold_idx, {})
    # Initialize Decision Tree with selected parameters
    tree = DecisionTreeClassifier(**tree_params)
    # Train Decision Tree model
    tree.fit(X_train, y_train)
    # Store trained Decision Tree model
    models['decision_tree'] = tree

    return models  # Return dictionary of trained models

def save_model(model, model_name, output_dir='artifacts/models'):
    """
    Save a trained model to disk as a pickle file.

    Args:
        model: Trained model object.
        model_name: Name for the saved model file.
        output_dir: Directory to save the model (default: 'artifacts/models').
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Build full path for the model file
    path = os.path.join(output_dir, f'{model_name}.pkl')
    # Open file in write-binary mode and save the model using pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)