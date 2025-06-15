"""
This script provides a utility function for generating stratified K-fold splits for cross-validation.
It uses scikit-learn's StratifiedKFold to ensure that each fold maintains the same class distribution as the original dataset.

Preprocessing Steps:
- Assumes input features (X) are provided as a pandas DataFrame and target labels (y) as a pandas Series.
- No additional preprocessing is performed within this function.

Dependencies:
- pandas
- scikit-learn

Notes/Assumptions:
- The target variable y should be categorical or discrete for stratification to be meaningful.
- The function returns a list of (train_index, test_index) tuples for each fold.
"""

from sklearn.model_selection import StratifiedKFold  # Import StratifiedKFold for stratified splitting
import pandas as pd  # Import pandas for DataFrame and Series types

def get_stratified_kfold_splits(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42):
    # Initialize StratifiedKFold with the specified number of splits, shuffling, and random state
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Generate and return the list of (train_index, test_index) splits
    return list(skf.split(X, y))
