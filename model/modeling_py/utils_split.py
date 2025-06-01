from sklearn.model_selection import StratifiedKFold
import pandas as pd

def get_stratified_kfold_splits(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))
