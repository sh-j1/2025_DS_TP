import os
import pickle
import pandas as pd

from modeling.utils_split import get_stratified_kfold_splits

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict_all_folds(model_dir, data_path, output_dir):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['stress_level'])

    X = df.drop(columns=['stress_level'])
    y = df['stress_level']

    splits = get_stratified_kfold_splits(X, y, n_splits=5)

    os.makedirs(output_dir, exist_ok=True)

    for i, (_, val_idx) in enumerate(splits):
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        for model_type in ['logistic_regression', 'decision_tree']:
            model_path = os.path.join(model_dir, f"{model_type}_fold{i+1}.pkl")
            model = load_model(model_path)

            y_pred = model.predict(X_val)

            result_df = pd.DataFrame({
                'y_true': y_val.values,
                'y_pred': y_pred
            })

            result_df.to_csv(os.path.join(output_dir, f"{model_type}_fold{i+1}_predictions.csv"), index=False)
            print(f"[OK] {model_type}_fold{i+1} prediction result saved.")
            
