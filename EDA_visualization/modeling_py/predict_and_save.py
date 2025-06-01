import os
import pickle
import pandas as pd

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict_all_folds_Train_Test(model_dir, Train_data_path, Test_data_path, output_dir, fold):
    os.makedirs(output_dir, exist_ok=True)
    df_train = pd.read_csv(Train_data_path)
    df_test = pd.read_csv(Test_data_path)
    X = df_train.drop(columns=['stress_level'])
    y = df_train['stress_level']
    X_t = df_test.drop(columns=['stress_level'])
    y_t = df_test['stress_level']

    fold_results=[]
    for model_type in ['logistic_regression', 'decision_tree']:
        model_path = os.path.join(model_dir, f"{model_type}_fold{fold}.pkl")
        model = load_model(model_path)
        y_pred = model.predict(X)
        Train_score = model.score(X,y)
        Test_score = model.score(X_t,y_t)
        result_df = pd.DataFrame({
            'y_true': y.values,
            'y_pred': y_pred
        })
        score_df = pd.DataFrame({
            'fold': [fold],
            'model_type': [model_type],
            'train_score': [Train_score],
            'test_score' : [Test_score]
        })
        fold_results.append(score_df)
        result_df.to_csv(os.path.join(output_dir, f"{model_type}_fold{fold}_predictions.csv"), index=False)
        print(f"[OK] {model_type}_fold{fold} prediction result saved.")
    return pd.concat(fold_results, ignore_index=True)
            