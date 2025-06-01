import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def train_models(X_train, y_train):
    models = {}

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    models['logistic_regression'] = logreg

    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(X_train, y_train)
    models['decision_tree'] = tree

    return models

def save_model(model, model_name, output_dir='artifacts/models'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{model_name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
