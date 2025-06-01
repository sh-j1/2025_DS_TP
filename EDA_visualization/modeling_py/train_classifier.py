import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
param_dict={
    'logistic_regression':{
        0:{'max_iter':500},
        1:{'max_iter':1000},
        2:{'max_iter':1500},
        3:{'max_iter':2000},
        4:{'max_iter':2500}
    },
    'decision_tree':{
        0:{'max_depth':3},
        1:{'max_depth':4},
        2:{'max_depth':5},
        3:{'max_depth':6},
        4:{'max_depth':7}
    }
}

def train_models(X_train, y_train, fold_idx):
    models = {}
    logreg_params = param_dict['logistic_regression'].get(fold_idx,{})
    logreg = LogisticRegression(**logreg_params)
    logreg.fit(X_train, y_train)
    models['logistic_regression'] = logreg

    tree_params = param_dict['decision_tree'].get(fold_idx,{})
    tree = DecisionTreeClassifier(**tree_params)
    tree.fit(X_train, y_train)
    models['decision_tree'] = tree

    return models

def save_model(model, model_name, output_dir='artifacts/models'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{model_name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)