# train_models_v2.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load FE final data
X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train_fe.csv")
X_test  = pd.read_csv(f"{PROCESSED_DIR}/X_test_fe.csv")
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train_fe.csv").squeeze()
y_test  = pd.read_csv(f"{PROCESSED_DIR}/y_test_fe.csv").squeeze()

print("Loaded FE data shapes →", X_train.shape, X_test.shape)

models_params = {
    'logistic_v2': {
        'model': LogisticRegression(max_iter=1000),
        'params': {'C':[0.01,0.1,1,10], 'solver':['liblinear','lbfgs']}
    },
    'decision_tree_v2': {
        'model': DecisionTreeClassifier(),
        'params': {'max_depth':[3,5,7,None], 'min_samples_split':[2,5,10]}
    },
    'random_forest_v2': {
        'model': RandomForestClassifier(),
        'params': {'n_estimators':[50,100,200], 'max_depth':[None,5,7], 'min_samples_split':[2,5]}
    },
    'xgboost_v2': {
        'model': XGBClassifier(eval_metric='logloss'),
        'params': {'n_estimators':[50,100,200], 'max_depth':[3,5,7], 'learning_rate':[0.01,0.1,0.2]}
    },
    'catboost_v2': {
        'model': CatBoostClassifier(verbose=False),
        'params': {'depth':[3,5,7], 'learning_rate':[0.01,0.1], 'iterations':[100,200]}
    }
}

for name, mp in models_params.items():
    print(f"\nRunning GridSearchCV for {name}...")

    grid = GridSearchCV(mp['model'], mp['params'], cv=3, scoring='recall', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_pred)

    print(f"{name} -> Best Params: {grid.best_params_}")
    print(f"{name} -> Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, ROC_AUC={roc:.4f}")

    joblib.dump(best_model, f"{MODEL_DIR}/{name}.pkl")
    print(f"Saved {name}.pkl")

print("\nAll v2 models trained successfully.")
