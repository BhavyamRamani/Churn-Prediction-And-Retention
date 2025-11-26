import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load FE data
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_fe.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_fe.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train_fe.csv"))["target"]
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test_fe.csv"))["target"]

print("Loaded FE shapes:", X_train.shape, X_test.shape)

# Define models
models = {
    "logistic_v1": LogisticRegression(max_iter=1000),
    "decision_tree_v1": DecisionTreeClassifier(),
    "random_forest_v1": RandomForestClassifier(n_estimators=100),
    "xgboost_v1": XGBClassifier(eval_metric='logloss'),
    "catboost_v1": CatBoostClassifier(verbose=False)
}

# Train and evaluate
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    print(f"{name} -> Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, ROC_AUC={roc:.4f}")

    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
    print(f"Saved {name}.pkl")
