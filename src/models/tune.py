import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load FE datasets
X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train_fe.csv")
X_test  = pd.read_csv(f"{PROCESSED_DIR}/X_test_fe.csv")
y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train_fe.csv").squeeze()
y_test  = pd.read_csv(f"{PROCESSED_DIR}/y_test_fe.csv").squeeze()

print("Loaded FE dataset:", X_train.shape, X_test.shape)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n {name} Report:")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

# ---------- Logistic Regression ----------
log_reg = LogisticRegression(max_iter=500, solver="liblinear")
log_reg_params = {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]}

grid_lr = GridSearchCV(log_reg, log_reg_params, cv=5, scoring="f1", n_jobs=-1, verbose=1)
grid_lr.fit(X_train, y_train)

best_lr = grid_lr.best_estimator_
metrics_lr = evaluate_model("Logistic_v3", best_lr, X_test, y_test)

joblib.dump(best_lr, f"{MODEL_DIR}/logistic_v3.pkl")
print("Saved logistic_v3.pkl")

# ---------- Random Forest ----------
rf = RandomForestClassifier(random_state=42)
rf_params = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10]
}

grid_rf = RandomizedSearchCV(rf, rf_params, cv=5, scoring="f1", n_jobs=-1, n_iter=5, verbose=1, random_state=42)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
metrics_rf = evaluate_model("RandomForest_v3", best_rf, X_test, y_test)

joblib.dump(best_rf, f"{MODEL_DIR}/rf_v3.pkl")
print("Saved rf_v3.pkl")

# ---------- XGBoost ----------
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0]
}

grid_xgb = RandomizedSearchCV(xgb, xgb_params, cv=5, scoring="f1", n_jobs=-1, n_iter=5, verbose=1, random_state=42)
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
metrics_xgb = evaluate_model("XGBoost_v3", best_xgb, X_test, y_test)

joblib.dump(best_xgb, f"{MODEL_DIR}/xgb_v3.pkl")
print("Saved xgb_v3.pkl")

# ---------- Compare ----------
results = {
    "Logistic_v3": metrics_lr,
    "RandomForest_v3": metrics_rf,
    "XGBoost_v3": metrics_xgb,
}

print("\nFinal Comparison:")
for m, res in results.items():
    print(m, res)

best_model_name = max(results, key=lambda x: results[x]["f1"])

file_map = {
    "Logistic_v3": "logistic_v3.pkl",
    "RandomForest_v3": "rf_v3.pkl",
    "XGBoost_v3": "xgb_v3.pkl",
}

best_file = file_map[best_model_name]
joblib.dump(joblib.load(f"{MODEL_DIR}/{best_file}"), f"{MODEL_DIR}/best_model.pkl")

print(f"\n Best model saved → {best_model_name} as best_model.pkl")
