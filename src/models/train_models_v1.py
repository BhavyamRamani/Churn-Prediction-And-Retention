# train_models_v1.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_fe.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_fe.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train_fe.csv")).squeeze()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test_fe.csv")).squeeze()
print(f"✅ Loaded processed data for v1: {X_train.shape}, {X_test.shape}")


# Identify categorical and numerical features from processed X
categorical_features = X_train.select_dtypes(include='object').columns.tolist()
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)


pipeline = Pipeline([
    ('preprocess', preprocessor),
    # keep n_components logic similar to before
    ('pca', PCA(n_components=min(13, len(numerical_features) + len(categorical_features) - 1)))
])

X_train_pca = pipeline.fit_transform(X_train)
X_test_pca = pipeline.transform(X_test)


models = {
    'logistic_v1': LogisticRegression(random_state=42, max_iter=1000),
    'decision_tree_v1': DecisionTreeClassifier(random_state=42),
    'random_forest_v1': RandomForestClassifier(random_state=42, n_estimators=100),
    'xgboost_v1': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'catboost_v1': CatBoostClassifier(verbose=False, random_state=42)
}


for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_pca, y_train)

    # Evaluate
    y_pred = model.predict(X_test_pca)
    acc = round(accuracy_score(y_test, y_pred), 4)
    prec = round(precision_score(y_test, y_pred), 4)
    rec = round(recall_score(y_test, y_pred), 4)
    roc = round(roc_auc_score(y_test, y_pred), 4)

    print(f"{name} -> Accuracy: {acc}, Precision: {prec}, Recall: {rec}, ROC_AUC: {roc}")

    # Save model + pipeline
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump({'model': model, 'pipeline': pipeline}, model_path)
    print(f"✅ Saved {name} with pipeline at {model_path}\n")

print("All v1 models trained and saved successfully.")
