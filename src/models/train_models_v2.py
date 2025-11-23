# train_models_v2.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# ===============================
# Paths and load processed data
# ===============================
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load cleaned / feature-engineered splits
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_fe.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_fe.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train_fe.csv")).squeeze()
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test_fe.csv")).squeeze()

print(f"✅ Loaded processed data for v2: {X_train.shape}, {X_test.shape}")

# ===============================
# Preprocessing (on processed X)
# ===============================
categorical_features = X_train.select_dtypes(include='object').columns.tolist()
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('pca', PCA(n_components=min(13, len(numerical_features) + len(categorical_features) - 1)))
])

X_train_pca = pipeline.fit_transform(X_train)
X_test_pca = pipeline.transform(X_test)

# ===============================
# Models and parameter grids
# ===============================
models_params = {
    'logistic_v2': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {'C':[0.01,0.1,1,10], 'solver':['liblinear','lbfgs']}
    },
    'decision_tree_v2': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth':[3,5,7,None], 'min_samples_split':[2,5,10]}
    },
    'random_forest_v2': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators':[50,100,200], 'max_depth':[None,5,7], 'min_samples_split':[2,5]}
    },
    'xgboost_v2': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {'n_estimators':[50,100,200], 'max_depth':[3,5,7], 'learning_rate':[0.01,0.1,0.2]}
    },
    'catboost_v2': {
        'model': CatBoostClassifier(verbose=False, random_state=42),
        'params': {'depth':[3,5,7], 'learning_rate':[0.01,0.1], 'iterations':[100,200]}
    }
}

# ===============================
# GridSearchCV and evaluation
# ===============================
for name, mp in models_params.items():
    print(f"Running GridSearchCV for {name}...")
    grid = GridSearchCV(mp['model'], mp['params'], cv=3, scoring='recall', n_jobs=-1)
    grid.fit(X_train_pca, y_train)
    
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_test_pca)
    acc = round(accuracy_score(y_test, y_pred),4)
    prec = round(precision_score(y_test, y_pred),4)
    rec = round(recall_score(y_test, y_pred),4)
    roc = round(roc_auc_score(y_test, y_pred),4)
    
    print(f"{name} -> Best Params: {grid.best_params_}")
    print(f"{name} -> Accuracy: {acc}, Precision: {prec}, Recall: {rec}, ROC_AUC: {roc}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()
    
    # Save best model + pipeline
    model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump({'model': best_model, 'pipeline': pipeline}, model_path)
    print(f"✅ Saved tuned {name} with pipeline at {model_path}\n")

print("All v2 models trained, tuned, and saved successfully.")
