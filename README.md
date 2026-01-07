# Customer Churn Prediction and Retention Analysis

## Overview
This project focuses on analyzing and predicting customer churn using the Telco Customer Churn dataset. It implements an end-to-end data science pipeline covering data preprocessing, feature engineering, predictive modeling, explainability, clustering, survival analysis, and interactive Streamlit applications for churn prediction.

The repository supports both experimentation through modular scripts and interactive exploration through deployed apps.

---

## Dataset
The project uses the **Telco Customer Churn dataset**, containing customer demographics, service usage, billing information, and churn labels.

---

## Project Structure
```
Churn-Prediction-And-Retention/
│
├── src/
│   ├── data/
│   │   ├── preprocess.py              # Data cleaning and preprocessing
│   │   ├── feature_engineering.py     # Feature engineering logic
│   │   ├── clustering.py              # Customer segmentation using clustering
│   │   ├── survival_analysis.py       # Time-to-event churn analysis
│   │   ├── shap_analysis.py            # SHAP-based explainability
│   │   ├── model_selector.py           # Model selection utilities
│   │   └── registry.py                # Model registry helpers
│   │
│   ├── models/
│   │   ├── train_models_v1.py          # Baseline model training
│   │   ├── train_models_v2.py          # Extended model training pipeline
│   │   └── tune.py                     # Hyperparameter tuning
│   │
│   └── config/
│       └── config.yaml                 # Configuration file
│
├── my_app.py                           # Streamlit app (Logistic Regression + PCA)
├── catboost_app.py                    # Streamlit app (CatBoost model)
├── models/                            # Saved trained models
├── Screenshots/                       # App screenshots
├── .dvc/                              # Data Version Control configuration
├── Dockerfile                         # Docker setup
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Analysis Performed

### Data Preprocessing
- Cleaning and handling missing values
- Encoding categorical variables
- Scaling numerical features

### Feature Engineering
- Creation of derived features for modeling
- Preparation of datasets for multiple analysis pipelines

### Predictive Modeling
- Training churn prediction models using:
  - Logistic Regression
  - CatBoost
- Hyperparameter tuning and model comparison

### Explainability
- SHAP-based feature importance analysis to interpret model predictions

### Customer Segmentation
- Clustering techniques to identify groups of customers with similar behavior

### Survival Analysis
- Time-to-event analysis to study churn risk across customer lifetime
- Generation of survival summaries

---

## Streamlit Applications

### Logistic Regression + PCA App (`my_app.py`)
- Interactive Streamlit application
- Applies preprocessing, PCA, and Logistic Regression
- Visualizes customers in reduced PCA space
- Predicts churn probability for user-provided inputs

Run:
```bash
streamlit run my_app.py
```

### CatBoost Churn Prediction App (`catboost_app.py`)
- Streamlit application using CatBoostClassifier
- Accepts customer inputs and predicts churn
- Designed for interactive churn scoring

Run:
```bash
streamlit run catboost_app.py
```

---

## Docker Support
A Dockerfile is included to containerize the project environment and ensure reproducibility.

---

## Tools and Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- CatBoost
- SHAP
- Lifelines (survival analysis)
- Streamlit
- Docker
- DVC

---

## Purpose
This project provides a comprehensive analytical framework for customer churn analysis, combining predictive modeling with explainability, segmentation, survival insights, and interactive applications for practical use.

---

## Author
Bhavyam Ramani

---

## License
This project is licensed under the MIT License.
