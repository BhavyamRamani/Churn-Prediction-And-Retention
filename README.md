````{"variant":"standard","title":"Churn Analysis Project README","id":"72651"}
🧠 Customer Churn Analysis

📋 Overview
This project focuses on predicting **customer churn** — i.e., identifying customers who are likely to stop using a company’s service. The goal is to build a machine learning pipeline that preprocesses raw customer data, trains various models, and evaluates their performance using key classification metrics.

The pipeline includes **data cleaning, encoding, scaling, feature engineering, model training, evaluation, and deployment preparation**. The final model can be used to identify at-risk customers and help the business improve retention strategies.

---

📂 Project Structure
```
churn-analysis/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Original dataset
│   ├── processed_data.csv                        # Cleaned dataset (after preprocessing)
│
├── src/
│   ├── preprocess.py                             # Data preprocessing script
│   ├── train.py                                  # Model training and evaluation script
│   ├── utils.py                                  # Helper functions (if any)
│
├── models/
│   ├── logistic_regression.pkl                   # Saved model example
│   ├── scaler.pkl                                # StandardScaler object
│
├── notebooks/
│   ├── EDA.ipynb                                 # Exploratory Data Analysis notebook
│
├── app/
│   ├── app_pca.py                                # Streamlit app for PCA & prediction
│
├── reports/
│   ├── churn_report.pdf                          # Final report
│   ├── figures/                                  # Graphs & plots (ROC, confusion matrix, etc.)
│
├── requirements.txt                              # Python dependencies
├── README.md                                     # Project documentation (this file)
└── LICENSE                                       # License information
```

---

## 🧰 Tech Stack
- **Language:** Python 3.10+
- **Libraries:**  
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `streamlit`
- **Modeling:** Logistic Regression, Random Forest, CatBoost, etc.
- **Visualization:** Matplotlib & Seaborn
- **Deployment (optional):** Streamlit

---

## 📊 Dataset Description
The dataset used in this project is the **Telco Customer Churn Dataset** from IBM Watson Analytics.

| Column | Description |
|--------|--------------|
| customerID | Unique ID for each customer |
| gender, SeniorCitizen, Partner, Dependents | Demographic information |
| tenure, MonthlyCharges, TotalCharges | Account & billing details |
| PhoneService, InternetService, Contract, PaymentMethod | Service information |
| Churn | Target variable (Yes = churned, No = retained) |

---

## 🧹 Data Preprocessing
All preprocessing steps are handled in **`src/preprocess.py`**:
- Handle invalid or missing categories (e.g., `EDUCATION`, `MARRIAGE`)
- Drop irrelevant features (like `customerID`)
- Encode categorical variables using **OneHotEncoder**
- Scale continuous features using **StandardScaler**
- Split the dataset into **train/test** sets

Example:
```bash
python src/preprocess.py
```

Output:
```
processed_data.csv saved to data/
```

---

## 🤖 Model Training
The model training and evaluation logic is in **`src/train.py`**:
- Loads preprocessed dataset  
- Trains multiple models (Logistic Regression, Random Forest, etc.)
- Evaluates performance using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- Saves the trained model and scaler as `.pkl` files


## 🎨 Visualization & PCA App
The **Streamlit app** (`app/app_pca.py`) visualizes the dataset in PCA space and allows live predictions.

Run locally:
```
streamlit run app/app_pca.py
```

This app:
- Displays PCA-reduced data visualization  
- Accepts user inputs for customer attributes  
- Predicts churn probability using the saved model  

---

## 📈 Results Summary
| Model | Accuracy | Recall | F1 Score | ROC-AUC |
|--------|-----------|---------|-----------|-----------|
| Logistic Regression | 0.75 | 0.71 | 0.82 | 0.85 |
| CatBoost | 0.68 | 0.52 | 0.92 | 0.88 |

**Key Insight:**  
The **Logistic Regression** model achieved strong recall, making it ideal when false negatives (missed churns) are costly.

---

## 🚀 Future Scope
- Add **SHAP/Feature Importance** visualization for model interpretability  
- Deploy as a **web dashboard** for marketing teams  
- Automate data refresh and retraining with new data  
- Integrate customer retention recommendations  

---

## 👤 Author
**Bhavyam Ramani
````

---
