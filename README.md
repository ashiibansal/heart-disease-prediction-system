# ❤️ Heart Disease Prediction Using Machine Learning

A Streamlit-based web application for **binary heart disease prediction** using a comparative machine learning pipeline built on the **Heart Statlog Cleveland Hungary** dataset.

This project combines **data preprocessing, feature selection, model comparison, and interactive deployment** into one interface. The system allows a user to enter patient parameters, generate a prediction, compare multiple models, and explore evaluation plots such as ROC curves, confusion matrices, leaderboard tables, and feature importance.

---

## 📌 Project Overview

Heart disease remains one of the leading causes of mortality worldwide. This project aims to support **early risk screening** using machine learning models trained on clinical features that are routinely collected in cardiovascular assessment.

The app is designed around a **comparative modelling workflow**, where multiple classifiers are trained and evaluated under standardized preprocessing conditions. The interface highlights the **top-performing models first** while still exposing the **full 10-model comparison** for academic analysis and report screenshots.

---

## 🚀 Features

- Interactive **Streamlit web interface**
- **Patient-wise prediction** from clinical input parameters
- **Top 3 / Top 5 primary view** for the best-performing models
- **Full 10-model leaderboard** in the expanded comparison section
- **User-selectable plots** so multiple models can be overlaid without clutter
- **Confusion matrices** for selected models
- **ROC curve comparison**
- **Precision / Recall / F1-score plots**
- **Probability distribution plots**
- **Feature importance visualisation** for tree/boosting models
- Clean and modern **dark dashboard UI**

---

## 🧠 Models Used

The project compares the following 10 machine learning classifiers:

1. Logistic Regression  
2. Naive Bayes  
3. K-Nearest Neighbors (KNN)  
4. Support Vector Machine (SVM)  
5. Decision Tree  
6. Random Forest  
7. XGBoost  
8. CatBoost  
9. AdaBoost  
10. Gradient Boosting  

---

## 📊 Dataset

**Dataset:** `heart_statlog_cleveland_hungary_final.csv`

This merged dataset contains **1,190 patient records** and **12 clinical features**, derived from widely used heart disease datasets.

### Selected Features Used in the App

The final deployed models use the following 5 most correlated features:

- `chest pain type`
- `ST slope`
- `oldpeak`
- `exercise angina`
- `max heart rate`

### Target Variable

- `target`
  - `0` → No heart disease
  - `1` → Heart disease present

---

## 🧪 Preprocessing Pipeline

The application follows a notebook-aligned preprocessing strategy:

### Data Cleaning
- Zero values in `cholesterol` are treated as physiologically invalid
- These values are replaced with the **median**

### Outlier Handling
- Numeric columns are clipped using **IQR-based outlier capping**

### Feature Transformation
A `ColumnTransformer` is used for preprocessing:

- **StandardScaler** → `oldpeak`, `max heart rate`
- **OneHotEncoder** → `chest pain type`, `ST slope`
- **Passthrough** → `exercise angina`

### Train-Test Split
- `75%` training
- `25%` testing
- `random_state = 42`

---

## 🏆 Performance Summary

Based on the comparative analysis, the approximate test accuracies are:

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | KNN | 90.3% |
| 2 | XGBoost | 87.2% |
| 2 | CatBoost | 87.2% |
| 4 | Random Forest | 86.6% |
| 5 | Gradient Boosting | 86.2% |
| 6 | Decision Tree | 84.2% |
| 6 | AdaBoost | 84.2% |
| 8 | Logistic Regression | 83.5% |
| 9 | Naive Bayes | 82.9% |
| 9 | SVM | 82.9% |

> KNN achieved the highest overall performance in this study, while ensemble methods showed strong and stable results across multiple metrics.

---

## 🖥️ App Interface

The Streamlit app is structured into two layers:

### 1. Primary View
This view focuses on the **top-performing shortlist** so that the interface stays clean and readable.

It includes:
- patient parameter input
- primary model prediction
- confidence and probability output
- top model comparison

### 2. Expanded Comparison View
This section exposes the **full 10-model comparison** for academic and analytical use.

It includes:
- full leaderboard
- full model comparison graph
- all-model prediction table
- selectable overlay plots
- feature importance panels

---

## 📈 Visualisations Included

The dashboard supports:

- **Confusion Matrix**
- **ROC Curve**
- **Precision / Recall / F1-score**
- **Accuracy leaderboard**
- **Probability distribution comparison**
- **Feature importance plots**
- **Model-wise prediction comparison for the same patient input**

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **XGBoost**
- **CatBoost**

---

## 📂 Project Structure

```bash
Heart-Disease-Prediction/
│
├── app.py
├── heart_statlog_cleveland_hungary_final.csv
├── README.md
└── assets/
    ├── dashboard.png
    ├── prediction_output.png
    ├── leaderboard.png
    └── roc_overlay.png
```

## Example Input Parameters

The app accepts the following patient inputs:

Chest Pain Type
Typical Angina
Atypical Angina
Non-Anginal Pain
Asymptomatic
ST Slope
Upsloping
Flat
Downsloping
Oldpeak
ST depression induced by exercise
Exercise-Induced Angina
Yes / No
Max Heart Rate
Numeric slider input

---
⚠️ Disclaimer

This application is built for educational and screening purposes only.
It is not a substitute for professional medical diagnosis, treatment, or clinical decision-making.
---

📚 References
UCI Machine Learning Repository — Heart Disease Dataset
Scikit-learn documentation
XGBoost documentation
CatBoost documentation

