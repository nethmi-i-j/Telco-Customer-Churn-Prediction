# Telco-Customer-Churn-Prediction
Complete end-to-end ML project

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn, built on the IBM Telco Customer Churn dataset. The pipeline covers EDA, preprocessing, model selection, hyperparameter tuning with Optuna, threshold tuning, and a deployed FastAPI service for real-time predictions.

## Overview

Customer churn is one of the most critical metrics for subscription-based businesses. This project builds a classification model to predict customer churn for a telecom company, using customer demographics, account information, and service subscriptions. The goal is to identify customers at risk of leaving so the business can take proactive retention action.

The final model is an XGBoost Classifier, tuned via Optuna, trained on a SMOTE-balanced dataset to handle class imbalance, and served through a FastAPI REST endpoint for real-time inference.

## Dataset

Source: IBM Telco Customer Churn dataset
Size: 7,043 customers, 21 columns (20 features + target)
Target: Churn (Yes/No)
Class distribution: Imbalanced — 5,174 "No" vs 1,869 "Yes"

Features span customer demographics (gender, senior citizen status, partner/dependents), account info (tenure, contract type, payment method, billing), and subscribed services (phone, internet, online security, streaming, etc.).

## Project Workflow

## Data Cleaning

Dropped customerID (non-predictive identifier)
Mapped SeniorCitizen from 0/1 to No/Yes for consistency with other categorical fields
Fixed 11 records where TotalCharges was stored as a blank string — these corresponded to customers with tenure = 0, so the blanks were replaced with 0.0 and the column cast to float

## Exploratory Data Analysis

Distribution plots for all categorical and numerical features
Churn breakdown against each categorical feature
Outlier check via boxplots (none found)
Correlation heatmap and VIF analysis on numerical features (flagged multicollinearity between tenure and TotalCharges, both VIF > 5)

## Preprocessing Pipeline (to prevent data leakage)

Numerical features: median imputation + standard scaling

Categorical features: most-frequent imputation + one-hot encoding

Combined via ColumnTransformer

## Model Selection

Compared Decision Tree, Random Forest, and XGBoost using 5-fold stratified cross-validation

Each fold pipeline included SMOTE oversampling (applied only on training folds, inside the pipeline, to avoid leakage)

Metrics tracked: Accuracy, F1, Precision, Recall, ROC-AUC

XGBoost performed best on F1 and was selected for tuning

## Hyperparameter Tuning

Used Optuna (50 trials) to optimize XGBoost over n_estimators, learning_rate, max_depth, subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda

Optimized for mean F1 score across the same 5-fold CV split

## Threshold Tuning

Evaluated precision/recall/F1 trade-offs across thresholds (0.25–0.50)

Selected threshold = 0.40 as the operating point — better suited to retention campaigns, where catching more potential churners (recall) is prioritized over precision

## Final Evaluation & Deployment

Confusion matrix and feature importance analysis on the test set

Model serialized with joblib

Wrapped in a FastAPI service for real-time inference (see Running the FastAPI Service)

## Key Results

Final tuned XGBoost performance on the held-out test set at threshold = 0.40:

Accuracy - 75.0%, Precision - 51.8%, Recall - 80.2%  F1 Score - 63.0%  ROC-AUC - 84.5%

Top insight: Contract type — particularly month-to-month contracts — is the strongest driver of churn, followed by tenure and internet service type.


Note: The default 0.5 threshold favors precision; 0.4 was chosen deliberately to maximize recall, since in a retention-campaign context the cost of missing an actual churner outweighs the cost of an unnecessary retention offer to a non-churner.

## Running the FastAPI Service

The trained pipeline (telco_customer_churn_xgboost_model.joblib) is loaded once at startup and served via a /customer_churn_prediction endpoint in main.py.

uvicorn main:app 

The API will be available at http://127.0.0.1:8000, with interactive Swagger docs at http://127.0.0.1:8000/docs.

Making Predictions via the API

Endpoint: POST /customer_churn_prediction

Request body:

json{
  "gender": "Female",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Bank transfer",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}

Response:

json{
  "label": 1,
  "result": "Churn"
}

## Model Details

Algorithm: XGBoost Classifier

Class imbalance handling: SMOTE (applied within the cross-validation pipeline to prevent leakage)

Hyperparameter tuning: Optuna, 50 trials, optimized for F1 score

Decision threshold: 0.40 (tuned for recall-oriented retention use case)

Serialization: joblib, saved as a full pipeline (preprocessing → SMOTE → model) so raw, unprocessed customer records can be passed directly to the API


## Tech Stack

Language: Python

Data handling: pandas, NumPy

Visualization: Matplotlib, Seaborn

Modeling: scikit-learn, XGBoost, imbalanced-learn

Hyperparameter optimization: Optuna

Statistics: statsmodels (VIF analysis)

Deployment: FastAPI, Uvicorn

Serialization: joblib



