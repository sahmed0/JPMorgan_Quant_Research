# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:43:18 2023

@author: ahmed

Estimates probability of default of a loan.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline

# Load and preprocess data


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Replace zero values in specific columns with NaN for handling
    df = df.assign(
        income=df['income'].replace(0, np.nan),
        total_debt_outstanding=df['total_debt_outstanding'].replace(0, np.nan),
        loan_amt_outstanding=df['loan_amt_outstanding'].replace(0, np.nan)
    )

    # Drop rows with NaN values in these critical columns
    df = df.dropna(
        subset=['income', 'total_debt_outstanding', 'loan_amt_outstanding'])

    # Calculate additional features
    df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']
    df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

    return df


# Define features and target
features = ['credit_lines_outstanding', 'debt_to_income',
            'payment_to_income', 'years_employed', 'fico_score']

# Define logistic regression pipeline


def create_pipeline():
    model = LogisticRegression(
        random_state=0, solver='liblinear', tol=1e-5, max_iter=500)
    return Pipeline([
        ('logistic_regression', model)
    ])

# Model training and evaluation


def train_and_evaluate(df, features):
    X = df[features]
    y = df['default']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Initialize and train pipeline
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions and evaluation on test set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate accuracy and AUC
    accuracy = metrics.accuracy_score(y_test, y_pred)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc_score = metrics.auc(fpr, tpr)

    print("Model Coefficients:",
          pipeline.named_steps['logistic_regression'].coef_)
    print("Intercept:", pipeline.named_steps['logistic_regression'].intercept_)
    print("Accuracy:", accuracy)
    print("AUC:", auc_score)


# Run the entire process
file_path = 'Loan_Data.csv'
df = load_and_preprocess_data(file_path)
train_and_evaluate(df, features)
