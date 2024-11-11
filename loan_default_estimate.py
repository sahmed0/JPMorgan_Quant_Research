# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:43:18 2023

@author: ahmed

"""

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd

# Read in loan data from a CSV file
df = pd.read_csv('Loan_Data_.csv')

# Define the variable features
features = ['credit_lines_outstanding', 'debt_to_income',
            'payment_to_income', 'years_employed', 'fico_score']

# Calculate the payment_to_income ratio
df['payment_to_income'] = df['loan_amt_outstanding'] / df['income']

# Calculate the debt_to_income ratio
df['debt_to_income'] = df['total_debt_outstanding'] / df['income']

clf = LogisticRegression(random_state=0, solver='liblinear',
                         tol=1e-5, max_iter=10000).fit(df[features], df['default'])
print(clf.coef_, clf.intercept_)

# Use the following code to check
y_pred = clf.predict(df[features])

fpr, tpr, thresholds = metrics.roc_curve(df['default'], y_pred)
print((1.0*(abs(df['default']-y_pred)).sum()) / len(df))
print(metrics.auc(fpr, tpr))
