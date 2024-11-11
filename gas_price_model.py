# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 2023

@author: ahmed

To estimate the price of natural gas on a given date, from historical prices
found in Nat_Gas.csv, using linear regression machine learning algorithm.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Reads data from the CSV file
data = pd.read_csv("Nat_Gas.csv")

# Converts date column to a datetime object
data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')

# Estimates price of gas on a given date


def estimate_price(input_date):
    # Gives end date based on input date (1 year later)
    end_date = input_date + timedelta(days=365)

    # Filters data for the correct time period
    relevant_data = data[(data['Dates'] >= input_date)
                         & (data['Dates'] <= end_date)]

    # Check if there is enough data for regression
    if relevant_data.empty or len(relevant_data) < 2:
        raise ValueError("Not enough data to perform linear regression")

    # Convert Dates to a numerical format (days since input_date)
    relevant_data['Days'] = (relevant_data['Dates'] - input_date).dt.days

    # Prepare the independent variable (X) and dependent variable (y)
    X = relevant_data['Days'].values.reshape(-1, 1)  # Days as the feature
    y = relevant_data['Price'].values                # Gas prices as the target

    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Predict the price for the input_date (which is day 0)
    estimated_price = model.predict([[0]])[0]

    return estimated_price


# Test:
# ask user to input date and convert to the correct format for calculation
input_date = input("Enter date:")
input_date = pd.to_datetime(input_date, format='%m/%d/%y')

# calculates and prints estimated gas price
test_price = estimate_price(input_date)
print("Estimated price of natural gas is: " + input_date)
