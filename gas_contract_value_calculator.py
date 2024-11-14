# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:09:52 2023

@author: ahmed

Calculates value of natural gas contracts for trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

# Load and process data
data = pd.read_csv("Nat_Gas.csv")
data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')
prices = data['Prices'].values
dates = data['Dates'].values

# Plotting
plt.plot(dates, prices, '-')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Natural Gas Prices')
plt.xticks(rotation=45)
plt.show()

# Calculate time in days from the start date
start_date = datetime(2020, 10, 31)
days_from_start = (data['Dates'] - pd.Timestamp(start_date)).dt.days

# Simple regression for the trend line
def simple_regression(x, y):
    xbar, ybar = np.mean(x), np.mean(y)
    slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    intercept = ybar - slope * xbar
    return slope, intercept

slope, intercept = simple_regression(days_from_start, prices)

# Plot trend
plt.plot(days_from_start, prices, label='Prices')
plt.plot(days_from_start, days_from_start * slope + intercept, label='Linear Trend')
plt.xlabel('Days from start date')
plt.ylabel('Price')
plt.title('Linear Trend of Monthly Input Prices')
plt.legend()
plt.show()

# Seasonal (annual) variation modeling
seasonal_residuals = prices - (days_from_start * slope + intercept)
sin_time = np.sin(2 * np.pi * days_from_start / 365)
cos_time = np.cos(2 * np.pi * days_from_start / 365)

def bilinear_regression(y, x1, x2):
    slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
    slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
    return slope1, slope2

slope1, slope2 = bilinear_regression(seasonal_residuals, sin_time, cos_time)
amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
shift = np.arctan2(slope2, slope1)

# Full price prediction using trend + seasonal component
def predict_price(days):
    return amplitude * np.sin(2 * np.pi * days / 365 + shift) + days * slope + intercept

# Continuous prediction for interpolation and plot
continuous_dates = pd.date_range(start=start_date, end='2024-09-30', freq='D')
predicted_prices = [predict_price((d - pd.Timestamp(start_date)).days) for d in continuous_dates]

# Plot predictions
plt.plot(continuous_dates, predicted_prices, label='Predicted Prices')
plt.plot(dates, prices, 'o', label='Observed Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Predicted Natural Gas Prices')
plt.legend()
plt.show()

# Contract value calculation
def calculate_contract_value(injection_dates, withdrawal_dates, purchase_prices, sale_prices,
                             injection_rate, withdrawal_rate, max_storage_volume, storage_cost_per_day):
    contract_value = 0
    for inj_date, with_date, purchase_price, sale_price in zip(injection_dates, withdrawal_dates, purchase_prices, sale_prices):
        storage_days = (with_date - inj_date).days
        storage_cost = storage_cost_per_day * storage_days * max_storage_volume
        stored_gas_value = (sale_price - purchase_price) * max_storage_volume
        contract_value += stored_gas_value - storage_cost
    return contract_value

# Example usage
injection_dates = [datetime(2023, 1, 15), datetime(2023, 5, 1), datetime(2023, 8, 15)]
withdrawal_dates = [datetime(2023, 4, 15), datetime(2023, 7, 1), datetime(2023, 12, 31)]
purchase_prices = [2.0, 2.5, 2.2]
sale_prices = [3.0, 3.2, 3.5]
injection_rate = 50000
withdrawal_rate = 70000
max_storage_volume = 1000000
storage_cost_per_day = 0.1

value = calculate_contract_value(
    injection_dates, withdrawal_dates, purchase_prices, sale_prices,
    injection_rate, withdrawal_rate, max_storage_volume, storage_cost_per_day
)
print(f"Contract Value: ${value:.2f}")
