# -*- coding: utf-8 -*-

"""
Created on Tue Dec 26 23:45:49 2023

@author: ahmed

Calculates the probability of default, using maximum likelihood estimation.

"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data
df = pd.read_csv('Loan_Data.csv')

# Extract 'default' and 'fico_score' columns
x = df['default'].to_list()
y = df['fico_score'].to_list()
n = len(x)

# Initialize cumulative counts for default and total per FICO score level
default = [0] * 851
total = [0] * 851

# Fill cumulative arrays based on FICO score
for i in range(n):
    y[i] = int(y[i])
    default[y[i] - 300] += x[i]
    total[y[i] - 300] += 1

for i in range(1, 551):
    default[i] += default[i - 1]
    total[i] += total[i - 1]

# Define a function to compute the negative log-likelihood


def negative_log_likelihood(points):
    """
    Computes the negative log-likelihood for given segmentation points.
    Points are FICO score thresholds for segmentation.
    """
    log_likelihood_sum = 0
    prev_point = 0
    for point in points:
        point = int(round(point))  # Ensure point is an integer
        prev_point = int(prev_point)  # Ensure prev_point is an integer
        if point <= prev_point or point >= 551:
            return np.inf  # Ensures point order and bounds
        n_segment = total[point] - total[prev_point]
        k_segment = default[point] - default[prev_point]
        p = k_segment / n_segment if n_segment > 0 else 0
        # Log-likelihood calculation
        if p > 0 and p < 1:
            log_likelihood_sum += k_segment * \
                np.log(p) + (n_segment - k_segment) * np.log(1 - p)
        prev_point = point
    return -log_likelihood_sum  # Return negative because we want to maximize


# Initial guess for segmentation points (e.g., 10 evenly spaced points in FICO range)
initial_guess = np.linspace(50, 500, 10, dtype=int)  # Adjust if needed

# Constraints: Points must be in increasing order and within valid FICO score range
bounds = [(1, 550)] * len(initial_guess)

# Minimize the negative log-likelihood
result = minimize(negative_log_likelihood, initial_guess,
                  bounds=bounds, method='L-BFGS-B')

# Print the results
print("Optimal log-likelihood:", -result.fun)
print("Optimal segmentation points:", [
      int(round(point)) + 300 for point in result.x])
