# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 23:45:49 2023

@author: ahmed

Calculates the probability of default, using maximum likelihood estimation.

"""

import pandas as pd
import numpy as np

# Reads csv file
df = pd.read_csv('Loan_Data.csv')

# creates lists for default and fico score
x = df['default'].to_list()
y = df['fico_score'].to_list()
n = len(x)
print(len(x), len(y))

# creates lists that will be filled later
default = [0 for i in range(851)]
total = [0 for i in range(851)]

# fill above lists with values from CSV file
for i in range(n):
    y[i] = int(y[i])
    default[y[i]-300] += x[i]
    total[y[i]-300] += 1

for i in range(0, 551):
    default[i] += default[i-1]
    total[i] += total[i-1]

# defines a function to calculate the log-likelihood of default


def log_likelihood(n, k):
    p = k/n
    if (p == 0 or p == 1):
        return 0
    return k*np.log(p) + (n-k)*np.log(1-p)


# define 'dp' as 3D array to store log-likelihood values
# Stores number of iterations performed in 1st dimension of dp
r = 10
dp = [[[-10**18, 0] for i in range(551)] for j in range(r+1)]

for i in range(r+1):  # stores rank of observations in 2nd dimension
    for j in range(551):
        if (i == 0):
            dp[i][j][0] = 0
        else:
            # stores log-likelihood and index of previous observation in 3rd dimension
            for k in range(j):
                if (total[j] == total[k]):
                    continue
                if (i == 1):  # calculates log-likelihood
                    dp[i][j][0] = log_likelihood(total[j], default[j])
                else:
                    if (dp[i][j][0] < (dp[i-1][k][0] + log_likelihood(total[j]-total[k], default[j] - default[k]))):
                        dp[i][j][0] = log_likelihood(
                            total[j]-total[k], default[j]-default[k]) + dp[i-1][k][0]
                        dp[i][j][1] = k

# prints results
print(round(dp[r][550][0], 4))

k = 550
l = []
while r >= 0:   # makes list of observations to output
    l.append(k+300)
    k = dp[r][k][1]
    r -= 1

# prints observations used in calculation
print(l)
