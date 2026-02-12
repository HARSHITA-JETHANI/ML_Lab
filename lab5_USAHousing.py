# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 22:37:11 2026

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Applied to USA_Housing Dataset
Minimal Changes Version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ===============================
# Load NEW Dataset
# ===============================

df = pd.read_csv("lab1/USA_Housing.csv")

print("========== BASIC INFORMATION ==========")
print("\nShape of Dataset:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nData Types:\n", df.dtypes)

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


print("\n========== CORRELATION MATRIX ==========")
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ===============================
# DATA CLEANING
# ===============================

# Drop non-numeric column (Address)
if "Address" in df.columns:
    df.drop(columns=["Address"], inplace=True)

df.fillna(df.mean(numeric_only=True), inplace=True)


# ===============================
# Separate Features and Target
# ===============================

X = df.drop(columns=["Price"])   # Target changed
y = df["Price"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ===============================
# Feature Scaling
# ===============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ===============================
# Linear Regression Model
# ===============================

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed.")


# ===============================
# Prediction
# ===============================

y_pred = model.predict(X_test)


# ===============================
# Evaluation (Regression Metrics)
# ===============================

print("\n========== MODEL EVALUATION ==========")

print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))


# Optional: Plot Actual vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.show()
