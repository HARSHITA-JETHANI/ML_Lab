# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 22:25:24 2026

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Hepatitis Dataset - Logistic Regression Full Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ===============================
# LOAD DATASET
# ===============================

df = pd.read_csv("hepatitis.csv")

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")


# ===============================
# BASIC INFORMATION
# ===============================

print("========== BASIC INFORMATION ==========")
print("\nShape of Dataset:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nData Types:\n", df.dtypes)

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== CLASS DISTRIBUTION ==========")
print(df["target"].value_counts())


# Visualize class distribution
plt.figure(figsize=(5,4))
sns.countplot(x=df["target"])
plt.title("Class Distribution (Live vs Die)")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.show()


print("\n========== CORRELATION MATRIX ==========")
correlation_matrix = df.corr()
print(correlation_matrix)

plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ===============================
# DATA CLEANING
# ===============================

df.fillna(df.mean(), inplace=True)

if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)

print("\nDataset Shape after dropping ID:", df.shape)


# ===============================
# SEPARATE FEATURES & TARGET
# ===============================

X = df.drop(columns=["target"])
y = df["target"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ===============================
# ENCODING TARGET
# ===============================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Encoded Target Classes:", np.unique(y))


# ===============================
# FEATURE SCALING
# ===============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Applied Successfully.")
print("Shape after scaling:", X_scaled.shape)


# ===============================
# TRAIN-TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# ===============================
# LOGISTIC REGRESSION MODEL
# ===============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel Training Completed.")


# ===============================
# PREDICTION
# ===============================

y_pred = model.predict(X_test)


# ===============================
# MODEL EVALUATION
# ===============================

print("\n========== MODEL EVALUATION ==========")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
