# -*- coding: utf-8 -*-
"""
Hepatitis Dataset - Machine Learning Operations
Final Corrected Lab Code (Error-Free)

@author: admin
"""

# ===============================
# 1. Import Required Libraries
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer


# ===============================
# 2. Load Dataset
# ===============================

df = pd.read_csv("hepatitis.csv")

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDataset Shape:", df.shape)


# ===============================
# 3. Convert All Columns to Numeric
# ===============================

df = df.apply(pd.to_numeric, errors="coerce")

print("\nMissing values after converting '?' to NaN:")
print(df.isnull().sum())


# ===============================
# 4. Separate Features and Target
# ===============================
# Target column is assumed to be named 'target'
# (If your column name is different, change it here)

X = df.drop(columns=["target"])
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ===============================
# 5. Handle Missing Values (CORRECT WAY)
# ===============================

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


# ===============================
# 6. Train-Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# 7. Feature Scaling
# ===============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# 8. Logistic Regression Model
# ===============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ===============================
# 9. Prediction
# ===============================

y_pred = model.predict(X_test)


# ===============================
# 10. Model Evaluation
# ===============================

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# 11. Confusion Matrix Visualization
# ===============================

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
