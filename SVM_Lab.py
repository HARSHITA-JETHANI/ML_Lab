# -*- coding: utf-8 -*-
"""
Hepatitis Dataset - SVM Classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("hepatitis.csv")

df.replace("?", np.nan, inplace=True)

df = df.apply(pd.to_numeric, errors="coerce")


df.fillna(df.mean(), inplace=True)

if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)


X = df.drop(columns=["target"])
y = df["target"]


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)




model = SVC(kernel='rbf')

model.fit(X_train, y_train)

print("\nSVM Model Training Completed.")


y_pred = model.predict(X_test)


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
plt.title("Confusion Matrix - SVM")
plt.show()
