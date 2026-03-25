# -*- coding: utf-8 -*-
"""
K-Means Clustering on Loan Dataset
"""

# ===============================
# 1. Import Libraries
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans


# ===============================
# 2. Load Dataset
# ===============================

df = pd.read_csv("E:/Harshi/6thSEM/ML_Lab/loan_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# ===============================
# 3. Data Preprocessing
# ===============================

# Handle missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

# Drop ID column if exists
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)


# ===============================
# 4. Feature Selection
# ===============================

# Use all features for clustering
X = df


# ===============================
# 5. Feature Scaling
# ===============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# 6. Elbow Method
# ===============================

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# ===============================
# 7. Apply K-Means
# ===============================

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

df["Cluster"] = y_kmeans

print("\nClustered Data:")
print(df.head())


# ===============================
# 8. Visualization
# ===============================

plt.figure(figsize=(6,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering - Loan Dataset")
plt.show()


# ===============================
# 9. Cluster Analysis
# ===============================

print("\nCluster-wise Mean Values:")
print(df.groupby("Cluster").mean())