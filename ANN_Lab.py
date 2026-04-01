# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:29:55 2026

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load the dataset
df = pd.read_csv("E:/Harshi/6thSEM/ML_Lab/hepatitis.csv")

# 2. Preprocess the data
# Drop the 'ID' column as it is not a useful feature for prediction
df = df.drop('ID', axis=1)

# The dataset contains '?' for missing values. Replace them with NaN
df = df.replace('?', np.nan)

# Convert all columns to numeric type
df = df.apply(pd.to_numeric)

# Fill missing values (NaN) with the median of each column
df = df.fillna(df.median())

# The target column has values 1 and 2. 
# Neural networks work best with binary targets as 0 and 1.
df['target'] = df['target'].map({1: 0, 2: 1})

# Separate features (X) and target (y)
X = df.drop('target', axis=1).values
y = df['target'].values

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# 4. Scale the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build the ANN Model
model = Sequential()

# Hidden Layer 1 (input shape matches the 19 features)
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))

# Hidden Layer 2
model.add(Dense(8, activation='relu'))

# Output Layer (1 neuron with sigmoid for binary classification)
model.add(Dense(1, activation='sigmoid'))

# 6. Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print Model Summary
print("\n--- Model Summary ---")
model.summary()
print("---------------------\n")

# 7. Train the Model
# We use 50 epochs because the dataset is quite small (155 rows)
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1 # set to 0 to hide training output per epoch
)

# 8. Evaluate the Model
print("\n--- Model Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
print("Test Loss:", loss)

# 9. Make Predictions and format them
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

# 10. Metrics
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 11. Plot Training History
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()