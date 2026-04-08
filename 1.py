# ===============================
# 📌 IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===============================
# 📌 CHECK FILES IN DIRECTORY
# ===============================
print("Files in folder:", os.listdir())

# ===============================
# 📌 LOAD DATASET (FIX PATH HERE)
# ===============================
df = pd.read_csv(r"C:\Users\Administrator\Downloads\train_u6lujuX_CVtuZ9i.csv")

# ===============================
# 📌 PREPROCESSING
# ===============================
df.fillna({
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': df['Dependents'].mode()[0],
    'Self_Employed': df['Self_Employed'].mode()[0],
    'LoanAmount': df['LoanAmount'].median(),
    'Loan_Amount_Term': df['Loan_Amount_Term'].mode()[0],
    'Credit_History': df['Credit_History'].mode()[0]
}, inplace=True)

df.drop('Loan_ID', axis=1, inplace=True)

# Encoding
le = LabelEncoder()
cols = ['Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'Property_Area', 'Loan_Status']

for col in cols:
    df[col] = le.fit_transform(df[col])

# ===============================
# 📊 VISUALIZATION
# ===============================
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Status Distribution")
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# ===============================
# 📌 MODEL
# ===============================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ===============================
# 📈 RESULTS
# ===============================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 🌳 TREE VISUALIZATION
# ===============================
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()