import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("USA_Housing.csv")

# Display basic info
print(type(df))
df.info(verbose=True)

# Statistical summary
df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])

# Column names
df.columns

# Pairplot
sns.pairplot(df)
plt.show()

# Price distribution
df['Price'].plot.hist(bins=25, figsize=(8,4))
plt.show()

df['Price'].plot.density()
plt.show()

# FIX 1: Drop Address BEFORE corr
df = df.drop("Address", axis=1)

# Correlation matrix
df.corr()

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, linewidths=2)
plt.show()

# Feature & target selection
l_column = list(df.columns)
len_feature = len(l_column)

X = df[l_column[0:len_feature-1]]
y = df[l_column[len_feature-1]]

print("Feature set size:", X.shape)
print("Variable set size:", y.shape)

X.head()
y.head()

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

print("Training feature set size:", X_train.shape)
print("Test feature set size:", X_test.shape)
print("Training variable set size:", y_train.shape)
print("Test variable set size:", y_test.shape)

# Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn import metrics

lm = LinearRegression()
lm.fit(X_train, y_train)

print("The intercept term of the linear model:", lm.intercept_)
print("The coefficients of the linear model:", lm.coef_)

# Coefficient DataFrame
cdf = pd.DataFrame(
    data=lm.coef_,
    index=X_train.columns,
    columns=["Coefficients"]
)

# Standard Error & t-statistics
n = X_train.shape[0]
k = X_train.shape[1]
dfN = n - k

train_pred = lm.predict(X_train)
train_error = np.square(train_pred - y_train)
sum_error = np.sum(train_error)

se = [0]*k
for i in range(k):
    r = (sum_error / dfN)
    r = r / np.sum(
        np.square(
            X_train[list(X_train.columns)[i]] -
            X_train[list(X_train.columns)[i]].mean()
        )
    )
    se[i] = np.sqrt(r)

cdf['Standard Error'] = se
cdf['t-statistic'] = cdf['Coefficients'] / cdf['Standard Error']
print(cdf)

# Feature importance order
print("Therefore, features arranged in the order of importance for predicting the house price")
print("-"*90)

l = list(cdf.sort_values('t-statistic', ascending=False).index)
print(' > \n'.join(l))

# Scatter plots (feature vs price)
from matplotlib import gridspec

fig = plt.figure(figsize=(18,10))
gs = gridspec.GridSpec(2,3)

for i in range(5):
    ax = plt.subplot(gs[i])
    ax.scatter(df[l[i]], df['Price'])
    ax.set_title(l[i] + " vs Price", fontsize=18)

plt.show()

# Model Evaluation
print("R-squared value of this fit:",
      round(metrics.r2_score(y_train, train_pred), 3))

predictions = lm.predict(X_test)

print("Type of the predicted object:", type(predictions))
print("Size of the predicted object:", predictions.shape)

# Actual vs Predicted
plt.figure(figsize=(10,7))
plt.title("Actual vs Predicted house prices", fontsize=22)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.scatter(y_test, predictions)
plt.show()

# Residual histogram
plt.figure(figsize=(10,7))
plt.title("Histogram of residuals")
sns.histplot(y_test - predictions, kde=True)
plt.show()

# Residuals vs predictions
plt.figure(figsize=(10,7))
plt.title("Residuals vs Predicted values")
plt.scatter(predictions, y_test - predictions)
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Error metrics
print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print("MSE:", metrics.mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print("R-squared value of predictions:",
      round(metrics.r2_score(y_test, predictions), 3))

# Min-Max calculation (as in lab)
min_val = np.min(predictions / 6000)
max_val = np.max(predictions / 12000)

print(min_val, max_val)

L = (100 - min_val) / (max_val - min_val)
plt.hist(L)
plt.show()
