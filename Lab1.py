import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("USA_Housing.csv")

print(df.head())
print(df.info(verbose=True))
print(df.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
print(df.columns)

sns.pairplot(df.sample(500))
plt.show()

df['Price'].plot.hist(bins=25, figsize=(8,4))
plt.show()

df['Price'].plot.density()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
