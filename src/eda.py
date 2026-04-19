# eda.py — Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/employee_features.csv')

# 1. Basic info
print(df.shape)        # how many rows and columns
print(df.head())       # first 5 rows
print(df.info())       # data types
print(df.describe())   # min, max, average of each column

# 2. Check for missing values
print(df.isnull().sum())

# 3. How many employees in each performance band?
print(df['perf_band'].value_counts())

# 4. Visualize class balance
plt.figure(figsize=(7, 4))
sns.countplot(x='perf_band', data=df, palette='Set2')
plt.title('Performance Band Distribution')
plt.savefig('outputs/class_balance.png')
plt.show()

# 5. How does training hours relate to performance?
plt.figure(figsize=(8, 5))
sns.boxplot(x='perf_band', y='training_hours', data=df, palette='Set2')
plt.title('Training Hours by Performance Band')
plt.savefig('outputs/training_vs_perf.png')
plt.show()

# 6. How does on-time delivery relate to performance?
plt.figure(figsize=(8, 5))
sns.boxplot(x='perf_band', y='on_time_delivery_rate', data=df, palette='Set3')
plt.title('On-Time Delivery Rate by Performance Band')
plt.savefig('outputs/delivery_vs_perf.png')
plt.show()

# 7. Heatmap of correlations (numeric columns only)
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('outputs/correlation_heatmap.png')
plt.show()
