# preprocessing.py — Load data and prepare it for ML

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ─── STEP 1: LOAD THE DATA (this was missing!) ──────────────────────
df = pd.read_csv('data/employee_features.csv')
print("Data loaded successfully! Shape:", df.shape)
print(df.head())

# ─── STEP 2: SEPARATE FEATURES AND TARGET ───────────────────────────
target = 'perf_band'
X = df.drop(columns=[target, 'employee_id'])  # input features
y = df[target]                                 # what we want to predict

print("\nFeatures shape:", X.shape)
print("Target value counts:\n", y.value_counts())

# ─── STEP 3: IDENTIFY COLUMN TYPES ──────────────────────────────────
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

print("\nNumeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# ─── STEP 4: BUILD PREPROCESSING PIPELINES ──────────────────────────
# For numbers: fill missing values → scale them
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# For categories: fill missing values → convert text to numbers
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both pipelines into one
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

print("\nPreprocessing pipeline built successfully!")
print("Ready to train the model.")