# main.py — Complete Training Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ─── 1. LOAD DATA ───────────────────────────────────────────────────
df = pd.read_csv('data/employee_features.csv')
print("Data loaded. Shape:", df.shape)

target = 'perf_band'
X = df.drop(columns=[target, 'employee_id'])
y = df[target]

# ─── 2. SPLIT DATA ──────────────────────────────────────────────────
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=13
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─── 3. PREPROCESSING ───────────────────────────────────────────────
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  RobustScaler())
])
cat_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])
pre = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# ─── 4. MODEL: LOGISTIC REGRESSION (BASELINE) ───────────────────────
lr_pipe = Pipeline([('pre', pre), ('clf', LogisticRegression(
    max_iter=500, class_weight='balanced', random_state=13
))])
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)
print("\n=== Logistic Regression ===")
print(classification_report(y_test, lr_pred))

# ─── 5. MODEL: RANDOM FOREST (BETTER) ───────────────────────────────
rf_pipe = Pipeline([('pre', pre), ('clf', RandomForestClassifier(
    class_weight='balanced', random_state=13
))])

# GridSearch finds the best settings automatically
param_grid = {
    'clf__n_estimators': [200, 400],
    'clf__max_depth': [8, 12, None],
    'clf__min_samples_leaf': [1, 3, 5]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
gs = GridSearchCV(rf_pipe, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("\nBest settings:", gs.best_params_)
print("Best CV F1:", round(gs.best_score_, 3))

best_model = gs.best_estimator_
rf_pred = best_model.predict(X_test)

print("\n=== Random Forest (Tuned) ===")
print(classification_report(y_test, rf_pred))

# ─── 6. CONFUSION MATRIX ────────────────────────────────────────────
cm = confusion_matrix(y_test, rf_pred, labels=['Low', 'Medium', 'High'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix — Random Forest')
plt.savefig('outputs/confusion_matrix.png')
plt.show()

# ─── 7. FEATURE IMPORTANCE ──────────────────────────────────────────
rf_clf = best_model.named_steps['clf']
ohe_cols = best_model.named_steps['pre'].named_transformers_['cat']['ohe'].get_feature_names_out(cat_cols)
all_feature_names = list(num_cols) + list(ohe_cols)
importances = rf_clf.feature_importances_

feat_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
feat_df = feat_df.sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feat_df, palette='viridis')
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')
plt.show()

# ─── 8. SAVE MODEL ──────────────────────────────────────────────────
joblib.dump(best_model, 'models/employee_perf_model.pkl')
print("\nModel saved to models/employee_perf_model.pkl")

# ─── 9. PREDICT NEW EMPLOYEE ────────────────────────────────────────
sample = X_test.iloc[:1]   # take the first test employee
pred_band = best_model.predict(sample)[0]
pred_proba = best_model.predict_proba(sample)[0]
classes = best_model.classes_

print("\n=== SAMPLE PREDICTION ===")
print("Employee data:")
print(sample.to_string())
print(f"\nPredicted Performance Band: {pred_band}")
for c, p in zip(classes, pred_proba):
    print(f"  {c}: {round(p*100, 1)}%")