# generate_data.py
# This file creates a fake HR dataset with 1000 employees

import pandas as pd
import numpy as np

# Set a seed so we get the same data every time we run
np.random.seed(42)

n = 1000  # number of employees

# --- Create each column ---
df = pd.DataFrame({
    'employee_id': range(1001, 1001 + n),

    'age': np.random.randint(22, 58, n),

    'gender': np.random.choice(['Male', 'Female'], n),

    'education': np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'],
        n, p=[0.15, 0.50, 0.28, 0.07]
    ),

    'department': np.random.choice(
        ['Engineering', 'Sales', 'HR', 'Finance', 'Marketing'],
        n
    ),

    'job_level': np.random.choice(
        ['Junior', 'Mid', 'Senior', 'Lead'],
        n, p=[0.35, 0.35, 0.20, 0.10]
    ),

    'experience_years': np.random.randint(0, 25, n),

    'salary_band': np.random.choice(['Low', 'Medium', 'High'], n, p=[0.30, 0.50, 0.20]),

    'training_hours': np.random.randint(0, 120, n),

    'projects_count': np.random.randint(1, 20, n),

    'on_time_delivery_rate': np.round(np.random.uniform(0.3, 1.0, n), 2),

    'avg_task_delay_days': np.random.randint(0, 15, n),

    'peer_feedback_score': np.round(np.random.uniform(1.0, 5.0, n), 1),

    'manager_score': np.round(np.random.uniform(1.0, 5.0, n), 1),

    'sick_days': np.random.randint(0, 20, n),

    'certifications_count': np.random.randint(0, 8, n),

    'kudos_count': np.random.randint(0, 30, n),
})

# --- Create the target column (what we want to predict) ---
# Higher scores on good things → higher chance of being a High performer
score = (
    df['on_time_delivery_rate'] * 30 +
    df['manager_score'] * 15 +
    df['peer_feedback_score'] * 10 +
    df['training_hours'] * 0.1 +
    df['certifications_count'] * 3 +
    df['kudos_count'] * 0.5 -
    df['avg_task_delay_days'] * 2 -
    df['sick_days'] * 0.5 +
    np.random.normal(0, 5, n)   # some randomness
)

# Map the score into 3 bands: Low / Medium / High
df['perf_band'] = pd.cut(
    score,
    bins=[-9999, 30, 50, 9999],
    labels=['Low', 'Medium', 'High']
)

# Save to CSV
df.to_csv('data/employee_features.csv', index=False)
print("Dataset created! Shape:", df.shape)
print(df['perf_band'].value_counts())