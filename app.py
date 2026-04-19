# app.py — Employee Performance Predictor (Streamlit)
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🏆",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d0f1a; }
    .stApp { background-color: #0d0f1a; color: #e8eaf6; }
    h1, h2, h3 { color: #e8eaf6 !important; }
    .metric-card {
        background: #13162a;
        border: 1px solid #1f2440;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .high-box  { background:#0d2e22; border:1px solid #00e5a0; border-radius:12px; padding:20px; }
    .med-box   { background:#2e2710; border:1px solid #ffd166; border-radius:12px; padding:20px; }
    .low-box   { background:#2e1018; border:1px solid #ff6584; border-radius:12px; padding:20px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.markdown("## 🏆 Employee Performance Predictor")
st.markdown("*ML-powered HR analytics tool — predict performance bands and get actionable insights*")
st.divider()

# ── Load or train model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'models/employee_perf_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Train a fresh model if pkl not found
        st.warning("Model file not found — training now from synthetic data...")
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, RobustScaler
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # Generate data inline
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'age': np.random.randint(22, 58, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'education': np.random.choice(['High School','Bachelor','Master','PhD'], n, p=[0.15,0.50,0.28,0.07]),
            'department': np.random.choice(['Engineering','Sales','HR','Finance','Marketing'], n),
            'job_level': np.random.choice(['Junior','Mid','Senior','Lead'], n, p=[0.35,0.35,0.20,0.10]),
            'experience_years': np.random.randint(0, 25, n),
            'salary_band': np.random.choice(['Low','Medium','High'], n, p=[0.30,0.50,0.20]),
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
        score = (
            df['on_time_delivery_rate'] * 30 + df['manager_score'] * 15 +
            df['peer_feedback_score'] * 10 + df['training_hours'] * 0.1 +
            df['certifications_count'] * 3 + df['kudos_count'] * 0.5 -
            df['avg_task_delay_days'] * 2 - df['sick_days'] * 0.5 +
            np.random.normal(0, 5, n)
        )
        df['perf_band'] = pd.cut(score, bins=[-9999,30,50,9999], labels=['Low','Medium','High'])
        y = df['perf_band']
        X = df.drop(columns=['perf_band'])

        cat_cols = X.select_dtypes(include='object').columns
        num_cols = X.select_dtypes(include='number').columns

        pre = ColumnTransformer([
            ('num', Pipeline([('i', SimpleImputer(strategy='median')), ('s', RobustScaler())]), num_cols),
            ('cat', Pipeline([('i', SimpleImputer(strategy='most_frequent')), ('o', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ])
        model = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=13))])
        model.fit(X, y)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)
        return model

model = load_model()

# ── Sidebar — Input Form ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Employee Details")

    department = st.selectbox("Department", ['Engineering','Sales','HR','Finance','Marketing'])
    job_level  = st.selectbox("Job Level",  ['Junior','Mid','Senior','Lead'])
    education  = st.selectbox("Education",  ['High School','Bachelor','Master','PhD'])
    gender     = st.selectbox("Gender",     ['Male','Female'])
    salary_band = st.selectbox("Salary Band", ['Low','Medium','High'])

    st.markdown("---")
    age              = st.slider("Age",                    22, 58,  30)
    experience_years = st.slider("Experience (years)",     0,  25,   5)
    training_hours   = st.slider("Training Hours",         0,  120, 40)
    certifications   = st.slider("Certifications",         0,   8,   2)
    projects_count   = st.slider("Projects Completed",     1,  20,   6)
    kudos_count      = st.slider("Kudos Received",         0,  30,   5)

    st.markdown("---")
    on_time_delivery = st.slider("On-Time Delivery Rate",  0.3, 1.0, 0.80, step=0.01)
    manager_score    = st.slider("Manager Score (1-5)",    1.0, 5.0, 3.5,  step=0.1)
    peer_feedback    = st.slider("Peer Feedback (1-5)",    1.0, 5.0, 3.5,  step=0.1)
    sick_days        = st.slider("Sick Days",              0,  20,   4)
    avg_task_delay   = st.slider("Avg Task Delay (days)",  0,  15,   2)

    predict_btn = st.button("⚡ Predict Performance", use_container_width=True, type="primary")

# ── Main area ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Model", "Random Forest")
col2.metric("Features", "16")
col3.metric("Classes", "High / Medium / Low")

st.divider()

if predict_btn:
    # Build input row
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'education': education,
        'department': department,
        'job_level': job_level,
        'experience_years': experience_years,
        'salary_band': salary_band,
        'training_hours': training_hours,
        'projects_count': projects_count,
        'on_time_delivery_rate': on_time_delivery,
        'avg_task_delay_days': avg_task_delay,
        'peer_feedback_score': peer_feedback,
        'manager_score': manager_score,
        'sick_days': sick_days,
        'certifications_count': certifications,
        'kudos_count': kudos_count,
    }])

    pred_band  = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0]
    classes    = model.classes_

    # ── Result box ────────────────────────────────────────────────
    emoji_map = {'High': '🏆', 'Medium': '📈', 'Low': '⚠️'}
    box_map   = {'High': 'high-box', 'Medium': 'med-box', 'Low': 'low-box'}
    desc_map  = {
        'High':   'Exceeding expectations. Consider for promotion or recognition.',
        'Medium': 'Solid contributor. Targeted training can unlock full potential.',
        'Low':    'Needs support and coaching. Recommend a performance improvement plan.'
    }

    st.markdown(f"""
    <div class="{box_map[pred_band]}">
      <h2>{emoji_map[pred_band]} Predicted Band: <strong>{pred_band.upper()} PERFORMER</strong></h2>
      <p style="color:#aaa; margin-top:8px">{desc_map[pred_band]}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("&nbsp;")

    # ── Probability chart ─────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📊 Prediction Confidence")
        colors = {'High':'#00e5a0', 'Medium':'#ffd166', 'Low':'#ff6584'}
        fig, ax = plt.subplots(figsize=(5, 2.5))
        fig.patch.set_facecolor('#13162a')
        ax.set_facecolor('#13162a')
        bars = ax.barh(list(classes), pred_proba,
                       color=[colors[c] for c in classes], height=0.5)
        ax.set_xlim(0, 1)
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values(): spine.set_visible(False)
        for bar, prob in zip(bars, pred_proba):
            ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', color='white', fontsize=10)
        st.pyplot(fig)

    with col_b:
        st.markdown("#### 🎯 HR Recommendations")
        actions = {
            'High':   ['Nominate for leadership fast-track', 'Assign as mentor to juniors', 'Consider next-cycle promotion'],
            'Medium': ['Enrol in upskilling course (60+ hrs)', 'Assign certification path (AWS/PMP)', 'Quarterly 1-on-1 coaching'],
            'Low':    ['Create 90-day PIP', 'Sprint planning workshop', 'Weekly progress review with manager']
        }
        for action in actions[pred_band]:
            st.markdown(f"→ {action}")

    # ── Feature importance ────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔍 What Matters Most (Feature Importance)")

    rf_clf = model.named_steps['clf']
    try:
        ohe_cols = model.named_steps['pre'].named_transformers_['cat']['o'].get_feature_names_out(
            model.named_steps['pre'].named_transformers_['cat']['i'].get_feature_names_out() if hasattr(
                model.named_steps['pre'].named_transformers_['cat']['i'], 'get_feature_names_out') else
            ['gender','education','department','job_level','salary_band']
        )
    except:
        ohe_cols = [f'cat_{i}' for i in range(rf_clf.n_features_in_ - 11)]

    num_feature_names = ['age','experience_years','training_hours','projects_count',
                         'on_time_delivery_rate','avg_task_delay_days','peer_feedback_score',
                         'manager_score','sick_days','certifications_count','kudos_count']
    all_names = num_feature_names + list(ohe_cols)

    imp_df = pd.DataFrame({'Feature': all_names[:len(rf_clf.feature_importances_)],
                           'Importance': rf_clf.feature_importances_})
    imp_df = imp_df.sort_values('Importance', ascending=True).tail(10)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_facecolor('#13162a')
    ax2.set_facecolor('#13162a')
    ax2.barh(imp_df['Feature'], imp_df['Importance'], color='#6c63ff')
    ax2.tick_params(colors='#aaa')
    for spine in ax2.spines.values(): spine.set_visible(False)
    st.pyplot(fig2)

    # ── Input summary ─────────────────────────────────────────────
    with st.expander("📋 View Input Data"):
        st.dataframe(input_data.T.rename(columns={0: 'Value'}))

else:
    st.info("👈 Fill in the employee details in the sidebar, then click **Predict Performance**.")
    st.markdown("""
    #### How to use this app
    1. Set the employee's details in the left sidebar
    2. Click the **Predict Performance** button
    3. See the predicted band (High / Medium / Low)
    4. Get confidence scores and HR recommendations
    5. View which features drove the prediction
    """)