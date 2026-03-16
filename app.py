"""
Customer Churn Prediction — Page 1: Input Form
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --navy:#0B1D3A; --card:#0f2548; --gold:#C9A84C; --gold2:#FFE08A;
    --text:#E8EDF5; --muted:#A8B8D0; --border:#2A4A72;
    --red:#E85555; --green:#3DBE8A; --amber:#F0922B;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--navy) !important;
    color: var(--text) !important;
}
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 960px; }

/* Header */
.page-header {
    background: linear-gradient(135deg, #0f2548 0%, #1a3a6a 50%, #0f2548 100%);
    border: 1px solid var(--border);
    border-left: 5px solid var(--gold2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2.5rem;
}
.page-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: var(--gold2) !important;
    margin: 0;
    letter-spacing: -0.5px;
}

/* Section title */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--gold2);
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
}

/* Input card */
.input-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

/* Labels */
label, .stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label {
    color: #D0DCEA !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}
div[data-baseweb="select"] > div {
    background: #152d55 !important;
    border-color: var(--border) !important;
    color: #FFFFFF !important;
}
div[data-baseweb="select"] span { color: #FFFFFF !important; font-weight: 500 !important; }
[data-testid="stNumberInput"] input {
    background: #152d55 !important;
    color: #FFFFFF !important;
    border-color: var(--border) !important;
    font-weight: 500 !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #C9A84C, #FFE08A) !important;
    color: #0B1D3A !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 3rem !important;
    width: 100%;
    letter-spacing: 0.04em;
    margin-top: 1rem;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

hr { border-color: var(--border) !important; }
p, span { color: var(--text); }

/* Hide default sidebar nav */
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Data & Model ──────────────────────────────────────────────────────────────
DATA_URL = (
    "https://gist.githubusercontent.com/arjunrao796123/"
    "7c30f2b6d4a3a3746b0154260a7f46e8/raw/"
    "733351a9f0e58e194bfe4d6c21253cdf186c7b90/Churn_data.csv"
)

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_URL)

@st.cache_resource(show_spinner=False)
def train_model():
    df = load_data()
    drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    df = df.drop(columns=drop_cols).rename(columns={"Churn": "Exited"})
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(int)
    target = "Exited" if "Exited" in df.columns else df.columns[-1]
    X, y = df.drop(columns=[target]), df[target]
    feat_names = X.columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    model.fit(X_tr_sc, y_tr)
    y_pred  = model.predict(X_te_sc)
    y_proba = model.predict_proba(X_te_sc)[:, 1]
    metrics = {
        "Accuracy":  round(accuracy_score(y_te, y_pred),  4),
        "Precision": round(precision_score(y_te, y_pred), 4),
        "Recall":    round(recall_score(y_te, y_pred),    4),
        "F1-Score":  round(f1_score(y_te, y_pred),        4),
        "AUC-ROC":   round(roc_auc_score(y_te, y_proba),  4),
    }
    cm = confusion_matrix(y_te, y_pred)
    return model, scaler, le, feat_names, metrics, cm

# Train on load
with st.spinner("🔄 Training model — first run takes ~20 seconds…"):
    try:
        model, scaler, le, feat_names, metrics, cm = train_model()
        # Store in session so results page can access
        st.session_state["model"]      = model
        st.session_state["scaler"]     = scaler
        st.session_state["le"]         = le
        st.session_state["feat_names"] = feat_names
        st.session_state["metrics"]    = metrics
        st.session_state["cm"]         = cm
        ready = True
    except Exception as e:
        st.error(f"Model training failed: {e}")
        ready = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>Customer Churn</h1>
</div>
""", unsafe_allow_html=True)

if not ready:
    st.stop()

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
with col3:
    age = st.slider("Age", 18, 92, 42)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🏦 Account Details</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    credit_score = st.slider("Credit Score", 300, 850, 620)
with col5:
    tenure = st.slider("Tenure (years)", 0, 10, 3)
with col6:
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

col7, col8 = st.columns(2)
with col7:
    balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 130000.0, step=1000.0)
with col8:
    estimated_sal = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 82000.0, step=1000.0)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📲 Engagement</div>', unsafe_allow_html=True)

col9, col10 = st.columns(2)
with col9:
    has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
with col10:
    is_active = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔍  Predict Customer Churn", use_container_width=True):
    # Save customer data to session state
    st.session_state["customer"] = {
        "CreditScore":     credit_score,
        "Geography":       geography,
        "Gender":          gender,
        "Age":             age,
        "Tenure":          tenure,
        "Balance":         balance,
        "NumOfProducts":   num_products,
        "HasCrCard":       1 if has_cr_card == "Yes" else 0,
        "IsActiveMember":  1 if is_active   == "Yes" else 0,
        "EstimatedSalary": estimated_sal,
    }
    st.switch_page("pages/1_Results.py")
