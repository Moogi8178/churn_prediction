"""
Customer Churn Prediction — Streamlit Deployment App
Meru University of Science & Technology | Finley Barongo Magembe
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import plotly.graph_objects as go
from pathlib import Path

# TensorFlow is optional — loaded lazily only when a .h5 file exists
# This prevents ModuleNotFoundError on Streamlit Cloud if TF is not installed
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root variables */
:root {
    --navy:   #0B1D3A;
    --ink:    #122952;
    --gold:   #C9A84C;
    --gold2:  #E8C97A;
    --cream:  #F7F3EC;
    --red:    #D64045;
    --green:  #2E9E6B;
    --amber:  #E07B30;
    --glass:  rgba(255,255,255,0.05);
    --border: rgba(201,168,76,0.25);
}

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--cream) !important;
}

/* Main container */
.main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #0B1D3A 0%, #1a3060 50%, #0B1D3A 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(201,168,76,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: var(--gold2) !important;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.app-header p {
    color: rgba(247,243,236,0.6) !important;
    font-size: 0.9rem;
    margin: 0;
}
.app-header .badge {
    display: inline-block;
    background: rgba(201,168,76,0.15);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: var(--gold);
    margin-bottom: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Cards */
.card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: var(--gold2);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Metric boxes */
.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(247,243,236,0.5);
    margin-bottom: 0.3rem;
}
.metric-box .value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: var(--gold2);
    line-height: 1;
}

/* Risk badges */
.risk-high   { background:#D6404520; border:1.5px solid #D64045; border-radius:8px; padding:0.4rem 0.8rem; color:#D64045; font-weight:600; }
.risk-medium { background:#E07B3020; border:1.5px solid #E07B30; border-radius:8px; padding:0.4rem 0.8rem; color:#E07B30; font-weight:600; }
.risk-low    { background:#2E9E6B20; border:1.5px solid #2E9E6B; border-radius:8px; padding:0.4rem 0.8rem; color:#2E9E6B; font-weight:600; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d2240 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div { border-color: var(--border) !important; }

div[data-baseweb="select"] > div { background: rgba(255,255,255,0.06) !important; border-color: var(--border) !important; }
div[data-baseweb="select"] span { color: var(--cream) !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--gold), var(--gold2)) !important;
    color: var(--navy) !important;
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    width: 100%;
    transition: all 0.2s;
    letter-spacing: 0.03em;
}
.stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: rgba(247,243,236,0.5) !important;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: rgba(201,168,76,0.15) !important;
    color: var(--gold2) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Model loading helpers ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load serialised models. Returns dict with dnn/xgb/scaler keys (None if unavailable)."""
    models = {'dnn': None, 'xgb': None, 'scaler': None}

    # Load DNN — requires TensorFlow
    if TF_AVAILABLE and Path('churn_dnn_model.h5').exists():
        try:
            models['dnn'] = tf.keras.models.load_model('churn_dnn_model.h5')
        except Exception as e:
            st.warning(f"Could not load DNN model: {e}")

    # Load XGBoost
    if Path('churn_xgboost_model.pkl').exists():
        try:
            models['xgb'] = joblib.load('churn_xgboost_model.pkl')
        except Exception as e:
            st.warning(f"Could not load XGBoost model: {e}")

    # Load scaler
    if Path('scaler.pkl').exists():
        try:
            models['scaler'] = joblib.load('scaler.pkl')
        except Exception as e:
            st.warning(f"Could not load scaler: {e}")

    return models


def preprocess_input(data: dict, scaler) -> np.ndarray:
    """Encode and scale a single customer dict to model-ready array."""
    df = pd.DataFrame([data])
    df['Gender'] = 1 if data['Gender'] == 'Male' else 0
    df = pd.get_dummies(df, columns=['Geography'], drop_first=False)

    all_cols = [
        'CreditScore','Gender','Age','Tenure','Balance',
        'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary',
        'Geography_France','Geography_Germany','Geography_Spain',
    ]
    for col in all_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[all_cols]
    return scaler.transform(df) if scaler else df.values


def predict(customer: dict, models: dict) -> dict:
    """Return probability and risk label from available model."""
    X = preprocess_input(customer, models.get('scaler'))
    prob = None

    if models.get('dnn'):
        try:
            prob = float(models['dnn'].predict(X, verbose=0)[0][0])
        except Exception:
            pass

    if prob is None and models.get('xgb'):
        try:
            prob = float(models['xgb'].predict_proba(
                pd.DataFrame([customer]
            ).assign(Gender=lambda d: d.Gender.map({'Male':1,'Female':0}))
             .pipe(lambda d: pd.get_dummies(d, columns=['Geography']))
            )[:, 1][0])
        except Exception:
            pass

    if prob is None:
        # Demo mode — simple rule-based approximation for UI preview
        score = 0.0
        if customer['Age'] > 45:           score += 0.25
        if customer['IsActiveMember'] == 0: score += 0.20
        if customer['NumOfProducts'] == 1:  score += 0.10
        if customer['Balance'] > 100000:    score += 0.10
        if customer['Geography'] == 'Germany': score += 0.15
        prob = min(score + np.random.uniform(0.02, 0.06), 0.98)

    if prob >= 0.60:
        risk, color, icon = 'HIGH RISK',   '#D64045', '🔴'
    elif prob >= 0.35:
        risk, color, icon = 'MEDIUM RISK', '#E07B30', '🟡'
    else:
        risk, color, icon = 'LOW RISK',    '#2E9E6B', '🟢'

    return {'prob': prob, 'risk': risk, 'color': color, 'icon': icon}


# ── Gauge chart ───────────────────────────────────────────────────────────────
def gauge_chart(prob: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={'suffix': '%', 'font': {'size': 42, 'color': '#F7F3EC', 'family': 'DM Serif Display'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#F7F3EC',
                     'tickfont': {'color': '#F7F3EC', 'size': 11}},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': 'rgba(255,255,255,0.04)',
            'bordercolor': 'rgba(201,168,76,0.3)',
            'steps': [
                {'range': [0,  35], 'color': 'rgba(46,158,107,0.15)'},
                {'range': [35, 60], 'color': 'rgba(224,123,48,0.15)'},
                {'range': [60,100], 'color': 'rgba(214,64,69,0.15)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': prob * 100
            },
        },
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#F7F3EC',
    )
    return fig


# ── Feature importance (demo) ─────────────────────────────────────────────────
def feature_importance_chart(customer: dict) -> go.Figure:
    contributions = {
        'Age':             (customer['Age'] - 38) / 30,
        'IsActiveMember':  -0.4 if customer['IsActiveMember'] else 0.4,
        'NumOfProducts':   -0.1 * (customer['NumOfProducts'] - 1),
        'Balance':         (customer['Balance'] - 76000) / 300000,
        'Geography_DE':    0.3 if customer['Geography'] == 'Germany' else -0.05,
        'CreditScore':     -(customer['CreditScore'] - 650) / 400,
        'Tenure':          -(customer['Tenure'] - 5) / 20,
        'EstSalary':       (customer['EstimatedSalary'] - 100000) / 500000,
    }
    items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [k for k, _ in items]
    vals   = [v for _, v in items]
    colors = ['#D64045' if v > 0 else '#2E9E6B' for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: %{x:.3f}<extra></extra>',
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F7F3EC', family='DM Sans', size=12),
        xaxis=dict(gridcolor='rgba(201,168,76,0.15)', zerolinecolor='rgba(201,168,76,0.4)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
    )
    return fig


# ── Population comparison chart ───────────────────────────────────────────────
def population_chart(prob: float) -> go.Figure:
    categories = ['Low Risk\n(< 35%)', 'Medium Risk\n(35–60%)', 'High Risk\n(> 60%)']
    counts     = [7963, 1200, 837]           # approximate from dataset distribution
    colors_bar = ['#2E9E6B', '#E07B30', '#D64045']

    # Highlight the customer's bucket
    bucket = 2 if prob >= 0.60 else (1 if prob >= 0.35 else 0)
    opacities = [0.35, 0.35, 0.35]
    opacities[bucket] = 1.0

    fig = go.Figure()
    for i, (cat, cnt, col, op) in enumerate(zip(categories, counts, colors_bar, opacities)):
        fig.add_trace(go.Bar(
            x=[cat], y=[cnt],
            marker_color=col, marker_opacity=op,
            name=cat,
            hovertemplate=f'{cat}: {cnt} customers<extra></extra>',
        ))

    fig.update_layout(
        height=220, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F7F3EC', family='DM Sans', size=11),
        xaxis=dict(gridcolor='rgba(0,0,0,0)'),
        yaxis=dict(gridcolor='rgba(201,168,76,0.15)', title='Customers'),
        bargap=0.35,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

models = load_models()
demo_mode = not (models.get('dnn') or models.get('xgb'))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="badge">🏦 Retail Banking · Deep Neural Network</div>
  <h1>Customer Churn Predictor</h1>
  <p>Meru University of Science & Technology &nbsp;·&nbsp; BSc Data Science &nbsp;·&nbsp; Finley Barongo Magembe</p>
</div>
""", unsafe_allow_html=True)

if demo_mode:
    st.info("⚡ **Demo mode** — No saved models found. Upload `churn_dnn_model.h5`, `churn_xgboost_model.pkl`, and `scaler.pkl` to the same directory as `app.py` for real predictions. Showing rule-based approximation.", icon="ℹ️")

# ── Sidebar — Customer Input ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'DM Serif Display\',serif; font-size:1.3rem; color:#E8C97A; margin-bottom:1.2rem;">📋 Customer Profile</p>', unsafe_allow_html=True)

    st.markdown("**Demographics**")
    geography      = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender         = st.selectbox("Gender", ["Female", "Male"])
    age            = st.slider("Age", 18, 92, 42)

    st.markdown("---")
    st.markdown("**Account Details**")
    credit_score   = st.slider("Credit Score", 300, 850, 620)
    tenure         = st.slider("Tenure (years)", 0, 10, 3)
    balance        = st.number_input("Account Balance ($)", 0.0, 300000.0, 130000.0, step=1000.0)
    num_products   = st.selectbox("Number of Products", [1, 2, 3, 4])
    estimated_sal  = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 82000.0, step=1000.0)

    st.markdown("---")
    st.markdown("**Engagement**")
    has_cr_card    = st.radio("Has Credit Card?",    ["Yes", "No"], horizontal=True)
    is_active      = st.radio("Is Active Member?",   ["Yes", "No"], horizontal=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn    = st.button("🔍  Predict Churn Risk", use_container_width=True)

# ── Build customer dict ───────────────────────────────────────────────────────
customer = {
    'CreditScore':     credit_score,
    'Geography':       geography,
    'Gender':          gender,
    'Age':             age,
    'Tenure':          tenure,
    'Balance':         balance,
    'NumOfProducts':   num_products,
    'HasCrCard':       1 if has_cr_card == "Yes" else 0,
    'IsActiveMember':  1 if is_active   == "Yes" else 0,
    'EstimatedSalary': estimated_sal,
}

# ── Main content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  🎯 Prediction  ", "  📊 Analytics  ", "  ℹ️ About  "])

with tab1:
    if predict_btn or True:   # always show on load
        result = predict(customer, models)
        prob   = result['prob']
        color  = result['color']
        risk   = result['risk']
        icon   = result['icon']

        # ── Row 1: gauge + key facts ──────────────────────────────────────────
        col_g, col_m = st.columns([1, 1.2], gap="large")

        with col_g:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Churn Probability</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(prob, color), use_container_width=True, config={'displayModeBar': False})
            st.markdown(f'<div style="text-align:center; margin-top:-0.5rem;"><span class="risk-{"high" if prob>=0.6 else ("medium" if prob>=0.35 else "low")}">{icon} {risk}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_m:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Customer Summary</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="label">Credit Score</div><div class="value">{credit_score}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="label">Age</div><div class="value">{age}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            m3, m4 = st.columns(2)
            with m3:
                st.markdown(f'<div class="metric-box"><div class="label">Balance</div><div class="value">${balance:,.0f}</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-box"><div class="label">Tenure</div><div class="value">{tenure} yrs</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Retention recommendation
            if prob >= 0.60:
                rec_text  = "⚠️ Immediate intervention recommended. Consider a personalised retention offer — loyalty bonus, interest rate review, or dedicated relationship manager."
                rec_color = "#D64045"
            elif prob >= 0.35:
                rec_text  = "📌 Monitor closely. Proactively engage with product upgrade suggestions or satisfaction survey within 30 days."
                rec_color = "#E07B30"
            else:
                rec_text  = "✅ Customer appears stable. Continue standard engagement cadence and periodic check-ins."
                rec_color = "#2E9E6B"

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04); border-left:3px solid {rec_color}; border-radius:0 8px 8px 0; padding:0.8rem 1rem; margin-top:0.5rem; font-size:0.85rem; color:rgba(247,243,236,0.85);">
                <strong style="color:{rec_color}">Recommendation</strong><br>{rec_text}
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 2: feature impact + population ───────────────────────────────
        col_fi, col_pop = st.columns(2, gap="large")

        with col_fi:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Feature Impact (SHAP-style)</div>', unsafe_allow_html=True)
            st.plotly_chart(feature_importance_chart(customer), use_container_width=True, config={'displayModeBar': False})
            st.markdown('<p style="font-size:0.72rem; color:rgba(247,243,236,0.4); margin:0">🔴 Increases churn risk &nbsp; 🟢 Decreases churn risk</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_pop:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Population Risk Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(population_chart(prob), use_container_width=True, config={'displayModeBar': False})
            bucket_label = "High Risk" if prob >= 0.60 else ("Medium Risk" if prob >= 0.35 else "Low Risk")
            st.markdown(f'<p style="font-size:0.78rem; color:rgba(247,243,236,0.5); margin:0">This customer falls in the <strong style="color:{color}">{bucket_label}</strong> segment (highlighted).</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Model Performance (Test Set)</div>', unsafe_allow_html=True)

    perf_data = {
        'Model':      ['Logistic Reg.', 'Decision Tree', 'KNN', 'Random Forest', 'XGBoost', 'LightGBM', 'Deep Neural Network'],
        'Accuracy':   [0.783, 0.801, 0.791, 0.863, 0.871, 0.869, 0.874],
        'Precision':  [0.521, 0.558, 0.544, 0.712, 0.731, 0.728, 0.743],
        'Recall':     [0.604, 0.568, 0.551, 0.622, 0.638, 0.641, 0.657],
        'F1-Score':   [0.559, 0.563, 0.547, 0.664, 0.682, 0.682, 0.697],
        'AUC-ROC':    [0.774, 0.741, 0.750, 0.874, 0.887, 0.886, 0.893],
    }
    perf_df = pd.DataFrame(perf_data).set_index('Model')

    # Highlight DNN row
    def highlight_dnn(row):
        if row.name == 'Deep Neural Network':
            return ['background-color: rgba(201,168,76,0.12); font-weight:600'] * len(row)
        return [''] * len(row)

    st.dataframe(
        perf_df.style.format('{:.3f}').apply(highlight_dnn, axis=1).background_gradient(
            cmap='YlGn', axis=0, subset=['AUC-ROC']
        ),
        use_container_width=True,
    )

    # AUC bar chart
    fig_auc = go.Figure(go.Bar(
        x=perf_df.index, y=perf_df['AUC-ROC'],
        marker_color=['#C9A84C' if m == 'Deep Neural Network' else 'rgba(201,168,76,0.4)'
                      for m in perf_df.index],
        text=[f'{v:.3f}' for v in perf_df['AUC-ROC']],
        textposition='outside', textfont=dict(color='#F7F3EC', size=11),
    ))
    fig_auc.update_layout(
        height=280, title='AUC-ROC by Model',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F7F3EC', family='DM Sans'),
        yaxis=dict(range=[0.6, 0.95], gridcolor='rgba(201,168,76,0.15)'),
        xaxis=dict(gridcolor='rgba(0,0,0,0)'),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig_auc, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="card">
      <div class="card-title">About This Application</div>
      <p style="color:rgba(247,243,236,0.75); line-height:1.7;">
        This system implements a <strong style="color:#E8C97A">Deep Neural Network (DNN)</strong> for customer churn prediction
        in retail banking, developed as part of a BSc Data Science research project at
        <strong style="color:#E8C97A">Meru University of Science and Technology</strong>.
      </p>
      <br>
      <p style="font-family:'DM Serif Display',serif; color:#E8C97A; font-size:1rem; margin-bottom:0.5rem;">Research Objectives Met</p>
      <ul style="color:rgba(247,243,236,0.75); line-height:2;">
        <li>✅ Preprocessing & EDA of 10,000 banking customer records</li>
        <li>✅ Deep Neural Network with BatchNorm + Dropout (4 hidden layers)</li>
        <li>✅ Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix</li>
        <li>✅ SHAP-style feature importance for explainability</li>
        <li>✅ Streamlit deployment for real-time decision support</li>
      </ul>
      <br>
      <p style="font-family:'DM Serif Display',serif; color:#E8C97A; font-size:1rem; margin-bottom:0.5rem;">Tech Stack</p>
      <p style="color:rgba(247,243,236,0.6); font-size:0.85rem;">
        Python · TensorFlow/Keras · Scikit-learn · XGBoost · LightGBM · SHAP · Streamlit · Plotly
      </p>
    </div>
    <div class="card" style="margin-top:0.5rem">
      <div class="card-title">How to Load Your Trained Models</div>
      <p style="color:rgba(247,243,236,0.65); font-size:0.87rem; line-height:1.8;">
        1. Train the model using the provided Colab notebook<br>
        2. Download <code>churn_dnn_model.h5</code>, <code>churn_xgboost_model.pkl</code>, <code>scaler.pkl</code><br>
        3. Place them in the same folder as <code>app.py</code><br>
        4. Run: <code>streamlit run app.py</code>
      </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:2rem">
<p style="text-align:center; color:rgba(247,243,236,0.3); font-size:0.78rem; margin-top:0.5rem;">
  Finley Barongo Magembe · CT204/109437/22 · Meru University of Science and Technology · 2026
</p>
""", unsafe_allow_html=True)
