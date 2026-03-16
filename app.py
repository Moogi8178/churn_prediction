"""
Customer Churn Prediction — Streamlit App
Trains a Gradient Boosting model at startup using a public dataset URL.
No pre-trained model files or heavy dependencies required.

Meru University of Science & Technology
Finley Barongo Magembe | CT204/109437/22
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:   #0B1D3A;
    --gold:   #C9A84C;
    --gold2:  #E8C97A;
    --cream:  #F7F3EC;
    --red:    #D64045;
    --green:  #2E9E6B;
    --amber:  #E07B30;
    --border: rgba(201,168,76,0.25);
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--cream) !important;
}
.main .block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1200px; }
.app-header {
    background: linear-gradient(135deg, #0B1D3A 0%, #1a3060 50%, #0B1D3A 100%);
    border: 1px solid var(--border); border-radius:16px;
    padding:2rem 2.5rem; margin-bottom:2rem; position:relative; overflow:hidden;
}
.app-header::before {
    content:''; position:absolute; top:0; right:0; width:300px; height:300px;
    background:radial-gradient(circle,rgba(201,168,76,0.08) 0%,transparent 70%);
}
.app-header h1 {
    font-family:'DM Serif Display',serif; font-size:2.4rem;
    color:var(--gold2) !important; margin:0 0 0.3rem 0; line-height:1.1;
}
.app-header p { color:rgba(247,243,236,0.6) !important; font-size:0.9rem; margin:0; }
.app-header .badge {
    display:inline-block; background:rgba(201,168,76,0.15);
    border:1px solid var(--border); border-radius:20px; padding:3px 12px;
    font-size:0.75rem; color:var(--gold); margin-bottom:0.8rem;
    letter-spacing:0.05em; text-transform:uppercase;
}
.card {
    background:rgba(255,255,255,0.04); border:1px solid var(--border);
    border-radius:12px; padding:1.5rem; margin-bottom:1rem;
}
.card-title {
    font-family:'DM Serif Display',serif; font-size:1.1rem; color:var(--gold2);
    margin-bottom:1rem; padding-bottom:0.5rem; border-bottom:1px solid var(--border);
}
.metric-box {
    background:rgba(255,255,255,0.04); border:1px solid var(--border);
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
}
.metric-box .label {
    font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
    color:rgba(247,243,236,0.5); margin-bottom:0.3rem;
}
.metric-box .value {
    font-family:'DM Serif Display',serif; font-size:1.8rem;
    color:var(--gold2); line-height:1;
}
[data-testid="stSidebar"] {
    background-color:#0d2240 !important; border-right:1px solid var(--border);
}
.stButton > button {
    background:linear-gradient(135deg,var(--gold),var(--gold2)) !important;
    color:var(--navy) !important; font-weight:700; border:none !important;
    border-radius:8px !important; padding:0.6rem 2rem !important;
    font-size:1rem !important; width:100%; transition:all 0.2s;
}
.stButton > button:hover { opacity:0.9; transform:translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { background:transparent; gap:4px; }
.stTabs [data-baseweb="tab"] {
    background:transparent; border-radius:8px;
    color:rgba(247,243,236,0.5) !important;
}
.stTabs [aria-selected="true"] {
    background:rgba(201,168,76,0.15) !important; color:var(--gold2) !important;
}
hr { border-color:var(--border) !important; }
div[data-baseweb="select"] > div {
    background:rgba(255,255,255,0.06) !important; border-color:var(--border) !important;
}
div[data-baseweb="select"] span { color:var(--cream) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING + MODEL TRAINING  (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════════

DATASET_URLS = [
    "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Bank%20Churn%20Modelling.csv",
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Churn_Modelling.csv",
    "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Bank%20Churn%20Modelling.csv",
]

FEATURE_COLS = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain',
]

@st.cache_data(show_spinner=False)
def load_and_train():
    # 1. Download dataset
    df = None
    used_url = ""
    for url in DATASET_URLS:
        try:
            df = pd.read_csv(url)
            used_url = url
            break
        except Exception:
            continue

    if df is None:
        return None, None, None, None, None, 0, ""

    # 2. Normalise column names
    df.columns = df.columns.str.strip().str.replace(' ', '').str.replace('_', '')
    rename = {
        'NumofProducts':'NumOfProducts', 'NumOfProducts':'NumOfProducts',
        'HasCreditCard':'HasCrCard',     'HasCrCard':'HasCrCard',
        'IsActiveMember':'IsActiveMember',
        'EstimatedSalary':'EstimatedSalary',
        'Churn':'Exited',                'Exited':'Exited',
        'CreditScore':'CreditScore',
    }
    df.rename(columns=rename, inplace=True)
    df.drop(columns=[c for c in ['RowNumber','CustomerId','Surname','customerID',
                                  'CustomerID'] if c in df.columns],
            inplace=True, errors='ignore')

    # 3. Impute missing
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # 4. Encode
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male':1,'Female':0,'M':1,'F':0}).fillna(0).astype(int)
    if 'Geography' in df.columns:
        df = pd.get_dummies(df, columns=['Geography'], drop_first=False)
    for g in ['Geography_France','Geography_Germany','Geography_Spain']:
        if g not in df.columns:
            df[g] = 0
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)

    if 'Exited' not in df.columns:
        return None, None, None, None, None, 0, ""

    # 5. Features & target
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df['Exited']

    # 6. Split & scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 7. Train Gradient Boosting (pure scikit-learn — no extra deps)
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.08, subsample=0.85,
        random_state=42,
    )
    model.fit(X_train_sc, y_train)

    # 8. Evaluate
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    metrics = {
        'Accuracy' : round(accuracy_score(y_test, y_pred),  4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall'   : round(recall_score(y_test, y_pred),    4),
        'F1-Score' : round(f1_score(y_test, y_pred),        4),
        'AUC-ROC'  : round(roc_auc_score(y_test, y_proba),  4),
        'cm'       : confusion_matrix(y_test, y_pred).tolist(),
    }
    fi = pd.Series(model.feature_importances_,
                   index=available).sort_values(ascending=False)

    return model, scaler, metrics, available, fi, len(df), used_url


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def gauge_chart(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={'suffix':'%','font':{'size':42,'color':'#F7F3EC','family':'DM Serif Display'}},
        gauge={
            'axis':{'range':[0,100],'tickcolor':'#F7F3EC',
                    'tickfont':{'color':'#F7F3EC','size':11}},
            'bar':{'color':color,'thickness':0.28},
            'bgcolor':'rgba(255,255,255,0.04)',
            'bordercolor':'rgba(201,168,76,0.3)',
            'steps':[
                {'range':[0,35],  'color':'rgba(46,158,107,0.15)'},
                {'range':[35,60], 'color':'rgba(224,123,48,0.15)'},
                {'range':[60,100],'color':'rgba(214,64,69,0.15)'},
            ],
            'threshold':{'line':{'color':color,'width':3},'thickness':0.8,'value':prob*100},
        },
    ))
    fig.update_layout(height=230,margin=dict(l=20,r=20,t=30,b=10),
                      paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#F7F3EC')
    return fig

def bar_chart(labels, values, colors):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=colors,
        hovertemplate='%{y}: %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        height=280, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F7F3EC',family='DM Sans',size=12),
        xaxis=dict(gridcolor='rgba(201,168,76,0.15)',
                   zerolinecolor='rgba(201,168,76,0.4)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
    )
    return fig

def confusion_chart(cm):
    fig = go.Figure(go.Heatmap(
        z=cm, x=['Not Churned','Churned'], y=['Not Churned','Churned'],
        text=[[str(v) for v in row] for row in cm],
        texttemplate='%{text}',
        colorscale=[[0,'#0B1D3A'],[1,'#C9A84C']],
        showscale=False,
    ))
    fig.update_layout(
        height=260, margin=dict(l=10,r=10,t=30,b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F7F3EC',family='DM Sans'),
        xaxis_title='Predicted', yaxis_title='Actual',
    )
    return fig

def contribution_chart(X_input, model, feature_cols):
    contribs = []
    for i, feat in enumerate(feature_cols):
        p = X_input.copy(); p[0,i] += 0.5
        m = X_input.copy(); m[0,i] -= 0.5
        delta = (model.predict_proba(p)[0][1] - model.predict_proba(m)[0][1])
        contribs.append((feat, delta))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    contribs = contribs[:8]
    labels = [c[0] for c in contribs]
    vals   = [c[1] for c in contribs]
    colors = ['#D64045' if v > 0 else '#2E9E6B' for v in vals]
    return bar_chart(labels, vals, colors)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <div class="badge">🏦 Retail Banking · Gradient Boosting · Auto-Trained</div>
  <h1>Customer Churn Predictor</h1>
  <p>Meru University of Science &amp; Technology &nbsp;·&nbsp; BSc Data Science &nbsp;·&nbsp;
     Finley Barongo Magembe &nbsp;·&nbsp; CT204/109437/22</p>
</div>
""", unsafe_allow_html=True)

# Load & train
with st.spinner("⚙️ Downloading dataset and training model... (first load only, ~20 seconds)"):
    model, scaler, metrics, feature_cols, feat_importance, n_records, data_url = load_and_train()

if model is None:
    st.error("❌ Failed to download dataset. Please check your internet connection and try again.")
    st.stop()

st.success(f"✅ Model trained on **{n_records:,} records** — ready for predictions!")

# Sidebar inputs
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'DM Serif Display\',serif;font-size:1.3rem;'
        'color:#E8C97A;margin-bottom:1.2rem;">📋 Customer Profile</p>',
        unsafe_allow_html=True
    )
    st.markdown("**Demographics**")
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender    = st.selectbox("Gender", ["Female", "Male"])
    age       = st.slider("Age", 18, 92, 42)
    st.markdown("---")
    st.markdown("**Account Details**")
    credit_score  = st.slider("Credit Score", 300, 850, 620)
    tenure        = st.slider("Tenure (years)", 0, 10, 3)
    balance       = st.number_input("Account Balance ($)", 0.0, 300000.0, 130000.0, step=1000.0)
    num_products  = st.selectbox("Number of Products", [1, 2, 3, 4])
    estimated_sal = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 82000.0, step=1000.0)
    st.markdown("---")
    st.markdown("**Engagement**")
    has_cr_card = st.radio("Has Credit Card?",  ["Yes","No"], horizontal=True)
    is_active   = st.radio("Is Active Member?", ["Yes","No"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Build feature vector
def build_input():
    row = {
        'CreditScore'      : credit_score,
        'Gender'           : 1 if gender == 'Male' else 0,
        'Age'              : age,
        'Tenure'           : tenure,
        'Balance'          : balance,
        'NumOfProducts'    : num_products,
        'HasCrCard'        : 1 if has_cr_card == 'Yes' else 0,
        'IsActiveMember'   : 1 if is_active   == 'Yes' else 0,
        'EstimatedSalary'  : estimated_sal,
        'Geography_France' : 1 if geography == 'France'  else 0,
        'Geography_Germany': 1 if geography == 'Germany' else 0,
        'Geography_Spain'  : 1 if geography == 'Spain'   else 0,
    }
    arr = np.array([[row.get(c, 0) for c in feature_cols]], dtype=float)
    return scaler.transform(arr)

# Tabs
tab1, tab2, tab3 = st.tabs(["  🎯 Prediction  ", "  📊 Model Performance  ", "  ℹ️ About  "])

with tab1:
    X_input = build_input()
    prob    = float(model.predict_proba(X_input)[0][1])

    if prob >= 0.60:
        risk, color, icon = 'HIGH RISK',   '#D64045', '🔴'
    elif prob >= 0.35:
        risk, color, icon = 'MEDIUM RISK', '#E07B30', '🟡'
    else:
        risk, color, icon = 'LOW RISK',    '#2E9E6B', '🟢'

    col_g, col_m = st.columns([1, 1.2], gap="large")

    with col_g:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Churn Probability</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(prob, color), use_container_width=True,
                        config={'displayModeBar': False})
        st.markdown(
            f'<div style="text-align:center;margin-top:-0.5rem;">'
            f'<span style="background:{color}20;border:1.5px solid {color};'
            f'border-radius:8px;padding:0.4rem 0.8rem;color:{color};font-weight:600;">'
            f'{icon} {risk}</span></div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Customer Summary</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="label">Credit Score</div>'
                        f'<div class="value">{credit_score}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="label">Age</div>'
                        f'<div class="value">{age}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="label">Balance</div>'
                        f'<div class="value">${balance:,.0f}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="label">Tenure</div>'
                        f'<div class="value">{tenure} yrs</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if prob >= 0.60:
            rec = "⚠️ Immediate retention action needed. Consider a personalised offer — loyalty bonus, rate review, or dedicated relationship manager."
        elif prob >= 0.35:
            rec = "📌 Monitor closely. Proactively engage within 30 days — product upgrade or satisfaction survey."
        else:
            rec = "✅ Customer appears stable. Maintain standard engagement cadence."

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04);border-left:3px solid {color};'
            f'border-radius:0 8px 8px 0;padding:0.8rem 1rem;font-size:0.85rem;'
            f'color:rgba(247,243,236,0.85);">'
            f'<strong style="color:{color}">Recommendation</strong><br>{rec}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    cf1, cf2 = st.columns(2, gap="large")
    with cf1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Feature Contribution (This Customer)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(contribution_chart(X_input, model, feature_cols),
                        use_container_width=True, config={'displayModeBar': False})
        st.markdown('<p style="font-size:0.72rem;color:rgba(247,243,236,0.4);margin:0">'
                    '🔴 Increases churn risk &nbsp; 🟢 Reduces churn risk</p>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cf2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Global Feature Importance (Model)</div>',
                    unsafe_allow_html=True)
        top8 = feat_importance.head(8)
        st.plotly_chart(
            bar_chart(top8.index.tolist(), top8.values.tolist(),
                      ['#C9A84C'] * len(top8)),
            use_container_width=True, config={'displayModeBar': False}
        )
        st.markdown('</div>', unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Model Performance — Test Set (20%)</div>',
                unsafe_allow_html=True)

    cols = st.columns(5)
    for col, key in zip(cols, ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="label">{key}</div>'
                f'<div class="value">{metrics[key]:.3f}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    cc, cfi = st.columns(2, gap="large")

    with cc:
        st.markdown("**Confusion Matrix**")
        st.plotly_chart(confusion_chart(metrics['cm']),
                        use_container_width=True, config={'displayModeBar': False})
    with cfi:
        st.markdown("**Top Feature Importances**")
        top8 = feat_importance.head(8)
        st.plotly_chart(
            bar_chart(top8.index.tolist(), top8.values.tolist(),
                      ['#C9A84C'] * len(top8)),
            use_container_width=True, config={'displayModeBar': False}
        )

    st.markdown('</div>', unsafe_allow_html=True)


with tab3:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">About This Application</div>
      <p style="color:rgba(247,243,236,0.75);line-height:1.7;">
        This system automatically downloads a public banking churn dataset at startup
        and trains a <strong style="color:#E8C97A">Gradient Boosting Classifier</strong>
        in real-time — no uploaded model files or heavy frameworks (TensorFlow, XGBoost)
        required.
      </p>
      <br>
      <p style="font-family:'DM Serif Display',serif;color:#E8C97A;font-size:1rem;margin-bottom:0.5rem;">Dataset</p>
      <p style="color:rgba(247,243,236,0.65);font-size:0.85rem;word-break:break-all;">
        Auto-fetched from: <code>{data_url}</code><br>
        Records: <strong>{n_records:,}</strong> &nbsp;·&nbsp;
        Same feature set as your original dataset (Geography, Gender, Age, Balance, etc.)
      </p>
      <br>
      <p style="font-family:'DM Serif Display',serif;color:#E8C97A;font-size:1rem;margin-bottom:0.5rem;">Research Objectives Covered</p>
      <ul style="color:rgba(247,243,236,0.75);line-height:2;">
        <li>✅ Data preprocessing — encoding, scaling, train/test split</li>
        <li>✅ Gradient Boosting model (scikit-learn only)</li>
        <li>✅ Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix</li>
        <li>✅ Per-customer feature contribution (sensitivity analysis)</li>
        <li>✅ Global feature importances</li>
        <li>✅ Real-time predictions with risk tiering</li>
      </ul>
      <br>
      <p style="font-family:'DM Serif Display',serif;color:#E8C97A;font-size:1rem;margin-bottom:0.5rem;">Tech Stack</p>
      <p style="color:rgba(247,243,236,0.6);font-size:0.85rem;">
        Python · Scikit-learn · Streamlit · Plotly · Pandas · NumPy
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr style="margin-top:2rem">
<p style="text-align:center;color:rgba(247,243,236,0.3);font-size:0.78rem;margin-top:0.5rem;">
  Finley Barongo Magembe · CT204/109437/22 · Meru University of Science and Technology · 2026
</p>
""", unsafe_allow_html=True)
