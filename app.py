"""
Customer Churn Prediction — Streamlit App
Meru University of Science & Technology | Finley Barongo Magembe | CT204/109437/22

Dependencies: streamlit, scikit-learn, pandas, numpy, matplotlib (all pre-installed on Streamlit Cloud)
No tensorflow, xgboost, plotly, or any other heavy package needed.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;600&display=swap');
:root{--navy:#0B1D3A;--gold:#C9A84C;--gold2:#E8C97A;--cream:#F7F3EC;
      --red:#D64045;--green:#2E9E6B;--amber:#E07B30;--border:rgba(201,168,76,0.25);}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--navy)!important;color:var(--cream)!important;}
.main .block-container{padding-top:1.5rem;padding-bottom:3rem;max-width:1200px;}
.app-header{background:linear-gradient(135deg,#0B1D3A 0%,#1a3060 50%,#0B1D3A 100%);
    border:1px solid var(--border);border-radius:16px;padding:2rem 2.5rem;margin-bottom:2rem;}
.app-header h1{font-family:'DM Serif Display',serif;font-size:2.4rem;color:var(--gold2)!important;margin:0 0 0.3rem 0;}
.app-header p{color:rgba(247,243,236,0.6)!important;font-size:0.9rem;margin:0;}
.badge{display:inline-block;background:rgba(201,168,76,0.15);border:1px solid var(--border);
    border-radius:20px;padding:3px 12px;font-size:0.75rem;color:var(--gold);
    margin-bottom:0.8rem;letter-spacing:.05em;text-transform:uppercase;}
.card{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:12px;padding:1.5rem;margin-bottom:1rem;}
.card-title{font-family:'DM Serif Display',serif;font-size:1.1rem;color:var(--gold2);
    margin-bottom:1rem;padding-bottom:.5rem;border-bottom:1px solid var(--border);}
.mbox{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:10px;padding:1rem;text-align:center;}
.mbox .lbl{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:rgba(247,243,236,.5);margin-bottom:.3rem;}
.mbox .val{font-family:'DM Serif Display',serif;font-size:1.8rem;color:var(--gold2);line-height:1;}
[data-testid="stSidebar"]{background-color:#0d2240!important;border-right:1px solid var(--border);}
div[data-baseweb="select"]>div{background:rgba(255,255,255,.06)!important;border-color:var(--border)!important;}
div[data-baseweb="select"] span{color:var(--cream)!important;}
.stButton>button{background:linear-gradient(135deg,var(--gold),var(--gold2))!important;
    color:var(--navy)!important;font-weight:700;border:none!important;border-radius:8px!important;
    padding:.6rem 2rem!important;font-size:1rem!important;width:100%;}
.stTabs [data-baseweb="tab-list"]{background:transparent;gap:4px;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;color:rgba(247,243,236,.5)!important;}
.stTabs [aria-selected="true"]{background:rgba(201,168,76,.15)!important;color:var(--gold2)!important;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
NAVY   = "#0B1D3A"
GOLD   = "#C9A84C"
GOLD2  = "#E8C97A"
CREAM  = "#F7F3EC"
RED    = "#D64045"
GREEN  = "#2E9E6B"
AMBER  = "#E07B30"

plt.rcParams.update({
    "figure.facecolor":  NAVY,
    "axes.facecolor":    "#0d2240",
    "axes.edgecolor":    "#2a4a6a",
    "axes.labelcolor":   CREAM,
    "xtick.color":       CREAM,
    "ytick.color":       CREAM,
    "text.color":        CREAM,
    "grid.color":        "#1e3a5a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
})

# ── Dataset ───────────────────────────────────────────────────────────────────
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
    df = df.drop(columns=drop_cols)
    df = df.rename(columns={"Churn": "Exited"})

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
    X_tr_sc, X_te_sc = scaler.fit_transform(X_tr), scaler.transform(X_te)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
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


def predict_customer(customer, model, scaler, le, feat_names):
    df = pd.DataFrame([customer])
    df["Gender"] = le.transform(df["Gender"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=False)
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(int)
    for c in feat_names:
        if c not in df.columns:
            df[c] = 0
    X_sc = scaler.transform(df[feat_names])
    prob = float(model.predict_proba(X_sc)[0][1])
    if prob >= 0.60:
        risk, color, icon = "HIGH RISK",   RED,   "🔴"
    elif prob >= 0.35:
        risk, color, icon = "MEDIUM RISK", AMBER, "🟡"
    else:
        risk, color, icon = "LOW RISK",    GREEN, "🟢"
    return {"prob": prob, "risk": risk, "color": color, "icon": icon}


# ── Chart helpers (matplotlib only) ──────────────────────────────────────────
def gauge_fig(prob, color):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#1a3060", lw=18, solid_capstyle="round")
    # Green zone
    t_g = np.linspace(np.pi, np.pi * 0.65, 100)
    ax.plot(np.cos(t_g), np.sin(t_g), color=GREEN, lw=18, alpha=0.3, solid_capstyle="round")
    # Amber zone
    t_a = np.linspace(np.pi * 0.65, np.pi * 0.35, 100)
    ax.plot(np.cos(t_a), np.sin(t_a), color=AMBER, lw=18, alpha=0.3, solid_capstyle="round")
    # Red zone
    t_r = np.linspace(np.pi * 0.35, 0, 100)
    ax.plot(np.cos(t_r), np.sin(t_r), color=RED,   lw=18, alpha=0.3, solid_capstyle="round")
    # Value arc
    t_v = np.linspace(np.pi, np.pi * (1 - prob), 200)
    ax.plot(np.cos(t_v), np.sin(t_v), color=color, lw=18, solid_capstyle="round")
    # Needle
    angle = np.pi * (1 - prob)
    ax.plot([0, 0.72 * np.cos(angle)], [0, 0.72 * np.sin(angle)],
            color=CREAM, lw=2.5, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.06, color=CREAM, zorder=5))
    # Text
    ax.text(0, -0.25, f"{prob*100:.1f}%",
            ha="center", va="center", fontsize=22, fontweight="bold", color=CREAM)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.5, 1.2)
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    return fig


def importance_fig(model, feat_names):
    imp   = model.feature_importances_
    idx   = np.argsort(imp)[-8:]
    names = [feat_names[i] for i in idx]
    vals  = imp[idx]
    mean  = vals.mean()
    colors = [GOLD if v >= mean else "#3a5a8a" for v in vals]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(names, vals, color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("Importance", fontsize=9)
    ax.axvline(mean, color=GOLD2, lw=1, linestyle="--", alpha=0.6, label="Mean")
    ax.legend(fontsize=8, framealpha=0.2)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    return fig


def confusion_fig(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="YlOrBr", aspect="auto")
    labels = ["Not Churned", "Churned"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual",    fontsize=9)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color=NAVY if cm[i, j] > cm.max() / 2 else CREAM)
    plt.tight_layout(pad=0.5)
    return fig


def metrics_fig(metrics):
    keys   = list(metrics.keys())
    vals   = list(metrics.values())
    colors = [GREEN, GOLD, AMBER, RED, "#6C8EBF"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bars = ax.bar(keys, vals, color=colors, edgecolor="none", width=0.5)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=CREAM)
    plt.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Downloading dataset & training model — ~20 sec on first run…"):
    try:
        model, scaler, le, feat_names, metrics, cm = train_model()
        ready = True
    except Exception as e:
        st.error(f"Training failed: {e}")
        ready = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="badge">🏦 Retail Banking · Gradient Boosting</div>
  <h1>Customer Churn Predictor</h1>
  <p>Meru University of Science & Technology &nbsp;·&nbsp; BSc Data Science &nbsp;·&nbsp;
     Finley Barongo Magembe &nbsp;·&nbsp; CT204/109437/22</p>
</div>
""", unsafe_allow_html=True)

# ── Metric strip ──────────────────────────────────────────────────────────────
if ready:
    c1, c2, c3, c4, c5 = st.columns(5)
    clrs = [GREEN, GOLD, AMBER, RED, "#6C8EBF"]
    for col, (k, v), clr in zip([c1,c2,c3,c4,c5], metrics.items(), clrs):
        with col:
            st.markdown(f"""
            <div class="mbox" style="border-top:3px solid {clr};">
              <div class="lbl">{k}</div>
              <div class="val" style="color:{clr};">{v:.3f}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'DM Serif Display\',serif;font-size:1.3rem;color:#E8C97A;margin-bottom:1.2rem;">📋 Customer Profile</p>', unsafe_allow_html=True)
    st.markdown("**Demographics**")
    geography = st.selectbox("Geography",       ["France", "Germany", "Spain"])
    gender    = st.selectbox("Gender",          ["Female", "Male"])
    age       = st.slider("Age",                18, 92, 42)
    st.markdown("---")
    st.markdown("**Account Details**")
    credit_score  = st.slider("Credit Score",   300, 850, 620)
    tenure        = st.slider("Tenure (years)", 0, 10, 3)
    balance       = st.number_input("Account Balance ($)",  0.0, 300000.0, 130000.0, step=1000.0)
    num_products  = st.selectbox("Number of Products",      [1, 2, 3, 4])
    estimated_sal = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 82000.0,  step=1000.0)
    st.markdown("---")
    st.markdown("**Engagement**")
    has_cr_card = st.radio("Has Credit Card?",  ["Yes", "No"], horizontal=True)
    is_active   = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Predict Churn Risk", use_container_width=True)

customer = {
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

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  🎯 Prediction  ", "  📊 Performance  ", "  ℹ️ About  "])

with tab1:
    if not ready:
        st.warning("Model not ready.")
    else:
        res   = predict_customer(customer, model, scaler, le, feat_names)
        prob  = res["prob"]
        color = res["color"]
        risk  = res["risk"]
        icon  = res["icon"]

        col_g, col_m = st.columns([1, 1.2], gap="large")

        with col_g:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Churn Probability</div>', unsafe_allow_html=True)
            st.pyplot(gauge_fig(prob, color), use_container_width=True)
            st.markdown(f'<div style="text-align:center;font-size:1.1rem;font-weight:700;color:{color};margin-top:0.3rem;">{icon} {risk}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_m:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Customer Summary</div>', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="mbox"><div class="lbl">Credit Score</div><div class="val">{credit_score}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="mbox"><div class="lbl">Age</div><div class="val">{age}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            m3, m4 = st.columns(2)
            with m3:
                st.markdown(f'<div class="mbox"><div class="lbl">Balance</div><div class="val">${balance:,.0f}</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="mbox"><div class="lbl">Tenure</div><div class="val">{tenure} yrs</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if prob >= 0.60:
                rec = "⚠️ Immediate retention action needed. Consider a loyalty bonus, interest rate review, or dedicated relationship manager."
            elif prob >= 0.35:
                rec = "📌 Monitor closely. Engage with product upgrade suggestions or a satisfaction survey within 30 days."
            else:
                rec = "✅ Customer appears stable. Maintain standard engagement and periodic check-ins."

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border-left:3px solid {color};
                border-radius:0 8px 8px 0;padding:.8rem 1rem;font-size:.85rem;color:rgba(247,243,236,.85);">
                <strong style="color:{color}">Recommendation</strong><br>{rec}
            </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        col_fi, col_cm = st.columns(2, gap="large")
        with col_fi:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Top Feature Importances</div>', unsafe_allow_html=True)
            st.pyplot(importance_fig(model, feat_names), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col_cm:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Confusion Matrix (Test Set)</div>', unsafe_allow_html=True)
            st.pyplot(confusion_fig(cm), use_container_width=True)
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f'<p style="font-size:.78rem;color:rgba(247,243,236,.5);margin:.5rem 0 0;">TP: {tp} &nbsp; TN: {tn} &nbsp; FP: {fp} &nbsp; FN: {fn}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    if ready:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Model Performance — Test Set (20%)</div>', unsafe_allow_html=True)
        st.pyplot(metrics_fig(metrics), use_container_width=True)
        perf_df = pd.DataFrame([metrics])
        st.dataframe(perf_df.style.format("{:.4f}").background_gradient(cmap="YlGn", axis=1),
                     use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="card">
      <div class="card-title">About This Application</div>
      <p style="color:rgba(247,243,236,.75);line-height:1.7;">
        This system uses a <strong style="color:#E8C97A">Gradient Boosting Classifier</strong>
        for customer churn prediction in retail banking, developed as part of a
        BSc Data Science research project at
        <strong style="color:#E8C97A">Meru University of Science and Technology</strong>.
        The model is trained automatically on startup — no uploads or model files needed.
      </p><br>
      <p style="font-family:'DM Serif Display',serif;color:#E8C97A;margin-bottom:.5rem;">Research Objectives</p>
      <ul style="color:rgba(247,243,236,.75);line-height:2;">
        <li>✅ Auto-downloads and preprocesses 10,000 banking records</li>
        <li>✅ Trains Gradient Boosting (200 estimators, depth 4, LR 0.05)</li>
        <li>✅ Evaluates with Accuracy, Precision, Recall, F1, AUC-ROC</li>
        <li>✅ Feature importance for explainability</li>
        <li>✅ Real-time single-customer churn prediction</li>
      </ul><br>
      <p style="font-family:'DM Serif Display',serif;color:#E8C97A;margin-bottom:.5rem;">Tech Stack</p>
      <p style="color:rgba(247,243,236,.6);font-size:.85rem;">
        Python · Scikit-learn · Streamlit · Matplotlib · Pandas · NumPy
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<hr><p style="text-align:center;color:rgba(247,243,236,.3);font-size:.78rem;">
Finley Barongo Magembe · CT204/109437/22 · Meru University of Science and Technology · 2026
</p>""", unsafe_allow_html=True)
