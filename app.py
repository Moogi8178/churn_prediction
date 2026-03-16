"""
Customer Churn Prediction — Streamlit App
Meru University of Science & Technology | Finley Barongo Magembe | CT204/109437/22
Two pages: (1) Prediction  (2) Model Performance
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:#0B1D3A; --navy2:#0f2548; --card:#0f2548;
    --gold:#C9A84C; --gold2:#E8C97A;
    --cream:#FFFFFF; --text:#E8EDF5; --muted:#A8B8D0;
    --red:#E85555; --green:#3DBE8A; --amber:#F0922B; --border:#2A4A72;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--navy) !important;
    color: var(--text) !important;
}
.main .block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1200px; }

/* Header */
.app-header {
    background: linear-gradient(135deg,#0f2548 0%,#1a3a6a 50%,#0f2548 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--gold);
    border-radius: 16px;
    padding: 1.8rem 2.5rem;
    margin-bottom: 1.8rem;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #FFE08A !important;
    margin: 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 12px rgba(232,201,122,0.15);
}
.app-header p { color: var(--muted) !important; font-size: 0.88rem; margin: 0; }

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #FFE08A;
    margin-bottom: 1rem;
    padding-bottom: .5rem;
    border-bottom: 1px solid var(--border);
    letter-spacing: 0.2px;
}

/* Metric boxes */
.mbox {
    background: #152d55;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.mbox .lbl {
    font-size:.72rem; text-transform:uppercase; letter-spacing:.1em;
    color:var(--muted); margin-bottom:.4rem; font-weight:500;
}
.mbox .val { font-family:'DM Serif Display',serif; font-size:1.8rem; color:var(--gold2); line-height:1; }

/* Verdict box */
.verdict-yes {
    background: #3a0f0f;
    border: 2px solid var(--red);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.verdict-no {
    background: #0f2d1f;
    border: 2px solid var(--green);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.verdict-label {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.verdict-sub { font-size: 1rem; font-weight: 600; letter-spacing: 0.05em; }

/* Explanation items */
.exp-item {
    background: #152d55;
    border-left: 4px solid var(--gold);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.5;
}
.exp-item.risk  { border-color: var(--red); }
.exp-item.safe  { border-color: var(--green); }
.exp-item.warn  { border-color: var(--amber); }

/* Sidebar */
[data-testid="stSidebar"] { background-color:#091830!important; border-right:2px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p { color:#D0DCEA!important; font-size:0.9rem!important; font-weight:500!important; }
[data-testid="stSlider"] label { color:#D0DCEA!important; font-weight:500!important; }
[data-testid="stSelectbox"] label { color:#D0DCEA!important; font-weight:500!important; }
[data-testid="stNumberInput"] label { color:#D0DCEA!important; font-weight:500!important; }
[data-testid="stNumberInput"] input { background:#152d55!important; color:var(--cream)!important; border-color:var(--border)!important; font-weight:500!important; }
[data-testid="stRadio"] label { color:#D0DCEA!important; font-weight:500!important; }
div[data-baseweb="select"]>div { background:#152d55!important; border-color:var(--border)!important; }
div[data-baseweb="select"] span { color:var(--cream)!important; font-weight:500!important; }

/* Button */
.stButton>button {
    background: linear-gradient(135deg, var(--gold), var(--gold2)) !important;
    color: var(--navy) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .65rem 2rem !important;
    font-size: 1rem !important;
    width: 100%;
}
.stButton>button:hover { opacity: 0.88; }

/* Nav pills (page selector) */
div[data-testid="stRadio"] > div { flex-direction: row !important; gap: 12px; }
div[data-testid="stRadio"] > div > label {
    background: #152d55 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    cursor: pointer;
}
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: #1e3f73 !important;
    border-color: var(--gold) !important;
    color: var(--gold2) !important;
}

hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
p, span { color: var(--text); }
strong { color: var(--cream) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COLOURS FOR MATPLOTLIB
# ══════════════════════════════════════════════════════════════════════════════
NAVY  = "#0B1D3A"
GOLD  = "#C9A84C"
GOLD2 = "#E8C97A"
CREAM = "#E8EDF5"
RED   = "#E85555"
GREEN = "#3DBE8A"
AMBER = "#F0922B"

plt.rcParams.update({
    "figure.facecolor": NAVY,
    "axes.facecolor":   "#0f2548",
    "axes.edgecolor":   "#2A4A72",
    "axes.labelcolor":  CREAM,
    "xtick.color":      CREAM,
    "ytick.color":      CREAM,
    "text.color":       CREAM,
    "grid.color":       "#1e3a5a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL
# ══════════════════════════════════════════════════════════════════════════════
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
    X, y   = df.drop(columns=[target]), df[target]
    feat_names = X.columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

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
    will_churn = prob >= 0.50          # binary threshold
    return prob, will_churn


# ══════════════════════════════════════════════════════════════════════════════
# EXPLANATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def build_explanation(customer, prob, will_churn):
    """
    Generate human-readable factor explanations based on the
    customer profile and domain knowledge of churn drivers.
    Returns list of (type, text) where type = 'risk'|'safe'|'warn'.
    """
    factors = []
    c = customer

    # ── Age ──────────────────────────────────────────────────────────────────
    if c["Age"] >= 50:
        factors.append(("risk",
            f"🔴 Age ({c['Age']} years): Older customers (50+) show significantly higher "
            f"churn rates — they are more likely to switch to competitor banks offering "
            f"better senior-focused products."))
    elif c["Age"] <= 30:
        factors.append(("warn",
            f"🟡 Age ({c['Age']} years): Younger customers tend to be less loyal and "
            f"more open to switching banks for better digital features or rates."))
    else:
        factors.append(("safe",
            f"🟢 Age ({c['Age']} years): Middle-aged customers typically show average "
            f"retention rates — no major age-related churn risk."))

    # ── Active member ─────────────────────────────────────────────────────────
    if c["IsActiveMember"] == 0:
        factors.append(("risk",
            "🔴 Inactive Member: This customer is NOT an active member. "
            "Inactive customers are 2–3× more likely to churn — disengagement "
            "is one of the strongest predictors of leaving."))
    else:
        factors.append(("safe",
            "🟢 Active Member: This customer is actively using bank services. "
            "Active engagement is a strong retention signal."))

    # ── Number of products ────────────────────────────────────────────────────
    if c["NumOfProducts"] == 1:
        factors.append(("risk",
            "🔴 Single Product: Holding only 1 bank product indicates low engagement. "
            "Customers with more products have stronger ties to the bank and are "
            "significantly less likely to leave."))
    elif c["NumOfProducts"] == 2:
        factors.append(("safe",
            "🟢 Two Products: Holding 2 products is associated with good retention — "
            "multi-product customers have greater switching costs."))
    elif c["NumOfProducts"] >= 3:
        factors.append(("warn",
            f"🟡 {c['NumOfProducts']} Products: Holding 3+ products can indicate over-commitment "
            f"and dissatisfaction — some customers with many products still churn if service quality drops."))

    # ── Geography ─────────────────────────────────────────────────────────────
    if c["Geography"] == "Germany":
        factors.append(("risk",
            "🔴 Geography (Germany): German customers in this dataset have the highest "
            "churn rate (~32%) compared to France (~16%) and Spain (~17%). "
            "This may reflect more competitive local banking alternatives."))
    elif c["Geography"] == "Spain":
        factors.append(("warn",
            "🟡 Geography (Spain): Spanish customers show moderate churn rates (~17%). "
            "No strong geographic risk signal."))
    else:
        factors.append(("safe",
            "🟢 Geography (France): French customers show the lowest churn rate (~16%) "
            "in this dataset — a mild protective factor."))

    # ── Credit score ──────────────────────────────────────────────────────────
    if c["CreditScore"] < 500:
        factors.append(("risk",
            f"🔴 Credit Score ({c['CreditScore']}): A low credit score suggests financial "
            f"stress or poor banking history, both linked to higher churn probability."))
    elif c["CreditScore"] >= 700:
        factors.append(("safe",
            f"🟢 Credit Score ({c['CreditScore']}): A strong credit score indicates financial "
            f"stability — these customers are generally more satisfied and less likely to churn."))
    else:
        factors.append(("warn",
            f"🟡 Credit Score ({c['CreditScore']}): Average credit score — moderate churn signal."))

    # ── Balance ───────────────────────────────────────────────────────────────
    if c["Balance"] == 0:
        factors.append(("warn",
            "🟡 Zero Balance: A $0 account balance may indicate dormant usage — "
            "customers who do not actively use their account are at moderate churn risk."))
    elif c["Balance"] > 150000:
        factors.append(("warn",
            f"🟡 High Balance (${c['Balance']:,.0f}): Surprisingly, very high balances can "
            f"correlate with churn — high-value customers may have higher expectations "
            f"and leave if service does not meet them."))
    else:
        factors.append(("safe",
            f"🟢 Balance (${c['Balance']:,.0f}): A moderate account balance suggests "
            f"normal banking activity — no strong balance-related churn signal."))

    # ── Tenure ────────────────────────────────────────────────────────────────
    if c["Tenure"] <= 1:
        factors.append(("risk",
            f"🔴 Short Tenure ({c['Tenure']} year): New customers are at higher risk — "
            f"the first 1–2 years are the most critical period for customer retention."))
    elif c["Tenure"] >= 7:
        factors.append(("safe",
            f"🟢 Long Tenure ({c['Tenure']} years): Long-standing customers are far less "
            f"likely to leave — loyalty deepens over time."))
    else:
        factors.append(("warn",
            f"🟡 Tenure ({c['Tenure']} years): Mid-range tenure — neither strongly loyal "
            f"nor at high early-departure risk."))

    # ── Gender ────────────────────────────────────────────────────────────────
    if c["Gender"] == "Female":
        factors.append(("warn",
            "🟡 Gender (Female): Female customers in this dataset churn at a slightly "
            "higher rate (~25%) than male customers (~16%). This may reflect product "
            "or service gaps in female-targeted offerings."))
    else:
        factors.append(("safe",
            "🟢 Gender (Male): Male customers show a slightly lower churn rate in "
            "this dataset (~16%). Mild protective factor."))

    # ── Credit card ───────────────────────────────────────────────────────────
    if c["HasCrCard"] == 0:
        factors.append(("warn",
            "🟡 No Credit Card: Not holding a bank credit card reduces product ties, "
            "slightly increasing the likelihood of switching to a competitor."))
    else:
        factors.append(("safe",
            "🟢 Has Credit Card: Holding a bank-issued credit card creates an additional "
            "product tie — a mild retention factor."))

    # Sort: risks first, then warnings, then safe
    order = {"risk": 0, "warn": 1, "safe": 2}
    factors.sort(key=lambda x: order[x[0]])
    return factors


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def importance_fig(model, feat_names):
    imp   = model.feature_importances_
    idx   = np.argsort(imp)[-8:]
    names = [feat_names[i] for i in idx]
    vals  = imp[idx]
    mean  = vals.mean()
    colors = [GOLD if v >= mean else "#3a5a8a" for v in vals]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh(names, vals, color=colors, edgecolor="none", height=0.6)
    ax.axvline(mean, color=GOLD2, lw=1.2, linestyle="--", alpha=0.7, label="Mean")
    ax.set_xlabel("Importance", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.2)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    return fig


def confusion_fig(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="YlOrBr", aspect="auto")
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
    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(keys, vals, color=colors, edgecolor="none", width=0.5)
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=CREAM)
    plt.tight_layout(pad=0.5)
    return fig


def prob_bar_fig(prob, will_churn):
    color = RED if will_churn else GREEN
    fig, ax = plt.subplots(figsize=(5, 0.9))
    ax.barh([0], [1],   color="#1e3a5a", height=0.5)
    ax.barh([0], [prob], color=color,    height=0.5)
    ax.axvline(0.5, color=GOLD2, lw=2, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%\n(threshold)", "75%", "100%"], fontsize=8)
    ax.grid(False)
    ax.text(prob + 0.01, 0, f"{prob*100:.1f}%", va="center",
            color=color, fontsize=11, fontweight="bold")
    plt.tight_layout(pad=0.3)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LOAD / TRAIN
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Loading dataset & training model — first run takes ~20 seconds…"):
    try:
        model, scaler, le, feat_names, metrics, cm = train_model()
        ready = True
    except Exception as e:
        st.error(f"Training failed: {e}")
        ready = False

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <h1>Customer Churn</h1>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE NAVIGATION  (2 pages via sidebar radio)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'DM Serif Display\',serif;font-size:1.2rem;'
        'color:#E8C97A;margin-bottom:1rem;">📌 Navigation</p>',
        unsafe_allow_html=True
    )
    page = st.radio("", ["🎯  Prediction", "📊  Model Performance"], label_visibility="collapsed")
    st.markdown("---")

    # ── Customer inputs ───────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:\'DM Serif Display\',serif;font-size:1.1rem;'
        'color:#E8C97A;margin-bottom:0.8rem;">📋 Customer Profile</p>',
        unsafe_allow_html=True
    )
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
    predict_btn = st.button("🔍  Run Prediction", use_container_width=True)

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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "🎯" in page:
    if not ready:
        st.warning("Model not ready. Please check your internet connection and reload.")
    else:
        prob, will_churn = predict_customer(customer, model, scaler, le, feat_names)
        factors          = build_explanation(customer, prob, will_churn)

        # ── SECTION 1: Binary verdict ─────────────────────────────────────────
        st.markdown('<h2 style="font-family:DM Serif Display,serif;color:#E8C97A;font-size:1.7rem;margin-bottom:1rem;border-bottom:2px solid #2A4A72;padding-bottom:0.5rem;">🔎 Prediction Result</h2>', unsafe_allow_html=True)

        col_v, col_p = st.columns([1, 1.6], gap="large")

        with col_v:
            if will_churn:
                st.markdown(f"""
                <div class="verdict-yes">
                  <div class="verdict-label" style="color:#E85555;">YES</div>
                  <div class="verdict-sub"   style="color:#E85555;">This customer WILL CHURN</div>
                  <div style="margin-top:1rem;font-size:0.85rem;color:#c0b0b0;">
                    The model predicts with <strong style="color:#E85555;">{prob*100:.1f}%</strong>
                    probability that this customer will leave the bank.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-no">
                  <div class="verdict-label" style="color:#3DBE8A;">NO</div>
                  <div class="verdict-sub"   style="color:#3DBE8A;">This customer will NOT CHURN</div>
                  <div style="margin-top:1rem;font-size:0.85rem;color:#b0c0b8;">
                    The model predicts with <strong style="color:#3DBE8A;">{(1-prob)*100:.1f}%</strong>
                    confidence that this customer will remain with the bank.
                  </div>
                </div>""", unsafe_allow_html=True)

        with col_p:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Churn Probability Score</div>', unsafe_allow_html=True)
            st.pyplot(prob_bar_fig(prob, will_churn), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="mbox"><div class="lbl">Probability</div>'
                            f'<div class="val" style="color:{"#E85555" if will_churn else "#3DBE8A"};">'
                            f'{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
            with c2:
                verdict_text = "CHURN" if will_churn else "RETAIN"
                verdict_col  = "#E85555" if will_churn else "#3DBE8A"
                st.markdown(f'<div class="mbox"><div class="lbl">Verdict</div>'
                            f'<div class="val" style="font-size:1.2rem;color:{verdict_col};">'
                            f'{verdict_text}</div></div>', unsafe_allow_html=True)
            with c3:
                risk_label = "High" if prob >= 0.6 else ("Medium" if prob >= 0.35 else "Low")
                risk_col   = "#E85555" if prob >= 0.6 else ("#F0922B" if prob >= 0.35 else "#3DBE8A")
                st.markdown(f'<div class="mbox"><div class="lbl">Risk Level</div>'
                            f'<div class="val" style="font-size:1.3rem;color:{risk_col};">'
                            f'{risk_label}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── SECTION 2: Factor-by-factor explanation ───────────────────────────
        st.markdown('<h2 style="font-family:DM Serif Display,serif;color:#E8C97A;font-size:1.7rem;margin-bottom:0.5rem;border-bottom:2px solid #2A4A72;padding-bottom:0.5rem;">📋 Prediction Explanation</h2>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:#A8B8D0;font-size:0.9rem;margin-bottom:1rem;">'
            f'The following factors from this customer\'s profile influenced the prediction. '
            f'Risk factors are shown first.</p>',
            unsafe_allow_html=True
        )

        col_exp1, col_exp2 = st.columns(2, gap="large")
        half = (len(factors) + 1) // 2

        with col_exp1:
            for ftype, ftext in factors[:half]:
                st.markdown(f'<div class="exp-item {ftype}">{ftext}</div>',
                            unsafe_allow_html=True)
        with col_exp2:
            for ftype, ftext in factors[half:]:
                st.markdown(f'<div class="exp-item {ftype}">{ftext}</div>',
                            unsafe_allow_html=True)

        st.markdown("---")

        # ── SECTION 3: Retention recommendation ──────────────────────────────
        st.markdown('<h2 style="font-family:DM Serif Display,serif;color:#E8C97A;font-size:1.7rem;margin-bottom:1rem;border-bottom:2px solid #2A4A72;padding-bottom:0.5rem;">💡 Retention Recommendation</h2>', unsafe_allow_html=True)

        if will_churn:
            rec_color = "#E85555"
            rec_items = []
            if customer["IsActiveMember"] == 0:
                rec_items.append("📲 Launch an immediate re-engagement campaign — personalised email or call from a relationship manager.")
            if customer["NumOfProducts"] == 1:
                rec_items.append("🎁 Offer a cross-sell incentive — e.g. a credit card, savings plan or mortgage consultation.")
            if customer["Geography"] == "Germany":
                rec_items.append("🌍 Review Germany-specific pricing and service offerings — competitive pressure is highest there.")
            if customer["Age"] >= 50:
                rec_items.append("👴 Consider a senior loyalty programme or preferential interest rates for this age group.")
            if customer["Tenure"] <= 1:
                rec_items.append("🤝 Assign a dedicated onboarding advisor — early-tenure churn can be prevented with personalised support.")
            if not rec_items:
                rec_items.append("⚠️ This customer shows multiple churn risk signals. A proactive outreach call is strongly recommended.")
            rec_items.append("📊 Flag this customer for the retention team's priority watchlist.")
        else:
            rec_color = "#3DBE8A"
            rec_items = [
                "✅ This customer is likely to stay — maintain standard service quality.",
                "📧 Include in periodic satisfaction surveys to monitor any change in sentiment.",
                "💼 Consider upselling an additional product to deepen their relationship with the bank.",
            ]

        st.markdown(f'<div class="card" style="border-left:4px solid {rec_color};">', unsafe_allow_html=True)
        for item in rec_items:
            st.markdown(
                f'<div style="padding:0.5rem 0;border-bottom:1px solid #1e3a5a;'
                f'font-size:0.92rem;color:#E8EDF5;">{item}</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "📊" in page:
    if not ready:
        st.warning("Model not ready.")
    else:
        st.markdown('<h2 style="font-family:DM Serif Display,serif;color:#E8C97A;font-size:1.7rem;margin-bottom:0.5rem;border-bottom:2px solid #2A4A72;padding-bottom:0.5rem;">📊 Model Performance</h2>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#A8B8D0;font-size:0.9rem;margin-bottom:1.5rem;">'
            'Evaluated on a held-out test set (20% of 10,000 records). '
            'Model trained with 200 estimators, max depth 4, learning rate 0.05.</p>',
            unsafe_allow_html=True
        )

        # Metric cards
        clrs = [GREEN, GOLD, AMBER, RED, "#6C8EBF"]
        cols = st.columns(5)
        for col, (k, v), clr in zip(cols, metrics.items(), clrs):
            with col:
                st.markdown(f"""
                <div class="mbox" style="border-top:3px solid {clr};">
                  <div class="lbl">{k}</div>
                  <div class="val" style="color:{clr};">{v:.3f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_mc, col_cf = st.columns(2, gap="large")

        with col_mc:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Performance Metrics Bar Chart</div>', unsafe_allow_html=True)
            st.pyplot(metrics_fig(metrics), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_cf:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Confusion Matrix</div>', unsafe_allow_html=True)
            st.pyplot(confusion_fig(cm), use_container_width=True)
            tn, fp, fn, tp = cm.ravel()
            st.markdown(
                f'<p style="font-size:.82rem;color:#A8B8D0;margin:.6rem 0 0;">'
                f'True Positives (correctly predicted churn): <strong>{tp}</strong> &nbsp;|&nbsp; '
                f'True Negatives (correctly predicted retain): <strong>{tn}</strong><br>'
                f'False Positives (wrongly flagged as churn): <strong>{fp}</strong> &nbsp;|&nbsp; '
                f'False Negatives (missed churners): <strong>{fn}</strong></p>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature importance
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Top 8 Feature Importances (Model-Learned)</div>', unsafe_allow_html=True)
        st.pyplot(importance_fig(model, feat_names), use_container_width=True)
        st.markdown(
            '<p style="font-size:.82rem;color:#A8B8D0;margin:.5rem 0 0;">'
            'Gold bars = above-average importance. These are the features the model relies on most when making predictions.</p>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # About the model
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">About the Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <table style="width:100%;border-collapse:collapse;font-size:0.88rem;color:#E8EDF5;">
          <tr style="border-bottom:1px solid #2A4A72;">
            <td style="padding:0.6rem;color:#A8B8D0;width:35%;">Algorithm</td>
            <td style="padding:0.6rem;">Gradient Boosting Classifier (Scikit-learn)</td>
          </tr>
          <tr style="border-bottom:1px solid #2A4A72;">
            <td style="padding:0.6rem;color:#A8B8D0;">Estimators</td>
            <td style="padding:0.6rem;">200 decision trees</td>
          </tr>
          <tr style="border-bottom:1px solid #2A4A72;">
            <td style="padding:0.6rem;color:#A8B8D0;">Max Depth</td>
            <td style="padding:0.6rem;">4 levels per tree</td>
          </tr>
          <tr style="border-bottom:1px solid #2A4A72;">
            <td style="padding:0.6rem;color:#A8B8D0;">Learning Rate</td>
            <td style="padding:0.6rem;">0.05</td>
          </tr>
          <tr style="border-bottom:1px solid #2A4A72;">
            <td style="padding:0.6rem;color:#A8B8D0;">Training Data</td>
            <td style="padding:0.6rem;">8,000 customer records (80% split)</td>
          </tr>
          <tr style="border-bottom:1px solid #2A4A72;">
            <td style="padding:0.6rem;color:#A8B8D0;">Test Data</td>
            <td style="padding:0.6rem;">2,000 customer records (20% split)</td>
          </tr>
          <tr>
            <td style="padding:0.6rem;color:#A8B8D0;">Prediction Threshold</td>
            <td style="padding:0.6rem;">50% probability → binary YES / NO</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
