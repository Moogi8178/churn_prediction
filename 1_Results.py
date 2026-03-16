"""
Customer Churn Prediction — Page 2: Results & Explanation
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Prediction Results",
    page_icon="📊",
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
.main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* Header */
.page-header {
    background: linear-gradient(135deg, #0f2548 0%, #1a3a6a 50%, #0f2548 100%);
    border: 1px solid var(--border);
    border-left: 5px solid var(--gold2);
    border-radius: 16px;
    padding: 1.6rem 2.5rem;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: var(--gold2) !important;
    margin: 0;
}

/* Section titles */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--gold2);
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
}

/* Verdict */
.verdict-yes {
    background: linear-gradient(135deg, #3a0f0f, #2a0a0a);
    border: 2px solid var(--red);
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
}
.verdict-no {
    background: linear-gradient(135deg, #0a2d1f, #082214);
    border: 2px solid var(--green);
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
}
.verdict-word {
    font-family: 'DM Serif Display', serif;
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.verdict-desc { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.04em; }
.verdict-prob { font-size: 0.9rem; margin-top: 1rem; color: #b0b8c8; }

/* Probability bar */
.prob-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.8rem;
    height: 100%;
}
.prob-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--gold2);
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* Metric mini boxes */
.mbox {
    background: #152d55;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.mbox .lbl { font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-bottom:.3rem; font-weight:500; }
.mbox .val { font-family:'DM Serif Display',serif; font-size:1.6rem; line-height:1; }

/* Explanation cards */
.exp-risk {
    background: #2d0f0f;
    border-left: 4px solid var(--red);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--text);
}
.exp-warn {
    background: #2d1a08;
    border-left: 4px solid var(--amber);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--text);
}
.exp-safe {
    background: #082d18;
    border-left: 4px solid var(--green);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--text);
}

/* Recommendation card */
.rec-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
}
.rec-item {
    padding: 0.65rem 0;
    border-bottom: 1px solid #1e3a5a;
    font-size: 0.92rem;
    color: var(--text);
    line-height: 1.5;
}
.rec-item:last-child { border-bottom: none; }

/* Summary table */
.summary-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    color: var(--text);
}
.summary-table td { padding: 0.6rem 0.8rem; border-bottom: 1px solid #1e3a5a; }
.summary-table td:first-child { color: var(--muted); width: 45%; font-weight: 500; }

/* Back button */
.stButton > button {
    background: #152d55 !important;
    color: var(--gold2) !important;
    font-weight: 700 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover { background: #1e3f73 !important; border-color: var(--gold) !important; }

hr { border-color: var(--border) !important; }

/* Hide default sidebar nav */
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Guard: must arrive from page 1 ───────────────────────────────────────────
if "customer" not in st.session_state or "model" not in st.session_state:
    st.warning("⚠️ Please go back and fill in the customer details first.")
    if st.button("← Go to Input Form"):
        st.switch_page("app.py")
    st.stop()

# ── Retrieve from session ─────────────────────────────────────────────────────
customer   = st.session_state["customer"]
model      = st.session_state["model"]
scaler     = st.session_state["scaler"]
le         = st.session_state["le"]
feat_names = st.session_state["feat_names"]

# ── Run prediction ────────────────────────────────────────────────────────────
def predict(customer, model, scaler, le, feat_names):
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
    return prob, prob >= 0.50

prob, will_churn = predict(customer, model, scaler, le, feat_names)

# ── Explanation engine ────────────────────────────────────────────────────────
def build_explanation(c):
    factors = []

    # Active member
    if c["IsActiveMember"] == 0:
        factors.append(("risk", "🔴 Inactive Member — This customer is NOT actively using bank services. Inactive customers are 2–3× more likely to churn. Disengagement is the single strongest predictor of leaving."))
    else:
        factors.append(("safe", "🟢 Active Member — This customer actively uses bank services. Active engagement is the strongest retention signal in the model."))

    # Age
    if c["Age"] >= 50:
        factors.append(("risk", f"🔴 Age ({c['Age']} yrs) — Customers aged 50+ show significantly higher churn rates. They are more likely to switch to banks offering better senior-focused products or rates."))
    elif c["Age"] <= 30:
        factors.append(("warn", f"🟡 Age ({c['Age']} yrs) — Younger customers tend to be less loyal and more open to switching banks for better digital features or interest rates."))
    else:
        factors.append(("safe", f"🟢 Age ({c['Age']} yrs) — Middle-aged customers show average retention rates. No significant age-related churn risk."))

    # Products
    if c["NumOfProducts"] == 1:
        factors.append(("risk", "🔴 Single Product — Holding only 1 bank product indicates low engagement and few switching costs. Customers with 2+ products are significantly less likely to leave."))
    elif c["NumOfProducts"] == 2:
        factors.append(("safe", "🟢 Two Products — Holding 2 products is strongly associated with retention. Multi-product customers have greater ties to the bank and higher switching costs."))
    else:
        factors.append(("warn", f"🟡 {c['NumOfProducts']} Products — Holding 3+ products sometimes indicates over-commitment. Some high-product customers still churn if service quality drops."))

    # Geography
    if c["Geography"] == "Germany":
        factors.append(("risk", "🔴 Geography: Germany — German customers in this dataset have the highest churn rate (~32%), compared to France (~16%) and Spain (~17%). This reflects a more competitive local banking market."))
    elif c["Geography"] == "Spain":
        factors.append(("warn", "🟡 Geography: Spain — Spanish customers show moderate churn rates (~17%). No strong geographic risk, but not the safest region either."))
    else:
        factors.append(("safe", "🟢 Geography: France — French customers show the lowest churn rate (~16%) in this dataset. A mild protective factor for retention."))

    # Credit score
    if c["CreditScore"] < 500:
        factors.append(("risk", f"🔴 Credit Score ({c['CreditScore']}) — A low credit score suggests financial stress or poor banking history, both of which correlate with higher churn probability."))
    elif c["CreditScore"] >= 700:
        factors.append(("safe", f"🟢 Credit Score ({c['CreditScore']}) — A strong credit score indicates financial stability. These customers are generally more satisfied and less likely to churn."))
    else:
        factors.append(("warn", f"🟡 Credit Score ({c['CreditScore']}) — An average credit score carries moderate churn risk. Not a strong signal in either direction."))

    # Tenure
    if c["Tenure"] <= 1:
        factors.append(("risk", f"🔴 Short Tenure ({c['Tenure']} yr) — New customers are at the highest risk of early departure. The first 1–2 years are the most critical retention period."))
    elif c["Tenure"] >= 7:
        factors.append(("safe", f"🟢 Long Tenure ({c['Tenure']} yrs) — Long-standing customers are far less likely to leave. Loyalty and switching costs both deepen significantly over time."))
    else:
        factors.append(("warn", f"🟡 Tenure ({c['Tenure']} yrs) — Mid-range tenure carries neither strong loyalty nor high early-exit risk."))

    # Balance
    if c["Balance"] == 0:
        factors.append(("warn", "🟡 Zero Balance — A $0 account balance often signals a dormant account. Customers who do not actively use their account are at moderate churn risk."))
    elif c["Balance"] > 150000:
        factors.append(("warn", f"🟡 High Balance (${c['Balance']:,.0f}) — Surprisingly, very high balances can correlate with churn. High-value customers have higher expectations and may leave if service falls short."))
    else:
        factors.append(("safe", f"🟢 Balance (${c['Balance']:,.0f}) — A moderate balance suggests normal banking activity. No strong balance-related churn signal."))

    # Gender
    if c["Gender"] == "Female":
        factors.append(("warn", "🟡 Gender: Female — Female customers in this dataset churn at a slightly higher rate (~25%) than male customers (~16%), possibly reflecting gaps in female-targeted products."))
    else:
        factors.append(("safe", "🟢 Gender: Male — Male customers show a slightly lower churn rate (~16%) in this dataset. A mild protective factor."))

    # Credit card
    if c["HasCrCard"] == 0:
        factors.append(("warn", "🟡 No Credit Card — Not holding a bank-issued credit card reduces product ties, slightly increasing the likelihood of switching to a competitor."))
    else:
        factors.append(("safe", "🟢 Has Credit Card — Holding a bank-issued credit card creates an additional product relationship — a mild but meaningful retention factor."))

    # Sort: risks → warnings → safe
    order = {"risk": 0, "warn": 1, "safe": 2}
    factors.sort(key=lambda x: order[x[0]])
    return factors

factors = build_explanation(customer)

# ── Matplotlib prob bar ───────────────────────────────────────────────────────
def prob_bar(prob, will_churn):
    color = "#E85555" if will_churn else "#3DBE8A"
    fig, ax = plt.subplots(figsize=(5, 1.1), facecolor="#0f2548")
    ax.set_facecolor("#0f2548")
    ax.barh([0], [1],    color="#1e3a5a", height=0.55, edgecolor="none")
    ax.barh([0], [prob], color=color,    height=0.55, edgecolor="none")
    ax.axvline(0.5, color="#FFE08A", lw=2, linestyle="--", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                       fontsize=9, color="#A8B8D0")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.text(min(prob + 0.02, 0.85), 0, f"{prob*100:.1f}%",
            va="center", color=color, fontsize=12, fontweight="bold")
    plt.tight_layout(pad=0.2)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# RENDER PAGE
# ══════════════════════════════════════════════════════════════════════════════

# Back button
if st.button("← Back to Input Form"):
    st.switch_page("app.py")

st.markdown("""
<div class="page-header">
  <h1>Prediction Results</h1>
</div>
""", unsafe_allow_html=True)

# ── SECTION 1: Verdict ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔎 Churn Verdict</div>', unsafe_allow_html=True)

col_v, col_p = st.columns([1, 1.5], gap="large")

with col_v:
    if will_churn:
        st.markdown(f"""
        <div class="verdict-yes">
          <div class="verdict-word" style="color:#E85555;">YES</div>
          <div class="verdict-desc" style="color:#E85555;">This customer WILL CHURN</div>
          <div class="verdict-prob">
            The model predicts a
            <strong style="color:#E85555;">{prob*100:.1f}%</strong>
            probability that this customer will leave the bank.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-no">
          <div class="verdict-word" style="color:#3DBE8A;">NO</div>
          <div class="verdict-desc" style="color:#3DBE8A;">This customer will NOT CHURN</div>
          <div class="verdict-prob">
            The model predicts a
            <strong style="color:#3DBE8A;">{(1-prob)*100:.1f}%</strong>
            confidence that this customer will stay with the bank.
          </div>
        </div>""", unsafe_allow_html=True)

with col_p:
    verdict_col  = "#E85555" if will_churn else "#3DBE8A"
    verdict_txt  = "CHURN"   if will_churn else "RETAIN"
    risk_label   = "High"    if prob >= 0.6 else ("Medium" if prob >= 0.35 else "Low")
    risk_col     = "#E85555" if prob >= 0.6 else ("#F0922B" if prob >= 0.35 else "#3DBE8A")

    st.markdown('<div class="prob-card">', unsafe_allow_html=True)
    st.markdown('<div class="prob-card-title">📈 Churn Probability Score</div>', unsafe_allow_html=True)
    st.pyplot(prob_bar(prob, will_churn), use_container_width=True)
    st.markdown('<p style="font-size:0.78rem;color:#A8B8D0;margin:0.3rem 0 1rem 0;">Dashed line = 50% decision threshold</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="mbox"><div class="lbl">Probability</div><div class="val" style="color:{verdict_col};">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="mbox"><div class="lbl">Verdict</div><div class="val" style="font-size:1.1rem;color:{verdict_col};">{verdict_txt}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="mbox"><div class="lbl">Risk Level</div><div class="val" style="font-size:1.2rem;color:{risk_col};">{risk_label}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SECTION 2: Explanation ────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Why This Prediction?</div>', unsafe_allow_html=True)
st.markdown('<p style="color:#A8B8D0;font-size:0.9rem;margin-bottom:1.2rem;">Each factor below explains how the customer\'s profile influenced the prediction. Risk factors are shown first.</p>', unsafe_allow_html=True)

half = (len(factors) + 1) // 2
col_e1, col_e2 = st.columns(2, gap="large")

css_map = {"risk": "exp-risk", "warn": "exp-warn", "safe": "exp-safe"}
with col_e1:
    for ftype, ftext in factors[:half]:
        st.markdown(f'<div class="{css_map[ftype]}">{ftext}</div>', unsafe_allow_html=True)
with col_e2:
    for ftype, ftext in factors[half:]:
        st.markdown(f'<div class="{css_map[ftype]}">{ftext}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SECTION 3: Recommendation ─────────────────────────────────────────────────
st.markdown('<div class="section-title">💡 Retention Recommendation</div>', unsafe_allow_html=True)

rec_color = "#E85555" if will_churn else "#3DBE8A"
rec_items = []

if will_churn:
    if customer["IsActiveMember"] == 0:
        rec_items.append("📲 Launch an immediate re-engagement campaign — a personalised call or email from a relationship manager.")
    if customer["NumOfProducts"] == 1:
        rec_items.append("🎁 Offer a cross-sell incentive such as a credit card, savings plan, or mortgage consultation to deepen product ties.")
    if customer["Geography"] == "Germany":
        rec_items.append("🌍 Review Germany-specific pricing and service offerings — competitive pressure is highest in this region.")
    if customer["Age"] >= 50:
        rec_items.append("👴 Consider a senior loyalty programme or preferential interest rates tailored to this age group.")
    if customer["Tenure"] <= 1:
        rec_items.append("🤝 Assign a dedicated onboarding advisor — early-tenure churn can be prevented with personalised support in the first year.")
    if customer["Balance"] == 0:
        rec_items.append("💳 Encourage account activation through zero-fee promotions or cashback offers to increase engagement.")
    if not rec_items:
        rec_items.append("⚠️ Multiple churn risk signals detected. A proactive outreach call from a senior relationship manager is strongly recommended.")
    rec_items.append("📊 Flag this customer for the retention team's priority watchlist immediately.")
else:
    rec_items = [
        "✅ This customer is likely to stay — maintain current service quality and engagement cadence.",
        "📧 Include in periodic satisfaction surveys to detect any early warning signs of dissatisfaction.",
        "💼 Consider upselling an additional product to further deepen the customer's relationship with the bank.",
        "🎯 Recognise and reward loyalty — a thank-you message or loyalty benefit can reinforce positive sentiment.",
    ]

st.markdown(f'<div class="rec-card" style="border-left: 4px solid {rec_color};">', unsafe_allow_html=True)
for item in rec_items:
    st.markdown(f'<div class="rec-item">{item}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SECTION 4: Customer Summary ───────────────────────────────────────────────
st.markdown('<div class="section-title">👤 Customer Profile Summary</div>', unsafe_allow_html=True)

c = customer
col_t1, col_t2 = st.columns(2, gap="large")
with col_t1:
    st.markdown(f"""
    <table class="summary-table">
      <tr><td>Geography</td><td>{c['Geography']}</td></tr>
      <tr><td>Gender</td><td>{c['Gender']}</td></tr>
      <tr><td>Age</td><td>{c['Age']} years</td></tr>
      <tr><td>Credit Score</td><td>{c['CreditScore']}</td></tr>
      <tr><td>Tenure</td><td>{c['Tenure']} years</td></tr>
    </table>""", unsafe_allow_html=True)
with col_t2:
    st.markdown(f"""
    <table class="summary-table">
      <tr><td>Account Balance</td><td>${c['Balance']:,.0f}</td></tr>
      <tr><td>Number of Products</td><td>{c['NumOfProducts']}</td></tr>
      <tr><td>Estimated Salary</td><td>${c['EstimatedSalary']:,.0f}</td></tr>
      <tr><td>Has Credit Card</td><td>{'Yes' if c['HasCrCard'] else 'No'}</td></tr>
      <tr><td>Active Member</td><td>{'Yes' if c['IsActiveMember'] else 'No'}</td></tr>
    </table>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("← Predict Another Customer", use_container_width=False):
    st.switch_page("app.py")
