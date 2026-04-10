"""
Customer Churn Prediction
Single-file multipage app with Login, Registration, Admin, Input, and Results pages.
"""

import streamlit as st
import numpy as np
import pandas as pd
import hashlib
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# USER STORE  (in-memory; replace with a DB for production)
# ══════════════════════════════════════════════════════════════════════════════
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

if "users" not in st.session_state:
    # Pre-seeded accounts: username → {password_hash, role, name}
    st.session_state["users"] = {
        "admin": {
            "password_hash": _hash("admin123"),
            "role": "admin",
            "name": "Administrator",
        },
        "analyst": {
            "password_hash": _hash("analyst123"),
            "role": "user",
            "name": "Bank Analyst",
        },
    }

# ── Initialise navigation & auth state ───────────────────────────────────────
for key, default in [
    ("page", "login"),          # login | register | admin | input | results
    ("logged_in", False),
    ("current_user", None),
    ("current_role", None),
    ("auth_error", ""),
    ("reg_success", False),
    ("prediction_log", []),     # list of dicts: {user, timestamp, customer, prob, verdict}
    ("saved_username", ""),     # remember-me: saved username
    ("saved_password", ""),     # remember-me: saved password
    ("remember_me", False),     # remember-me checkbox state
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --navy: #0B1D3A;
    --card: #112244;
    --gold: #C9A84C;
    --gold2: #FFE08A;
    --text: #FFFFFF;
    --muted: #C0CFDF;
    --border: #3A5A8A;
    --red: #FF5555;
    --green: #2ECC8A;
    --amber: #FF9933;
}

html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main,
section[data-testid="stSidebar"],
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] {
    background-color: #0B1D3A !important;
    color: #FFFFFF !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ══════════════════════════════════════════════════
   PER-PAGE BACKGROUND IMAGES  (bank / finance theme)
   Each page gets a unique Unsplash photo with a dark
   navy overlay so all text remains readable.
   ══════════════════════════════════════════════════ */

/* LOGIN page — grand bank building exterior */
[data-page="login"] .stApp,
.stApp[data-page="login"] {
    background-image:
        linear-gradient(135deg, rgba(11,29,58,0.88) 0%, rgba(17,34,68,0.82) 100%),
        url('https://images.unsplash.com/photo-1541354329998-f4d9a9f9297f?w=1920&q=80') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}

/* REGISTER page — modern bank interior lobby */
[data-page="register"] .stApp,
.stApp[data-page="register"] {
    background-image:
        linear-gradient(135deg, rgba(11,29,58,0.88) 0%, rgba(17,34,68,0.82) 100%),
        url('https://images.unsplash.com/photo-1501167786227-4cba60f6d58f?w=1920&q=80') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}

/* INPUT / PREDICT page — financial data / analytics desk */
[data-page="input"] .stApp,
.stApp[data-page="input"] {
    background-image:
        linear-gradient(135deg, rgba(11,29,58,0.90) 0%, rgba(17,34,68,0.85) 100%),
        url('https://images.unsplash.com/photo-1563013544-824ae1b704d3?w=1920&q=80') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}

/* RESULTS page — customer meeting / financial review */
[data-page="results"] .stApp,
.stApp[data-page="results"] {
    background-image:
        linear-gradient(135deg, rgba(11,29,58,0.90) 0%, rgba(17,34,68,0.85) 100%),
        url('https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=1920&q=80') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}

/* ADMIN page — executive dashboard / control room */
[data-page="admin"] .stApp,
.stApp[data-page="admin"] {
    background-image:
        linear-gradient(135deg, rgba(11,29,58,0.92) 0%, rgba(17,34,68,0.88) 100%),
        url('https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=1920&q=80') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
}

/* Universal fallback background (covers all inner containers) */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: transparent !important;
}
[data-testid="block-container"],
.main {
    background: transparent !important;
}
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] {
    background: transparent !important;
}

/* ── BUTTON TEXT FIX: stop white bleeding into button labels ── */
/* Streamlit wraps button text in <p> tags; target them directly */
div.stButton > button { color: #FFE08A !important; }
div.stButton > button p { color: inherit !important; }
div.stButton > button:hover { opacity: 0.9; }

/* Specific overrides per button class */
.predict-btn div.stButton > button,
.predict-btn div.stButton > button p { color: #0B1D3A !important; }

.primary-btn div.stButton > button,
.primary-btn div.stButton > button p { color: #0B1D3A !important; }

.sec-btn div.stButton > button,
.sec-btn div.stButton > button p,
.back-btn div.stButton > button,
.back-btn div.stButton > button p { color: #FFE08A !important; }

.danger-btn div.stButton > button,
.danger-btn div.stButton > button p { color: #FF5555 !important; }

div.stDownloadButton > button { color: #0B1D3A !important; }
div.stDownloadButton > button p { color: #0B1D3A !important; }
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1050px;
    background-color: #0B1D3A !important;
}

/* ── Page Header ── */
.page-header {
    background: #112244;
    border: 2px solid #3A5A8A;
    border-left: 6px solid #FFE08A;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #FFE08A !important;
    margin: 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 16px rgba(255,224,138,0.3);
}
.page-header p {
    color: #C0CFDF !important;
    font-size: 1rem !important;
    font-weight: 400 !important;
    margin: 0.5rem 0 0 0 !important;
}

/* ── Auth Card ── */
.auth-card {
    background-color: #112244;
    border: 1.5px solid #3A5A8A;
    border-radius: 18px;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin: 0 auto;
    max-width: 480px;
}
.auth-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #FFE08A;
    margin-bottom: 0.3rem;
}
.auth-subtitle {
    color: #C0CFDF;
    font-size: 0.95rem;
    margin-bottom: 1.8rem;
}

/* ── Section Titles ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFE08A;
    background-color: #1a3a6a;
    padding: 0.75rem 1.2rem;
    border-left: 6px solid #FFE08A;
    border-radius: 0 8px 8px 0;
    margin: 2rem 0 1.2rem 0;
    display: block;
}

/* ── Input Cards ── */
.input-card {
    background-color: #112244 !important;
    border: 1.5px solid #3A5A8A;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
}

/* ── Text Inputs (login/register) ── */
[data-testid="stTextInput"] input {
    background-color: #1a3560 !important;
    color: #FFFFFF !important;
    border: 1.5px solid #3A5A8A !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #FFE08A !important;
    box-shadow: 0 0 0 2px rgba(255,224,138,0.15) !important;
}

/* ── ALL text labels force white ── */
label,
.stSlider label,
.stSelectbox label,
.stNumberInput label,
.stRadio label,
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label,
[data-testid="stMarkdown"] p,
p {
    color: #FFFFFF !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

[data-testid="stSlider"] span,
[data-testid="stSlider"] p {
    color: #FFE08A !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

div[data-baseweb="select"] > div {
    background-color: #1a3560 !important;
    border: 1.5px solid #3A5A8A !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

[data-testid="stNumberInput"] input {
    background-color: #1a3560 !important;
    color: #FFFFFF !important;
    border: 1.5px solid #3A5A8A !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
[data-testid="stNumberInput"] button {
    background-color: #1a3560 !important;
    color: #FFFFFF !important;
    border-color: #3A5A8A !important;
}

[data-testid="stRadio"] label span { color: #FFFFFF !important; font-weight: 600 !important; font-size: 1rem !important; }
[data-testid="stRadio"] > label    { color: #FFFFFF !important; font-weight: 700 !important; font-size: 1rem !important; }

/* ── Metric Boxes ── */
.mbox {
    background-color: #1a3560;
    border: 1.5px solid #3A5A8A;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.mbox .lbl {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #C0CFDF;
    margin-bottom: 0.4rem;
    font-weight: 700;
    display: block;
}
.mbox .val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    line-height: 1;
    color: #FFE08A;
    display: block;
}

/* ── Admin table ── */
.admin-table { width:100%; border-collapse:collapse; font-size:0.95rem; }
.admin-table th {
    background: #1a3560;
    color: #FFE08A;
    text-transform: uppercase;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    padding: 0.8rem 1rem;
    text-align: left;
    font-weight: 700;
}
.admin-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #1e3a5a;
    color: #FFFFFF;
    vertical-align: middle;
}
.admin-table tr:last-child td { border-bottom: none; }
.badge-admin  { background:#C9A84C22; color:#FFE08A; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.8rem; font-weight:700; border:1px solid #C9A84C; }
.badge-user   { background:#2ECC8A22; color:#2ECC8A; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.8rem; font-weight:700; border:1px solid #2ECC8A; }

/* ── Alert boxes ── */
.alert-error {
    background: #2d0f0f;
    border: 1.5px solid #FF5555;
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    color: #FF9999;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}
.alert-success {
    background: #052214;
    border: 1.5px solid #2ECC8A;
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    color: #2ECC8A;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

/* ── Nav bar ── */
.topnav {
    background: #112244;
    border: 1.5px solid #3A5A8A;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.topnav-brand {
    font-family: 'DM Serif Display', serif;
    color: #FFE08A;
    font-size: 1.15rem;
}
.topnav-user {
    color: #C0CFDF;
    font-size: 0.9rem;
}

/* ── Verdict boxes ── */
.verdict-yes {
    background-color: #2a0808;
    border: 2.5px solid #FF5555;
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,85,85,0.2);
}
.verdict-no {
    background-color: #052214;
    border: 2.5px solid #2ECC8A;
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(46,204,138,0.2);
}
.verdict-word {
    font-family: 'DM Serif Display', serif;
    font-size: 5.5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.verdict-desc { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.04em; }
.verdict-prob { font-size: 0.92rem; margin-top: 1rem; color: #C0CFDF; line-height: 1.5; }

/* ── Prob card ── */
.prob-card {
    background-color: #112244;
    border: 1.5px solid #3A5A8A;
    border-radius: 14px;
    padding: 1.8rem;
}
.prob-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #FFE08A;
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #3A5A8A;
    display: block;
}

/* ── Explanation items ── */
.exp-risk {
    background-color: #2d0f0f;
    border-left: 5px solid #FF5555;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.95rem;
    line-height: 1.65;
    color: #FFFFFF;
}
.exp-warn {
    background-color: #2d1a06;
    border-left: 5px solid #FF9933;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.95rem;
    line-height: 1.65;
    color: #FFFFFF;
}
.exp-safe {
    background-color: #062d16;
    border-left: 5px solid #2ECC8A;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.95rem;
    line-height: 1.65;
    color: #FFFFFF;
}

/* ── Rec card ── */
.rec-card {
    background-color: #112244;
    border: 1.5px solid #3A5A8A;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
}
.rec-item {
    padding: 0.75rem 0;
    border-bottom: 1px solid #1e3a5a;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #FFFFFF;
    font-weight: 500;
}
.rec-item:last-child { border-bottom: none; }

/* ── Summary table ── */
.summary-table { width:100%; border-collapse:collapse; font-size:0.95rem; }
.summary-table td { padding:0.7rem 0.9rem; border-bottom:1px solid #1e3a5a; color:#FFFFFF; }
.summary-table td:first-child { color:#C0CFDF; width:45%; font-weight:700; font-size:0.88rem; text-transform:uppercase; letter-spacing:0.05em; }

/* ── Predict button ── */
.predict-btn div.stButton > button {
    background: linear-gradient(135deg,#C9A84C,#FFE08A) !important;
    color: #0B1D3A !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.9rem 3rem !important;
    width: 100%;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.predict-btn div.stButton > button p,
.predict-btn div.stButton > button span { color: #0B1D3A !important; font-weight: 800 !important; }

/* ── Primary button (login/register) ── */
.primary-btn div.stButton > button {
    background: linear-gradient(135deg,#C9A84C,#FFE08A) !important;
    color: #0B1D3A !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100%;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.primary-btn div.stButton > button p,
.primary-btn div.stButton > button span { color: #0B1D3A !important; font-weight: 800 !important; }

/* ── Back / secondary button ── */
.back-btn div.stButton > button,
.sec-btn div.stButton > button {
    background-color: #1a3560 !important;
    color: #FFE08A !important;
    font-weight: 700 !important;
    border: 1.5px solid #3A5A8A !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.95rem !important;
}
.back-btn div.stButton > button p,
.back-btn div.stButton > button span,
.sec-btn div.stButton > button p,
.sec-btn div.stButton > button span { color: #FFE08A !important; font-weight: 700 !important; }

/* ── Danger button ── */
.danger-btn div.stButton > button {
    background-color: #2d0f0f !important;
    color: #FF5555 !important;
    font-weight: 700 !important;
    border: 1.5px solid #FF5555 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.95rem !important;
}
.danger-btn div.stButton > button p,
.danger-btn div.stButton > button span { color: #FF5555 !important; font-weight: 700 !important; }

/* ── Global fallback: all Streamlit buttons show text ── */
.stButton > button p,
.stButton > button span {
    color: inherit !important;
    font-weight: inherit !important;
}

strong { color: #FFE08A !important; }
hr { border-color: #3A5A8A !important; }
[data-testid="stSidebarNav"]     { display: none; }
[data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — inject page name into DOM so per-page CSS backgrounds can fire
# ══════════════════════════════════════════════════════════════════════════════
def set_page_bg(page_name: str):
    """
    Injects a <script> that sets data-page="<page_name>" on .stApp so the
    per-page CSS background-image rules activate.
    Also injects an inline <style> block as a reliable fallback using a
    direct class applied to stApp.
    """
    bg_map = {
        "login":    ("https://images.unsplash.com/photo-1541354329998-f4d9a9f9297f?w=1920&q=80",  "rgba(11,29,58,0.88), rgba(17,34,68,0.82)"),
        "register": ("https://images.unsplash.com/photo-1501167786227-4cba60f6d58f?w=1920&q=80",  "rgba(11,29,58,0.88), rgba(17,34,68,0.82)"),
        "input":    ("https://images.unsplash.com/photo-1563013544-824ae1b704d3?w=1920&q=80",     "rgba(11,29,58,0.90), rgba(17,34,68,0.85)"),
        "results":  ("https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=1920&q=80",     "rgba(11,29,58,0.90), rgba(17,34,68,0.85)"),
        "admin":    ("https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=1920&q=80",  "rgba(11,29,58,0.92), rgba(17,34,68,0.88)"),
    }
    img_url, gradient = bg_map.get(page_name, bg_map["login"])
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(135deg, {gradient}),
                          url('{img_url}') !important;
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
        background-repeat: no-repeat !important;
    }}
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="block-container"],
    .main,
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"] {{
        background: transparent !important;
    }}
    </style>
    <script>
        const app = window.parent.document.querySelector('.stApp');
        if (app) app.setAttribute('data-page', '{page_name}');
    </script>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — TOP NAV BAR (shown when logged in)
# ══════════════════════════════════════════════════════════════════════════════
def render_nav():
    users = st.session_state["users"]
    user  = st.session_state["current_user"]
    role  = st.session_state["current_role"]
    name  = users[user]["name"] if user else ""

    cols = st.columns([3, 1, 1, 1] if role == "admin" else [3, 1, 1])
    with cols[0]:
        st.markdown(
            f'<div class="topnav-brand">🏦 Churn Predictor'
            f'<span style="color:#C0CFDF;font-size:0.85rem;font-family:\'DM Sans\',sans-serif;font-weight:400;margin-left:0.7rem;">'
            f'Welcome, {name}</span></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown('<div class="sec-btn">', unsafe_allow_html=True)
        if st.button("🔍 Predict", use_container_width=True, key="nav_predict"):
            st.session_state["page"] = "input"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    if role == "admin":
        with cols[2]:
            st.markdown('<div class="sec-btn">', unsafe_allow_html=True)
            if st.button("⚙️ Admin", use_container_width=True, key="nav_admin"):
                st.session_state["page"] = "admin"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button("🚪 Logout", use_container_width=True, key="nav_logout"):
                for k in ["logged_in","current_user","current_role","page","customer"]:
                    st.session_state[k] = False if k == "logged_in" else (None if k in ["current_user","current_role"] else ("login" if k == "page" else st.session_state.get(k)))
                st.session_state["page"] = "login"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        with cols[2]:
            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button("🚪 Logout", use_container_width=True, key="nav_logout"):
                for k in ["logged_in","current_user","current_role","customer"]:
                    st.session_state[k] = False if k == "logged_in" else None
                st.session_state["page"] = "login"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin:0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
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
    drop_cols = [c for c in ["RowNumber","CustomerId","Surname"] if c in df.columns]
    df = df.drop(columns=drop_cols).rename(columns={"Churn":"Exited"})
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
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42
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

# Only train if logged in (avoid slowing down the login page)
ready = False
model = scaler = le = feat_names = metrics = cm = None
if st.session_state["logged_in"]:
    with st.spinner("🔄 Loading model — first run takes ~20 seconds…"):
        try:
            model, scaler, le, feat_names, metrics, cm = train_model()
            ready = True
        except Exception as e:
            st.error(f"Model training failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
def run_predict(customer):
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


# ══════════════════════════════════════════════════════════════════════════════
# EXPLANATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def build_explanation(c):
    factors = []
    if c["IsActiveMember"] == 0:
        factors.append(("risk","🔴 Inactive Member — This customer is NOT actively using bank services. Inactive customers are 2–3× more likely to churn."))
    else:
        factors.append(("safe","🟢 Active Member — This customer actively uses bank services. Active engagement is the strongest retention signal."))
    if c["Age"] >= 50:
        factors.append(("risk",f"🔴 Age ({c['Age']} yrs) — Customers aged 50+ show significantly higher churn rates."))
    elif c["Age"] <= 30:
        factors.append(("warn",f"🟡 Age ({c['Age']} yrs) — Younger customers tend to be less loyal and more open to switching."))
    else:
        factors.append(("safe",f"🟢 Age ({c['Age']} yrs) — Middle-aged customers show average retention rates."))
    if c["NumOfProducts"] == 1:
        factors.append(("risk","🔴 Single Product — Only 1 bank product means low switching costs. Customers with 2+ products are less likely to leave."))
    elif c["NumOfProducts"] == 2:
        factors.append(("safe","🟢 Two Products — Holding 2 products is strongly associated with retention."))
    else:
        factors.append(("warn",f"🟡 {c['NumOfProducts']} Products — 3+ products sometimes indicates over-commitment and hidden churn risk."))
    if c["Geography"] == "Germany":
        factors.append(("risk","🔴 Geography: Germany — German customers have the highest churn rate (~32%)."))
    elif c["Geography"] == "Spain":
        factors.append(("warn","🟡 Geography: Spain — Spanish customers show moderate churn rates (~17%)."))
    else:
        factors.append(("safe","🟢 Geography: France — French customers show the lowest churn rate (~16%)."))
    if c["CreditScore"] < 500:
        factors.append(("risk",f"🔴 Credit Score ({c['CreditScore']}) — A low score suggests financial stress, linked to higher churn."))
    elif c["CreditScore"] >= 700:
        factors.append(("safe",f"🟢 Credit Score ({c['CreditScore']}) — A strong credit score indicates financial stability."))
    else:
        factors.append(("warn",f"🟡 Credit Score ({c['CreditScore']}) — Average score carries moderate churn risk."))
    if c["Tenure"] <= 1:
        factors.append(("risk",f"🔴 Short Tenure ({c['Tenure']} yr) — New customers are at highest churn risk."))
    elif c["Tenure"] >= 7:
        factors.append(("safe",f"🟢 Long Tenure ({c['Tenure']} yrs) — Long-standing customers are far less likely to leave."))
    else:
        factors.append(("warn",f"🟡 Tenure ({c['Tenure']} yrs) — Mid-range tenure carries neither strong loyalty nor high exit risk."))
    if c["Balance"] == 0:
        factors.append(("warn","🟡 Zero Balance — A $0 balance often signals a dormant account at moderate churn risk."))
    elif c["Balance"] > 150000:
        factors.append(("warn",f"🟡 High Balance (${c['Balance']:,.0f}) — Very high balances can correlate with churn if service falls short."))
    else:
        factors.append(("safe",f"🟢 Balance (${c['Balance']:,.0f}) — A moderate balance suggests normal banking activity."))
    if c["Gender"] == "Female":
        factors.append(("warn","🟡 Gender: Female — Female customers churn at a slightly higher rate (~25%) in this dataset."))
    else:
        factors.append(("safe","🟢 Gender: Male — Male customers show a slightly lower churn rate (~16%)."))
    if c["HasCrCard"] == 0:
        factors.append(("warn","🟡 No Credit Card — Slightly increases the likelihood of switching to a competitor."))
    else:
        factors.append(("safe","🟢 Has Credit Card — A bank-issued credit card creates an additional product tie."))
    order = {"risk":0,"warn":1,"safe":2}
    factors.sort(key=lambda x: order[x[0]])
    return factors


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def prob_bar_fig(prob, will_churn):
    color = "#E85555" if will_churn else "#3DBE8A"
    fig, ax = plt.subplots(figsize=(5, 1.1), facecolor="#0f2548")
    ax.set_facecolor("#0f2548")
    ax.barh([0], [1],    color="#1e3a5a", height=0.55, edgecolor="none")
    ax.barh([0], [prob], color=color,    height=0.55, edgecolor="none")
    ax.axvline(0.5, color="#FFE08A", lw=2, linestyle="--", alpha=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%","25%","50%","75%","100%"], fontsize=9, color="#A8B8D0")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0)
    ax.text(min(prob+0.02, 0.82), 0, f"{prob*100:.1f}%",
            va="center", color=color, fontsize=12, fontweight="bold")
    plt.tight_layout(pad=0.2)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████  PAGE — LOGIN  ████████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["page"] == "login":
    set_page_bg("login")
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div style="text-align:center;margin-bottom:2rem;margin-top:1rem;">
            <div style="font-size:3.5rem;">🏦</div>
            <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:#FFE08A;margin-top:0.3rem;">
                Churn Predictor
            </div>
            <div style="color:#C0CFDF;font-size:0.95rem;margin-top:0.3rem;">
                Bank Customer Intelligence Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="auth-title" style="margin-top:1rem;">Sign In</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Enter your credentials to access the platform</div>', unsafe_allow_html=True)

        if st.session_state.get("reg_success"):
            st.markdown('<div class="alert-success">✅ Account created successfully! Please sign in.</div>', unsafe_allow_html=True)
            st.session_state["reg_success"] = False

        if st.session_state.get("auth_error"):
            st.markdown(f'<div class="alert-error">⚠️ {st.session_state["auth_error"]}</div>', unsafe_allow_html=True)
            st.session_state["auth_error"] = ""

        # Use return values directly — never rely on session_state keys for
        # password inputs because Streamlit does not update them reliably on rerun.
        saved_user = st.session_state.get("saved_username", "")
        saved_pw   = st.session_state.get("saved_password", "")
        saved_rem  = st.session_state.get("remember_me", False)

        username = st.text_input("Username", value=saved_user, placeholder="Enter username")
        password = st.text_input("Password", value=saved_pw,   type="password", placeholder="Enter password")
        remember = st.checkbox("🔐 Remember my password", value=saved_rem)

        st.markdown('<div class="primary-btn" style="margin-top:1.2rem;">', unsafe_allow_html=True)
        if st.button("Sign In →", use_container_width=True, key="do_login"):
            users = st.session_state["users"]
            entered_user = username.strip()
            entered_pw   = password
            if entered_user in users and users[entered_user]["password_hash"] == _hash(entered_pw):
                if remember:
                    st.session_state["saved_username"] = entered_user
                    st.session_state["saved_password"] = entered_pw
                    st.session_state["remember_me"]    = True
                else:
                    st.session_state["saved_username"] = ""
                    st.session_state["saved_password"] = ""
                    st.session_state["remember_me"]    = False
                st.session_state["logged_in"]    = True
                st.session_state["current_user"] = entered_user
                st.session_state["current_role"] = users[entered_user]["role"]
                st.session_state["page"]         = "admin" if users[entered_user]["role"] == "admin" else "input"
                st.rerun()
            else:
                st.session_state["auth_error"] = "Invalid username or password."
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<hr style='margin:1.5rem 0;border-color:#3A5A8A;'>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#C0CFDF;font-size:0.9rem;">Don\'t have an account?</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-btn" style="margin-top:0.6rem;">', unsafe_allow_html=True)
        if st.button("Create Account", use_container_width=True, key="go_register"):
            st.session_state["page"] = "register"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████  PAGE — REGISTER  ████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "register":
    set_page_bg("register")
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("""
        <div style="text-align:center;margin-bottom:2rem;margin-top:1rem;">
            <div style="font-size:3.5rem;">🏦</div>
            <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:#FFE08A;margin-top:0.3rem;">
                Create Account
            </div>
            <div style="color:#C0CFDF;font-size:0.95rem;margin-top:0.3rem;">
                Join the Churn Predictor Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="auth-title" style="margin-top:1rem;">Register</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Fill in your details to create a new account</div>', unsafe_allow_html=True)

        if st.session_state.get("auth_error"):
            st.markdown(f'<div class="alert-error">⚠️ {st.session_state["auth_error"]}</div>', unsafe_allow_html=True)
            st.session_state["auth_error"] = ""

        full_name = st.text_input("Full Name", placeholder="e.g. Jane Doe", key="reg_name")
        reg_user  = st.text_input("Username", placeholder="Choose a username", key="reg_user")
        reg_pw    = st.text_input("Password", type="password", placeholder="Min 6 characters", key="reg_pw")
        reg_pw2   = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="reg_pw2")

        st.markdown('<div class="primary-btn" style="margin-top:1.2rem;">', unsafe_allow_html=True)
        if st.button("Create Account →", use_container_width=True, key="do_register"):
            users = st.session_state["users"]
            if not full_name or not reg_user or not reg_pw:
                st.session_state["auth_error"] = "All fields are required."
                st.rerun()
            elif reg_user in users:
                st.session_state["auth_error"] = f"Username '{reg_user}' is already taken."
                st.rerun()
            elif len(reg_pw) < 6:
                st.session_state["auth_error"] = "Password must be at least 6 characters."
                st.rerun()
            elif reg_pw != reg_pw2:
                st.session_state["auth_error"] = "Passwords do not match."
                st.rerun()
            else:
                st.session_state["users"][reg_user] = {
                    "password_hash": _hash(reg_pw),
                    "role": "user",
                    "name": full_name,
                }
                st.session_state["reg_success"] = True
                st.session_state["page"] = "login"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<hr style='margin:1.5rem 0;border-color:#3A5A8A;'>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:#C0CFDF;font-size:0.9rem;">Already have an account?</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-btn" style="margin-top:0.6rem;">', unsafe_allow_html=True)
        if st.button("← Back to Sign In", use_container_width=True, key="go_login"):
            st.session_state["auth_error"] = ""
            st.session_state["page"] = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████  PAGE — ADMIN DASHBOARD  ██████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "admin":
    import datetime

    if not st.session_state["logged_in"] or st.session_state["current_role"] != "admin":
        st.session_state["page"] = "login"
        st.rerun()

    set_page_bg("admin")
    render_nav()

    st.markdown("""
    <div class="page-header">
      <h1>⚙️ Admin Dashboard</h1>
      <p>Manage users and monitor platform activity</p>
    </div>
    """, unsafe_allow_html=True)

    users = st.session_state["users"]

    # ── Stats row ──────────────────────────────────────────────────────────
    total    = len(users)
    admins   = sum(1 for u in users.values() if u["role"] == "admin")
    analysts = total - admins

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="mbox"><div class="lbl">Total Users</div><div class="val">{total}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="mbox"><div class="lbl">Admins</div><div class="val" style="color:#FFE08A;">{admins}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="mbox"><div class="lbl">Analysts</div><div class="val" style="color:#2ECC8A;">{analysts}</div></div>', unsafe_allow_html=True)

    # ── User list ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">👥 User Accounts</div>', unsafe_allow_html=True)

    rows = ""
    for uname, udata in users.items():
        badge = f'<span class="badge-admin">Admin</span>' if udata["role"] == "admin" else f'<span class="badge-user">Analyst</span>'
        rows += f"""
        <tr>
            <td>{udata['name']}</td>
            <td><code style="color:#C9A84C;background:#1a3560;padding:0.1rem 0.5rem;border-radius:5px;">{uname}</code></td>
            <td>{badge}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:#112244;border:1.5px solid #3A5A8A;border-radius:14px;overflow:hidden;margin-bottom:1.5rem;">
        <table class="admin-table">
            <thead><tr><th>Name</th><th>Username</th><th>Role</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Add new user ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">➕ Add New User</div>', unsafe_allow_html=True)

    if st.session_state.get("auth_error"):
        st.markdown(f'<div class="alert-error">⚠️ {st.session_state["auth_error"]}</div>', unsafe_allow_html=True)
        st.session_state["auth_error"] = ""
    if st.session_state.get("admin_msg"):
        st.markdown(f'<div class="alert-success">✅ {st.session_state["admin_msg"]}</div>', unsafe_allow_html=True)
        st.session_state["admin_msg"] = ""

    with st.form("add_user_form", clear_on_submit=True):
        ac1, ac2 = st.columns(2)
        with ac1:
            new_name = st.text_input("Full Name", placeholder="e.g. Mary Wanjiku")
            new_user = st.text_input("Username", placeholder="e.g. mwanjiku")
        with ac2:
            new_pw   = st.text_input("Password", type="password", placeholder="Min 6 chars")
            new_role = st.selectbox("Role", ["user", "admin"])
        submitted = st.form_submit_button("➕ Add User", use_container_width=False)
        if submitted:
            if not new_name or not new_user or not new_pw:
                st.session_state["auth_error"] = "All fields required."
                st.rerun()
            elif new_user in users:
                st.session_state["auth_error"] = f"Username '{new_user}' already exists."
                st.rerun()
            elif len(new_pw) < 6:
                st.session_state["auth_error"] = "Password must be at least 6 characters."
                st.rerun()
            else:
                st.session_state["users"][new_user] = {
                    "password_hash": _hash(new_pw),
                    "role": new_role,
                    "name": new_name,
                }
                st.session_state["admin_msg"] = f"User '{new_user}' added successfully."
                st.rerun()

    # ── Delete user ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🗑️ Remove User</div>', unsafe_allow_html=True)

    deletable = [u for u in users if u != st.session_state["current_user"]]
    if deletable:
        del_col1, del_col2 = st.columns([2, 1])
        with del_col1:
            del_target = st.selectbox("Select user to remove", deletable, key="del_user_select")
        with del_col2:
            st.markdown('<div style="margin-top:1.8rem;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button(f"🗑️ Remove", use_container_width=True, key="do_delete"):
                del st.session_state["users"][del_target]
                st.session_state["admin_msg"] = f"User '{del_target}' removed."
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#C0CFDF;">No other users to remove.</p>', unsafe_allow_html=True)

    # ── Model metrics (if loaded) ──────────────────────────────────────────
    if ready:
        st.markdown('<div class="section-title">📊 Model Performance Metrics</div>', unsafe_allow_html=True)
        mcols = st.columns(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            with mcols[i]:
                st.markdown(f'<div class="mbox"><div class="lbl">{k}</div><div class="val">{v:.1%}</div></div>', unsafe_allow_html=True)

    # ── Prediction Report ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">📥 Customer Prediction Report</div>', unsafe_allow_html=True)

    log = st.session_state.get("prediction_log", [])
    if not log:
        st.markdown('<p style="color:#C0CFDF;">No predictions have been made yet. Reports will appear here once users run predictions.</p>', unsafe_allow_html=True)
    else:
        log_df = pd.DataFrame(log)

        # Summary stats row
        total_preds  = len(log_df)
        churn_preds  = (log_df["Verdict"] == "CHURN").sum()
        retain_preds = total_preds - churn_preds
        churn_rate   = churn_preds / total_preds * 100

        rp1, rp2, rp3, rp4 = st.columns(4)
        with rp1: st.markdown(f'<div class="mbox"><div class="lbl">Total Predictions</div><div class="val">{total_preds}</div></div>', unsafe_allow_html=True)
        with rp2: st.markdown(f'<div class="mbox"><div class="lbl">Predicted Churn</div><div class="val" style="color:#FF5555;">{churn_preds}</div></div>', unsafe_allow_html=True)
        with rp3: st.markdown(f'<div class="mbox"><div class="lbl">Predicted Retain</div><div class="val" style="color:#2ECC8A;">{retain_preds}</div></div>', unsafe_allow_html=True)
        with rp4: st.markdown(f'<div class="mbox"><div class="lbl">Churn Rate</div><div class="val" style="color:#FF9933;">{churn_rate:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Show table
        display_cols = ["timestamp", "user", "Geography", "Gender", "Age", "CreditScore",
                        "Tenure", "Balance", "NumOfProducts", "EstimatedSalary",
                        "Churn Probability", "Verdict"]
        available_cols = [c for c in display_cols if c in log_df.columns]
        preview_df = log_df[available_cols].copy()
        preview_df.columns = [c.replace("_", " ").title() for c in available_cols]

        # Colour-code Verdict column
        def style_verdict(val):
            color = "#FF5555" if val == "CHURN" else "#2ECC8A"
            return f"color: {color}; font-weight: bold;"

        rows_html = ""
        for _, row in log_df[available_cols].iterrows():
            verdict_color = "#FF5555" if row.get("Verdict") == "CHURN" else "#2ECC8A"
            cells = ""
            for col in available_cols:
                val = row[col]
                if col == "Verdict":
                    cells += f'<td style="color:{verdict_color};font-weight:700;">{val}</td>'
                elif col == "Balance":
                    cells += f'<td>${float(val):,.0f}</td>'
                elif col == "Churn Probability":
                    cells += f'<td>{float(val):.1f}%</td>'
                else:
                    cells += f'<td>{val}</td>'
            rows_html += f"<tr>{cells}</tr>"

        header_html = "".join(f"<th>{c.replace('_',' ').title()}</th>" for c in available_cols)
        st.markdown(f"""
        <div style="background:#112244;border:1.5px solid #3A5A8A;border-radius:14px;overflow:auto;margin-bottom:1.5rem;max-height:400px;">
            <table class="admin-table">
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── PDF Report Generator ──────────────────────────────────────────────
        def generate_pdf_report(log_records):
            import io, datetime
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, Image as RLImage, PageBreak, KeepTogether,
            )
            from reportlab.graphics.shapes import Drawing, Rect, String
            from reportlab.lib.colors import HexColor

            # ── Colour palette ────────────────────────────────────────────────
            NAVY     = HexColor("#0B1D3A")
            CARD     = HexColor("#112244")
            GOLD     = HexColor("#C9A84C")
            GOLD2    = HexColor("#FFE08A")
            WHITE    = HexColor("#FFFFFF")
            MUTED    = HexColor("#C0CFDF")
            RED      = HexColor("#E85555")
            GREEN    = HexColor("#2ECC8A")
            AMBER    = HexColor("#FF9933")
            BORDER   = HexColor("#3A5A8A")

            buf = io.BytesIO()
            doc = SimpleDocTemplate(
                buf,
                pagesize=A4,
                leftMargin=2*cm, rightMargin=2*cm,
                topMargin=2.2*cm, bottomMargin=2.2*cm,
            )
            W, H = A4
            content_width = W - 4*cm

            # ── Styles ────────────────────────────────────────────────────────
            base = getSampleStyleSheet()

            def ps(name, **kw):
                defaults = dict(fontName="Helvetica", fontSize=11,
                                textColor=WHITE, leading=16)
                defaults.update(kw)
                return ParagraphStyle(name, **defaults)

            sTitle    = ps("sTitle",    fontName="Helvetica-Bold", fontSize=28,
                           textColor=GOLD2, alignment=TA_CENTER, leading=36, spaceAfter=4)
            sSubtitle = ps("sSubtitle", fontSize=13, textColor=MUTED,
                           alignment=TA_CENTER, leading=18, spaceAfter=2)
            sSection  = ps("sSection",  fontName="Helvetica-Bold", fontSize=13,
                           textColor=GOLD, leading=18, spaceBefore=10, spaceAfter=4)
            sField    = ps("sField",    fontSize=10, textColor=MUTED,  leading=14)
            sValue    = ps("sValue",    fontName="Helvetica-Bold", fontSize=10,
                           textColor=WHITE, leading=14)
            sVerdict  = ps("sVerdict",  fontName="Helvetica-Bold", fontSize=16,
                           alignment=TA_CENTER, leading=22)
            sFooter   = ps("sFooter",   fontSize=8,  textColor=MUTED,
                           alignment=TA_CENTER, leading=11)
            sCustHead = ps("sCustHead", fontName="Helvetica-Bold", fontSize=14,
                           textColor=GOLD2, leading=20, spaceBefore=6, spaceAfter=2)
            sBody     = ps("sBody",     fontSize=10, textColor=WHITE, leading=15)

            # ── Page background / header callback ─────────────────────────────
            def on_page(canvas_obj, doc_obj):
                canvas_obj.saveState()
                canvas_obj.setFillColor(NAVY)
                canvas_obj.rect(0, 0, W, H, fill=1, stroke=0)
                # Gold top bar
                canvas_obj.setFillColor(GOLD)
                canvas_obj.rect(0, H - 1.1*cm, W, 1.1*cm, fill=1, stroke=0)
                canvas_obj.setFillColor(NAVY)
                canvas_obj.setFont("Helvetica-Bold", 11)
                canvas_obj.drawCentredString(W/2, H - 0.72*cm,
                    "CUSTOMER CHURN PREDICTION REPORT  |  CONFIDENTIAL")
                # Gold bottom bar
                canvas_obj.setFillColor(CARD)
                canvas_obj.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
                canvas_obj.setFillColor(MUTED)
                canvas_obj.setFont("Helvetica", 8)
                canvas_obj.drawString(2*cm, 0.35*cm,
                    f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                canvas_obj.drawRightString(W - 2*cm, 0.35*cm,
                    f"Page {doc_obj.page}")
                canvas_obj.restoreState()

            # ── Helper: churn probability bar chart (matplotlib → BytesIO) ────
            def make_prob_chart(prob, will_churn, width_px=480, height_px=110):
                fig, ax = plt.subplots(figsize=(width_px/96, height_px/96), dpi=96)
                fig.patch.set_facecolor("#112244")
                ax.set_facecolor("#112244")

                bar_color = "#E85555" if will_churn else "#2ECC8A"
                bg_color  = "#1a3560"

                ax.barh(0, 1,   height=0.55, color=bg_color,  left=0)
                ax.barh(0, prob, height=0.55, color=bar_color, left=0)
                ax.axvline(0.5, color="#FFE08A", linewidth=1.8, linestyle="--", alpha=0.9)

                ax.set_xlim(0, 1)
                ax.set_ylim(-0.55, 0.55)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                                   color="#C0CFDF", fontsize=8)
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(axis="x", colors="#C0CFDF", length=0)

                label = f"{prob*100:.1f}%"
                ax.text(min(prob + 0.02, 0.97), 0, label,
                        va="center", ha="left" if prob < 0.85 else "right",
                        color=bar_color, fontsize=10, fontweight="bold")
                ax.text(0.505, 0.42, "50% threshold",
                        va="bottom", ha="left",
                        color="#FFE08A", fontsize=7, transform=ax.get_xaxis_transform())

                plt.tight_layout(pad=0.3)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", dpi=96,
                            facecolor=fig.get_facecolor())
                plt.close(fig)
                img_buf.seek(0)
                return img_buf

            # ── Story assembly ────────────────────────────────────────────────
            story = []
            now_str = datetime.datetime.now().strftime("%B %d, %Y  |  %H:%M:%S")

            # Cover block
            story.append(Spacer(1, 1.2*cm))
            story.append(Paragraph("🏦 Customer Churn Prediction", sTitle))
            story.append(Paragraph("Prediction Analysis Report", sSubtitle))
            story.append(Spacer(1, 0.4*cm))
            story.append(HRFlowable(width="100%", thickness=2,
                                    color=GOLD, spaceAfter=10))

            # Report metadata table
            admin_user  = st.session_state.get("current_user", "N/A")
            admin_name  = st.session_state["users"].get(admin_user, {}).get("name", "N/A")
            meta_data = [
                ["Report Generated By:", admin_name],
                ["Username:", admin_user],
                ["Report Date:", now_str],
                ["Total Predictions:", str(len(log_records))],
                ["Churn Predicted:",   str(sum(1 for r in log_records if r.get("Verdict") == "CHURN"))],
                ["Retain Predicted:",  str(sum(1 for r in log_records if r.get("Verdict") == "RETAIN"))],
            ]
            meta_tbl = Table(meta_data, colWidths=[5*cm, content_width - 5*cm])
            meta_tbl.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (0,-1), CARD),
                ("BACKGROUND",  (1,0), (1,-1), NAVY),
                ("TEXTCOLOR",   (0,0), (0,-1), MUTED),
                ("TEXTCOLOR",   (1,0), (1,-1), WHITE),
                ("FONTNAME",    (0,0), (0,-1), "Helvetica"),
                ("FONTNAME",    (1,0), (1,-1), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 10),
                ("TOPPADDING",  (0,0), (-1,-1), 7),
                ("BOTTOMPADDING",(0,0),(-1,-1), 7),
                ("LEFTPADDING", (0,0), (-1,-1), 10),
                ("GRID",        (0,0), (-1,-1), 0.5, BORDER),
                ("ROWBACKGROUNDS",(0,0),(-1,-1),[CARD, HexColor("#0d1e3d")]),
            ]))
            story.append(meta_tbl)
            story.append(Spacer(1, 0.6*cm))
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=BORDER, spaceAfter=6))

            # ── Per-customer sections ─────────────────────────────────────────
            for idx, rec in enumerate(log_records, 1):
                prob      = float(rec.get("Churn Probability", 0)) / 100.0
                verdict   = rec.get("Verdict", "UNKNOWN")
                will_churn = verdict == "CHURN"
                v_color    = RED if will_churn else GREEN
                risk_label = "HIGH RISK" if prob >= 0.6 else ("MEDIUM RISK" if prob >= 0.35 else "LOW RISK")
                risk_color = RED if prob >= 0.6 else (AMBER if prob >= 0.35 else GREEN)

                block = []

                # Customer header
                block.append(Spacer(1, 0.3*cm))
                block.append(Paragraph(
                    f"Customer #{idx}  —  Predicted by: {rec.get('user','N/A')}  |  {rec.get('timestamp','N/A')}",
                    sCustHead))
                block.append(HRFlowable(width="100%", thickness=1,
                                        color=GOLD, spaceAfter=6))

                # Verdict banner
                v_text = "✖  WILL CHURN" if will_churn else "✔  WILL NOT CHURN"
                v_bg   = HexColor("#2a0808") if will_churn else HexColor("#062d16")
                v_para = Paragraph(
                    f'<font color="{"#E85555" if will_churn else "#2ECC8A"}">'
                    f'<b>{v_text}</b></font>', sVerdict)
                verdict_tbl = Table([[v_para]], colWidths=[content_width])
                verdict_tbl.setStyle(TableStyle([
                    ("BACKGROUND",   (0,0), (-1,-1), v_bg),
                    ("TOPPADDING",   (0,0), (-1,-1), 10),
                    ("BOTTOMPADDING",(0,0), (-1,-1), 10),
                    ("LEFTPADDING",  (0,0), (-1,-1), 10),
                    ("BOX",          (0,0), (-1,-1), 1.5,
                     RED if will_churn else GREEN),
                    ("ROWBACKGROUNDS",(0,0),(-1,-1),[v_bg]),
                ]))
                block.append(verdict_tbl)
                block.append(Spacer(1, 0.25*cm))

                # Probability bar chart
                chart_buf = make_prob_chart(prob, will_churn)
                chart_img = RLImage(chart_buf,
                                    width=content_width,
                                    height=content_width * (110/480))
                block.append(chart_img)
                block.append(Spacer(1, 0.15*cm))

                # Risk + probability summary row
                summary_data = [
                    [
                        Paragraph('<font color="#C0CFDF">Churn Probability</font>', sField),
                        Paragraph('<font color="#C0CFDF">Verdict</font>', sField),
                        Paragraph('<font color="#C0CFDF">Risk Level</font>', sField),
                    ],
                    [
                        Paragraph(f'<font color="{"#E85555" if will_churn else "#2ECC8A"}">'
                                  f'<b>{prob*100:.1f}%</b></font>', sValue),
                        Paragraph(f'<font color="{"#E85555" if will_churn else "#2ECC8A"}">'
                                  f'<b>{verdict}</b></font>', sValue),
                        Paragraph(f'<font color="{"#E85555" if prob>=0.6 else ("#FF9933" if prob>=0.35 else "#2ECC8A")}">'
                                  f'<b>{risk_label}</b></font>', sValue),
                    ],
                ]
                col3 = content_width / 3
                sum_tbl = Table(summary_data, colWidths=[col3]*3)
                sum_tbl.setStyle(TableStyle([
                    ("BACKGROUND",   (0,0), (-1,-1), CARD),
                    ("TOPPADDING",   (0,0), (-1,-1), 8),
                    ("BOTTOMPADDING",(0,0), (-1,-1), 8),
                    ("LEFTPADDING",  (0,0), (-1,-1), 14),
                    ("ALIGN",        (0,0), (-1,-1), "LEFT"),
                    ("GRID",         (0,0), (-1,-1), 0.5, BORDER),
                ]))
                block.append(sum_tbl)
                block.append(Spacer(1, 0.25*cm))

                # Customer profile table (2 columns side by side)
                block.append(Paragraph("Customer Profile", sSection))
                profile_left = [
                    ["Geography",      str(rec.get("Geography","—"))],
                    ["Gender",         str(rec.get("Gender","—"))],
                    ["Age",            f"{rec.get('Age','—')} years"],
                    ["Credit Score",   str(rec.get("CreditScore","—"))],
                    ["Tenure",         f"{rec.get('Tenure','—')} years"],
                ]
                profile_right = [
                    ["Account Balance",   f"${float(rec.get('Balance',0)):,.0f}"],
                    ["No. of Products",   str(rec.get("NumOfProducts","—"))],
                    ["Estimated Salary",  f"${float(rec.get('EstimatedSalary',0)):,.0f}"],
                    ["Has Credit Card",   "Yes" if rec.get("HasCrCard") == 1 else "No"],
                    ["Active Member",     "Yes" if rec.get("IsActiveMember") == 1 else "No"],
                ]
                combined = [l + r for l, r in zip(profile_left, profile_right)]
                half = content_width / 2
                q    = half / 2
                prof_tbl = Table(combined, colWidths=[q, q, q, q])
                prof_tbl.setStyle(TableStyle([
                    ("BACKGROUND",  (0,0), (0,-1), CARD),
                    ("BACKGROUND",  (2,0), (2,-1), CARD),
                    ("BACKGROUND",  (1,0), (1,-1), HexColor("#0d1e3d")),
                    ("BACKGROUND",  (3,0), (3,-1), HexColor("#0d1e3d")),
                    ("TEXTCOLOR",   (0,0), (0,-1), MUTED),
                    ("TEXTCOLOR",   (2,0), (2,-1), MUTED),
                    ("TEXTCOLOR",   (1,0), (1,-1), WHITE),
                    ("TEXTCOLOR",   (3,0), (3,-1), WHITE),
                    ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
                    ("FONTNAME",    (1,0), (1,-1), "Helvetica-Bold"),
                    ("FONTNAME",    (3,0), (3,-1), "Helvetica-Bold"),
                    ("FONTSIZE",    (0,0), (-1,-1), 9),
                    ("TOPPADDING",  (0,0), (-1,-1), 6),
                    ("BOTTOMPADDING",(0,0),(-1,-1), 6),
                    ("LEFTPADDING", (0,0), (-1,-1), 8),
                    ("GRID",        (0,0), (-1,-1), 0.5, BORDER),
                ]))
                block.append(prof_tbl)
                block.append(Spacer(1, 0.2*cm))
                block.append(HRFlowable(width="100%", thickness=0.5,
                                        color=BORDER, spaceAfter=4))

                story.append(KeepTogether(block[:6]))  # header + verdict + chart
                for item in block[6:]:
                    story.append(item)

                if idx < len(log_records):
                    story.append(PageBreak())

            # ── Footer note ───────────────────────────────────────────────────
            story.append(Spacer(1, 0.8*cm))
            story.append(HRFlowable(width="100%", thickness=1,
                                    color=GOLD, spaceAfter=8))
            story.append(Paragraph(
                "This report is generated by the Customer Churn Prediction System. "
                "All predictions are model-based estimates and should be reviewed by "
                "an authorised bank analyst before acting on them.",
                sFooter))

            doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
            buf.seek(0)
            return buf.read()

        pdf_bytes = generate_pdf_report(log)
        st.download_button(
            label="⬇️ Download Full PDF Report",
            data=pdf_bytes,
            file_name=f"churn_prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="download_report_pdf",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████  PAGE — INPUT FORM  ███████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "input":

    if not st.session_state["logged_in"]:
        st.session_state["page"] = "login"
        st.rerun()

    set_page_bg("input")
    render_nav()

    st.markdown("""
    <div class="page-header">
      <h1>Customer Churn</h1>
      <p>Fill in all fields in the customer profile below, then run the churn prediction model</p>
    </div>
    """, unsafe_allow_html=True)

    if not ready:
        st.stop()

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        geography = st.selectbox("Geography", ["— select —", "France", "Germany", "Spain"])
    with c2:
        gender = st.selectbox("Gender", ["— select —", "Female", "Male"])
    with c3:
        age = st.number_input(
            "Age (18 - 92)",
            min_value=0, max_value=92,
            value=None, step=1,
            placeholder="e.g. 42",
        )
    st.markdown("<hr style='border-color:#1e3a5a;margin:1.2rem 0;'>", unsafe_allow_html=True)

    # ── Account Details ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏦 Account Details</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        credit_score = st.number_input(
            "Credit Score (300 - 850)",
            min_value=0, max_value=850,
            value=None, step=1,
            placeholder="e.g. 650",
        )
    with c5:
        tenure = st.number_input(
            "Tenure (years, 0 - 10)",
            min_value=0, max_value=10,
            value=None, step=1,
            placeholder="e.g. 3",
        )
    with c6:
        num_products = st.selectbox("Number of Products", ["— select —", 1, 2, 3, 4])
    c7, c8 = st.columns(2)
    with c7:
        balance = st.number_input(
            "Account Balance ($)",
            min_value=0.0, max_value=300_000.0,
            value=None, step=500.0,
            placeholder="e.g. 125000.00",
        )
    with c8:
        estimated_sal = st.number_input(
            "Estimated Salary ($)",
            min_value=0.0, max_value=250_000.0,
            value=None, step=500.0,
            placeholder="e.g. 80000.00",
        )
    st.markdown("<hr style='border-color:#1e3a5a;margin:1.2rem 0;'>", unsafe_allow_html=True)

    # ── Engagement ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📲 Engagement</div>', unsafe_allow_html=True)
    c9, c10 = st.columns(2)
    with c9:
        has_cr_card = st.radio("Has Credit Card?", ["— select —", "Yes", "No"], horizontal=True)
    with c10:
        is_active = st.radio("Is Active Member?", ["— select —", "Yes", "No"], horizontal=True)
    st.markdown("<hr style='border-color:#1e3a5a;margin:1.2rem 0;'>", unsafe_allow_html=True)

    # ── Validation ────────────────────────────────────────────────────────────
    missing = []
    if geography == "— select —":    missing.append("Geography")
    if gender    == "— select —":    missing.append("Gender")
    if age       is None:            missing.append("Age")
    if credit_score is None:         missing.append("Credit Score")
    if tenure    is None:            missing.append("Tenure")
    if num_products == "— select —": missing.append("Number of Products")
    if balance   is None:            missing.append("Account Balance")
    if estimated_sal is None:        missing.append("Estimated Salary")
    if has_cr_card == "— select —":  missing.append("Has Credit Card")
    if is_active  == "— select —":   missing.append("Is Active Member")
    if credit_score is not None and credit_score < 300:
        missing.append("Credit Score must be at least 300")

    if missing:
        st.markdown(
            f'<div class="alert-error">⚠️ Please complete the following fields before predicting: '
            f'<strong>{", ".join(missing)}</strong></div>',
            unsafe_allow_html=True,
        )

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict_clicked = st.button(
        "🔍  Predict Customer Churn",
        use_container_width=True,
        disabled=bool(missing),
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked and not missing:
        import datetime
        customer_data = {
            "CreditScore":     int(credit_score),
            "Geography":       geography,
            "Gender":          gender,
            "Age":             int(age),
            "Tenure":          int(tenure),
            "Balance":         float(balance),
            "NumOfProducts":   int(num_products),
            "HasCrCard":       1 if has_cr_card == "Yes" else 0,
            "IsActiveMember":  1 if is_active   == "Yes" else 0,
            "EstimatedSalary": float(estimated_sal),
        }
        st.session_state["customer"] = customer_data
        if ready:
            _prob, _churn = run_predict(customer_data)
            st.session_state["prediction_log"].append({
                "user":            st.session_state["current_user"],
                "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Geography":       geography,
                "Gender":          gender,
                "Age":             int(age),
                "CreditScore":     int(credit_score),
                "Tenure":          int(tenure),
                "Balance":         float(balance),
                "NumOfProducts":   int(num_products),
                "HasCrCard":       1 if has_cr_card == "Yes" else 0,
                "IsActiveMember":  1 if is_active   == "Yes" else 0,
                "EstimatedSalary": float(estimated_sal),
                "Churn Probability": round(_prob * 100, 2),
                "Verdict":         "CHURN" if _churn else "RETAIN",
            })
        st.session_state["page"] = "results"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████  PAGE — RESULTS  ████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "results":

    if not st.session_state["logged_in"]:
        st.session_state["page"] = "login"
        st.rerun()

    set_page_bg("results")
    render_nav()

    if "customer" not in st.session_state or not ready:
        st.warning("No prediction data found. Please go back and fill in the form.")
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back to Input Form"):
            st.session_state["page"] = "input"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    customer = st.session_state["customer"]
    prob, will_churn = run_predict(customer)
    factors = build_explanation(customer)

    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Back to Input Form"):
        st.session_state["page"] = "input"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="page-header">
      <h1>Prediction Results</h1>
    </div>
    """, unsafe_allow_html=True)

    # ── VERDICT ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔎 Churn Verdict</div>', unsafe_allow_html=True)

    col_v, col_p = st.columns([1, 1.5], gap="large")

    with col_v:
        if will_churn:
            st.markdown(f"""
            <div class="verdict-yes">
              <div class="verdict-word" style="color:#E85555;">YES</div>
              <div class="verdict-desc" style="color:#E85555;">This customer WILL CHURN</div>
              <div class="verdict-prob">Model predicts a <strong style="color:#E85555;">{prob*100:.1f}%</strong> probability this customer will leave the bank.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-no">
              <div class="verdict-word" style="color:#3DBE8A;">NO</div>
              <div class="verdict-desc" style="color:#3DBE8A;">This customer will NOT CHURN</div>
              <div class="verdict-prob">Model predicts a <strong style="color:#3DBE8A;">{(1-prob)*100:.1f}%</strong> confidence this customer will stay.</div>
            </div>""", unsafe_allow_html=True)

    with col_p:
        verdict_col = "#E85555" if will_churn else "#3DBE8A"
        verdict_txt = "CHURN"   if will_churn else "RETAIN"
        risk_label  = "High"    if prob >= 0.6  else ("Medium" if prob >= 0.35 else "Low")
        risk_col    = "#E85555" if prob >= 0.6  else ("#F0922B" if prob >= 0.35 else "#3DBE8A")

        st.markdown('<div class="prob-card">', unsafe_allow_html=True)
        st.markdown('<div class="prob-card-title">📈 Churn Probability Score</div>', unsafe_allow_html=True)
        st.pyplot(prob_bar_fig(prob, will_churn), use_container_width=True)
        st.markdown('<p style="font-size:0.88rem;color:#C0CFDF;font-weight:500;margin:0.3rem 0 1rem 0;">Dashed line = 50% decision threshold — YES if above, NO if below</p>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f'<div class="mbox"><div class="lbl">Probability</div><div class="val" style="color:{verdict_col};">{prob*100:.1f}%</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="mbox"><div class="lbl">Verdict</div><div class="val" style="font-size:1.1rem;color:{verdict_col};">{verdict_txt}</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="mbox"><div class="lbl">Risk Level</div><div class="val" style="font-size:1.2rem;color:{risk_col};">{risk_label}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── EXPLANATION ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Why This Prediction?</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#C0CFDF;font-size:1rem;font-weight:500;margin-bottom:1.2rem;">Each factor below shows how this customer\'s profile influenced the prediction.</p>', unsafe_allow_html=True)

    css_map = {"risk":"exp-risk","warn":"exp-warn","safe":"exp-safe"}
    half = (len(factors) + 1) // 2
    e1, e2 = st.columns(2, gap="large")
    with e1:
        for ftype, ftext in factors[:half]:
            st.markdown(f'<div class="{css_map[ftype]}">{ftext}</div>', unsafe_allow_html=True)
    with e2:
        for ftype, ftext in factors[half:]:
            st.markdown(f'<div class="{css_map[ftype]}">{ftext}</div>', unsafe_allow_html=True)

    # ── RECOMMENDATION ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 Retention Recommendation</div>', unsafe_allow_html=True)
    rec_color = "#E85555" if will_churn else "#3DBE8A"
    rec_items = []
    if will_churn:
        if customer["IsActiveMember"] == 0:
            rec_items.append("📲 Launch an immediate re-engagement campaign — a personalised call or email from a relationship manager.")
        if customer["NumOfProducts"] == 1:
            rec_items.append("🎁 Offer a cross-sell incentive such as a credit card, savings plan, or mortgage consultation.")
        if customer["Geography"] == "Germany":
            rec_items.append("🌍 Review Germany-specific pricing and service offerings — competitive pressure is highest in this region.")
        if customer["Age"] >= 50:
            rec_items.append("👴 Consider a senior loyalty programme or preferential interest rates tailored to this age group.")
        if customer["Tenure"] <= 1:
            rec_items.append("🤝 Assign a dedicated onboarding advisor — early-tenure churn can be prevented with personalised support.")
        if customer["Balance"] == 0:
            rec_items.append("💳 Encourage account activation through zero-fee promotions or cashback offers.")
        if not rec_items:
            rec_items.append("⚠️ Multiple churn risk signals detected. A proactive outreach call from a senior relationship manager is strongly recommended.")
        rec_items.append("📊 Flag this customer for the retention team's priority watchlist immediately.")
    else:
        rec_items = [
            "✅ This customer is likely to stay — maintain current service quality and engagement.",
            "📧 Include in periodic satisfaction surveys to detect any early warning signs.",
            "💼 Consider upselling an additional product to further deepen the customer relationship.",
            "🎯 Recognise and reward loyalty — a thank-you message or benefit can reinforce positive sentiment.",
        ]

    st.markdown(f'<div class="rec-card" style="border-left:4px solid {rec_color};">', unsafe_allow_html=True)
    for item in rec_items:
        st.markdown(f'<div class="rec-item">{item}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CUSTOMER SUMMARY ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">👤 Customer Profile Summary</div>', unsafe_allow_html=True)
    c = customer
    t1, t2 = st.columns(2, gap="large")
    with t1:
        st.markdown(f"""
        <table class="summary-table">
          <tr><td>Geography</td><td>{c['Geography']}</td></tr>
          <tr><td>Gender</td><td>{c['Gender']}</td></tr>
          <tr><td>Age</td><td>{c['Age']} years</td></tr>
          <tr><td>Credit Score</td><td>{c['CreditScore']}</td></tr>
          <tr><td>Tenure</td><td>{c['Tenure']} years</td></tr>
        </table>""", unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
        <table class="summary-table">
          <tr><td>Account Balance</td><td>${c['Balance']:,.0f}</td></tr>
          <tr><td>Number of Products</td><td>{c['NumOfProducts']}</td></tr>
          <tr><td>Estimated Salary</td><td>${c['EstimatedSalary']:,.0f}</td></tr>
          <tr><td>Has Credit Card</td><td>{'Yes' if c['HasCrCard'] else 'No'}</td></tr>
          <tr><td>Active Member</td><td>{'Yes' if c['IsActiveMember'] else 'No'}</td></tr>
        </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="back-btn">', unsafe_allow_html=True)
    if st.button("← Predict Another Customer", use_container_width=False):
        st.session_state["page"] = "input"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
