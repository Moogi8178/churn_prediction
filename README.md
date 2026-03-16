# 🏦 Customer Churn Predictor — Streamlit App

**Meru University of Science and Technology · BSc Data Science**
**Student:** Finley Barongo Magembe | CT204/109437/22

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your trained model files here (from the Colab notebook):
#    - churn_dnn_model.h5
#    - churn_xgboost_model.pkl
#    - scaler.pkl

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, set main file to `app.py`
4. Upload model files via the **Secrets** or file upload feature
5. Click **Deploy** — your app gets a public URL!

---

## Demo Mode

If no model files are found, the app runs in **demo mode** using a rule-based
approximation so the UI can be previewed without trained models.

---

## File Structure

```
streamlit_app/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
├── churn_dnn_model.h5        ← (Add after training in Colab)
├── churn_xgboost_model.pkl   ← (Add after training in Colab)
└── scaler.pkl                ← (Add after training in Colab)
```

---

## App Features

| Tab | Contents |
|-----|----------|
| 🎯 Prediction | Gauge chart, risk badge, feature impact, recommendation |
| 📊 Analytics  | Model comparison table + AUC bar chart |
| ℹ️ About      | Project info and model loading instructions |
