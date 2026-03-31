import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection", page_icon="🔒", layout="wide"
)

MODEL_PATH = os.path.join("Models", "best_model_xgb.pkl")
SCALER_PATH = os.path.join("Models", "scaler.pkl")
TOP_FEATURES_PATH = os.path.join("Models", "top_features.pkl")


@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    top_features = joblib.load(TOP_FEATURES_PATH)
    return model, scaler, top_features


model, scaler, top_features = load_models()

RAW_FEATURES = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Time",
]

SCALED_FEATURES = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "scaled_Amount",
    "scaled_Time",
]

AMOUNT_MEAN, AMOUNT_STD = 88.35, 250.12
TIME_MEAN, TIME_STD = 94813.86, 47488.15

st.markdown(
    """
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #667eea; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #888; text-align: center; margin-bottom: 2rem; }
    .fraud-alert { background: linear-gradient(135deg, #ff4757, #ff6b81); color: white; padding: 2rem; border-radius: 12px; text-align: center; font-size: 1.5rem; }
    .legit-alert { background: linear-gradient(135deg, #2ed573, #7bed9f); color: white; padding: 2rem; border-radius: 12px; text-align: center; font-size: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">🔒 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">ML-Powered Real-Time Transaction Analysis</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Single Transaction", "Batch Prediction"])


def get_prediction(user_features):
    scaled_row = {
        **{
            k: user_features.get(k, 0.0)
            for k in RAW_FEATURES
            if k not in ["Amount", "Time"]
        },
        "scaled_Amount": (user_features.get("Amount", 0) - AMOUNT_MEAN) / AMOUNT_STD,
        "scaled_Time": (user_features.get("Time", 0) - TIME_MEAN) / TIME_STD,
    }
    scaler_input = [scaled_row[k] for k in SCALED_FEATURES]
    X_scaled = scaler.transform([scaler_input])
    model_input = {top_features[i]: X_scaled[0][i] for i in range(len(top_features))}
    X_model = np.array([[model_input[f] for f in top_features]])
    proba = float(model.predict_proba(X_model)[0][1])
    pred = int(proba >= 0.5)
    return pred, proba


with tab1:
    st.header("Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input(
            "Time (seconds from first transaction)", value=0.0, step=1.0
        )
        amount = st.number_input("Amount ($)", value=0.0, step=0.01)

    st.subheader("PCA Features (V1 - V28)")

    cols = st.columns(4)
    features = {}
    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            features[f"V{i}"] = st.number_input(
                f"V{i}", value=0.0, step=0.0001, format="%.6f"
            )

    features["Time"] = time
    features["Amount"] = amount

    if st.button("Detect Fraud", type="primary", use_container_width=True):
        pred, proba = get_prediction(features)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        if pred == 1:
            st.markdown(
                '<div class="fraud-alert">⚠️ FRAUD DETECTED!</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="legit-alert">✓ LEGITIMATE TRANSACTION</div>',
                unsafe_allow_html=True,
            )

        with col1:
            st.metric("Fraud Probability", f"{proba * 100:.2f}%")
        with col2:
            st.metric("Confidence", f"{max(proba, 1 - proba) * 100:.2f}%")
        with col3:
            st.metric("Risk Level", "HIGH" if pred == 1 else "LOW")

with tab2:
    st.header("Batch Prediction")
    st.info("Upload a CSV file with V1-V28, Amount, and Time columns")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Process Batch", type="primary"):
            results = []
            for idx, row in df.iterrows():
                user_features = {
                    f"V{i}": float(row.get(f"V{i}", 0) or 0) for i in range(1, 29)
                }
                user_features["Amount"] = float(row.get("Amount", 0) or 0)
                user_features["Time"] = float(row.get("Time", 0) or 0)
                pred, proba = get_prediction(user_features)
                results.append(
                    {
                        "Prediction": "Fraud" if pred == 1 else "Legit",
                        "Probability": f"{proba * 100:.2f}%",
                    }
                )

            results_df = pd.DataFrame(results)
            st.write("Results:")
            st.dataframe(results_df)

            fraud_count = sum(1 for r in results if r["Prediction"] == "Fraud")
            st.metric("Total Transactions", len(results))
            st.metric("Fraudulent Transactions", fraud_count)
            st.metric("Legitimate Transactions", len(results) - fraud_count)

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #888; margin-top: 2rem;">
    <p>Powered by XGBoost ML Model | Credit Card Fraud Detection System</p>
</div>
""",
    unsafe_allow_html=True,
)
