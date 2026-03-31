import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

DEFAULT_VALUES = {
    "V1": -1.36,
    "V2": -0.07,
    "V3": 2.54,
    "V4": 1.38,
    "V5": -0.34,
    "V6": 0.46,
    "V7": 0.24,
    "V8": 0.10,
    "V9": 0.36,
    "V10": 0.09,
    "V11": -0.55,
    "V12": -0.62,
    "V13": -0.99,
    "V14": -0.31,
    "V15": 1.47,
    "V16": -0.47,
    "V17": 0.21,
    "V18": 0.03,
    "V19": 0.40,
    "V20": 0.25,
    "V21": -0.02,
    "V22": 0.28,
    "V23": -0.11,
    "V24": 0.07,
    "V25": 0.13,
    "V26": -0.19,
    "V27": 0.13,
    "V28": -0.02,
    "Amount": 149.62,
    "Time": 0.0,
}

FRAUD_EXAMPLES = {
    "High Amount Fraud": {
        "V1": -11.88,
        "V2": 10.07,
        "V3": -9.83,
        "V4": -2.07,
        "V5": -5.36,
        "V6": -2.61,
        "V7": -4.92,
        "V8": 7.31,
        "V9": 1.91,
        "V10": 4.36,
        "V11": -1.59,
        "V12": 2.71,
        "V13": -0.69,
        "V14": 4.63,
        "V15": -0.92,
        "V16": 1.11,
        "V17": 1.99,
        "V18": 0.51,
        "V19": -0.68,
        "V20": 1.48,
        "V21": 0.21,
        "V22": 0.11,
        "V23": 1.01,
        "V24": -0.51,
        "V25": 1.44,
        "V26": 0.25,
        "V27": 0.94,
        "V28": 0.82,
        "Amount": 25691.16,
        "Time": 172786.0,
    },
    "Low Amount Fraud": {
        "V1": -3.21,
        "V2": 2.45,
        "V3": -2.12,
        "V4": 1.35,
        "V5": -0.92,
        "V6": -0.78,
        "V7": 1.25,
        "V8": 0.45,
        "V9": -0.85,
        "V10": 2.15,
        "V11": -1.22,
        "V12": 1.85,
        "V13": -0.45,
        "V14": 2.85,
        "V15": -0.68,
        "V16": 0.95,
        "V17": 1.45,
        "V18": -0.32,
        "V19": -0.55,
        "V20": 0.78,
        "V21": 0.12,
        "V22": 0.35,
        "V23": -0.22,
        "V24": 0.45,
        "V25": -0.18,
        "V26": 0.55,
        "V27": -0.35,
        "V28": 0.22,
        "Amount": 5.50,
        "Time": 85000.0,
    },
}

st.markdown(
    """
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #667eea; text-align: center; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #888; text-align: center; margin-bottom: 2rem; }
    .fraud-alert { background: linear-gradient(135deg, #ff4757, #ff6b81); color: white; padding: 2rem; border-radius: 12px; text-align: center; font-size: 1.5rem; }
    .legit-alert { background: linear-gradient(135deg, #2ed573, #7bed9f); color: white; padding: 2rem; border-radius: 12px; text-align: center; font-size: 1.5rem; }
    .plot-container { background: #f8f9fa; padding: 1rem; border-radius: 12px; margin-top: 1rem; }
    .feature-highlight { background: #fff3cd; padding: 0.5rem; border-radius: 8px; border-left: 4px solid #ffc107; }
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
    return pred, proba, model_input


def get_feature_importance():
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": top_features, "Importance": importance})
    feat_imp = feat_imp.sort_values("Importance", ascending=True)
    return feat_imp


def create_gauge_chart(probability):
    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw={"projection": "polar"})
    theta = np.linspace(0, np.pi, 100)
    r = np.ones(100)
    colors = ["#2ed573", "#ff4757"]
    for i, color in enumerate(colors):
        ax.fill_between(
            theta[i * 50 : (i + 1) * 50],
            0,
            r[i * 50 : (i + 1) * 50],
            alpha=0.3,
            color=color,
        )
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Fraud Probability: {probability * 100:.1f}%", fontsize=12, fontweight="bold"
    )
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return fig


def create_prediction_explanation(pred, proba, model_input):
    st.markdown("---")
    st.subheader("📊 Prediction Analysis")

    col1, col2, col3 = st.columns(3)

    if pred == 1:
        st.markdown(
            '<div class="fraud-alert">⚠️ FRAUD DETECTED!</div>', unsafe_allow_html=True
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
        risk = "HIGH" if pred == 1 else "LOW"
        color = "🔴" if pred == 1 else "🟢"
        st.metric("Risk Level", f"{color} {risk}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Feature Importance")
        feat_imp = get_feature_importance()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            feat_imp["Feature"].tail(10),
            feat_imp["Importance"].tail(10),
            color="#667eea",
        )
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Most Important Features", fontweight="bold")
        st.pyplot(fig)

    with col2:
        st.markdown("### 📈 Risk Gauge")
        fig_gauge = create_gauge_chart(proba)
        st.pyplot(fig_gauge)

    st.markdown("### 🎯 Key Contributing Factors")

    top_contributing = sorted(
        model_input.items(), key=lambda x: abs(x[1]), reverse=True
    )[:5]
    contributing_df = pd.DataFrame(
        top_contributing, columns=["Feature", "Scaled Value"]
    )
    st.dataframe(contributing_df, use_container_width=True)

    risk_factors = []
    for feat, val in top_contributing[:3]:
        if feat in ["V14", "V17", "V12", "V10"]:
            risk_factors.append(
                f"- **{feat}** has high negative value ({val:.3f}), common in fraudulent transactions"
            )
        elif feat == "scaled_Amount":
            risk_factors.append(
                f"- **Amount** is {'unusually high' if val > 1 else 'normal'} for this transaction"
            )
        elif feat == "scaled_Time":
            risk_factors.append(
                f"- **Time** pattern: {'unusual hours' if abs(val) > 1 else 'normal business hours'}"
            )

    if risk_factors:
        st.markdown("#### Risk Analysis:")
        for factor in risk_factors:
            st.markdown(factor)

    st.markdown("### 📋 Transaction Summary")

    summary_data = {
        "Metric": [
            "Transaction Amount",
            "Time from First Transaction",
            "Prediction",
            "Confidence Score",
        ],
        "Value": [
            f"${model_input.get('scaled_Amount', 0) * AMOUNT_STD + AMOUNT_MEAN:.2f}",
            f"{int(model_input.get('scaled_Time', 0) * TIME_STD + TIME_MEAN)} seconds",
            "Fraud" if pred == 1 else "Legitimate",
            f"{max(proba, 1 - proba) * 100:.2f}%",
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


tab1, tab2, tab3 = st.tabs(
    ["🔍 Single Transaction", "📁 Batch Prediction", "ℹ️ About Model"]
)

with tab1:
    st.header("Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input(
            "Time (seconds from first transaction)",
            value=0.0,
            step=1.0,
            help="Seconds elapsed since the first transaction in the dataset",
        )
        amount = st.number_input(
            "Amount ($)", value=149.62, step=0.01, help="Transaction amount in USD"
        )

    st.markdown("#### PCA Features (V1 - V28)")
    st.caption(
        "These are dimensionality-reduced features from PCA transformation. Default values represent a sample legitimate transaction."
    )

    cols = st.columns(4)
    features = {}
    for i in range(1, 29):
        with cols[(i - 1) % 4]:
            default_val = DEFAULT_VALUES.get(f"V{i}", 0.0)
            features[f"V{i}"] = st.number_input(
                f"V{i}", value=float(default_val), step=0.01, format="%.4f"
            )

    features["Time"] = time
    features["Amount"] = amount

    st.markdown("---")
    st.markdown("#### Quick Examples")
    example_col1, example_col2, example_col3 = st.columns(3)

    with example_col1:
        if st.button("📊 Sample Legitimate", use_container_width=True):
            features = DEFAULT_VALUES.copy()
            st.session_state["example_loaded"] = "legit"
            st.rerun()

    with example_col2:
        if st.button("⚠️ High Amount Fraud", use_container_width=True):
            features = FRAUD_EXAMPLES["High Amount Fraud"].copy()
            st.session_state["example_loaded"] = "fraud_high"
            st.rerun()

    with example_col3:
        if st.button("⚠️ Low Amount Fraud", use_container_width=True):
            features = FRAUD_EXAMPLES["Low Amount Fraud"].copy()
            st.session_state["example_loaded"] = "fraud_low"
            st.rerun()

    st.markdown("---")

    if st.button("🔍 Detect Fraud", type="primary", use_container_width=True):
        pred, proba, model_input = get_prediction(features)
        create_prediction_explanation(pred, proba, model_input)

with tab2:
    st.header("Batch Prediction")
    st.info(
        "📤 Upload a CSV file with V1-V28, Amount, and Time columns for bulk prediction"
    )

    sample_csv = pd.DataFrame([DEFAULT_VALUES])
    sample_csv_str = sample_csv.to_csv(index=False)
    st.download_button(
        "📥 Download Sample CSV",
        sample_csv_str,
        "sample_transactions.csv",
        "text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            results = []
            all_model_inputs = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in df.iterrows():
                user_features = {
                    f"V{i}": float(row.get(f"V{i}", 0) or 0) for i in range(1, 29)
                }
                user_features["Amount"] = float(row.get("Amount", 0) or 0)
                user_features["Time"] = float(row.get("Time", 0) or 0)

                pred, proba, model_input = get_prediction(user_features)
                results.append(
                    {
                        "Transaction": idx + 1,
                        "Prediction": "Fraud" if pred == 1 else "Legit",
                        "Fraud Probability": f"{proba * 100:.2f}%",
                        "Confidence": f"{max(proba, 1 - proba) * 100:.2f}%",
                    }
                )
                all_model_inputs.append(model_input)

                progress = (idx + 1) / len(df)
                progress_bar.progress(progress)
                status_text.text(f"Processing transaction {idx + 1} of {len(df)}...")

            results_df = pd.DataFrame(results)

            st.markdown("---")
            st.subheader("📊 Batch Analysis Results")

            col1, col2, col3, col4 = st.columns(4)
            fraud_count = sum(1 for r in results if r["Prediction"] == "Fraud")
            legit_count = len(results) - fraud_count
            avg_fraud_prob = np.mean(
                [float(r["Fraud Probability"].replace("%", "")) for r in results]
            )

            with col1:
                st.metric("Total Transactions", len(results))
            with col2:
                st.metric(
                    "Fraudulent",
                    fraud_count,
                    delta=f"{fraud_count / len(results) * 100:.1f}%",
                )
            with col3:
                st.metric(
                    "Legitimate",
                    legit_count,
                    delta=f"{legit_count / len(results) * 100:.1f}%",
                )
            with col4:
                st.metric("Avg Fraud Prob", f"{avg_fraud_prob:.2f}%")

            st.dataframe(results_df, use_container_width=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            fraud_counts = [fraud_count, legit_count]
            labels = ["Fraud", "Legitimate"]
            colors = ["#ff4757", "#2ed573"]
            axes[0].pie(
                fraud_counts,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            axes[0].set_title("Fraud vs Legitimate Distribution", fontweight="bold")

            fraud_probs = [
                float(r["Fraud Probability"].replace("%", "")) for r in results
            ]
            axes[1].hist(fraud_probs, bins=20, color="#667eea", edgecolor="white")
            axes[1].axvline(x=50, color="red", linestyle="--", label="Threshold (50%)")
            axes[1].set_xlabel("Fraud Probability (%)")
            axes[1].set_ylabel("Count")
            axes[1].set_title("Distribution of Fraud Probabilities", fontweight="bold")
            axes[1].legend()

            st.pyplot(fig)

            csv_results = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv_results,
                "prediction_results.csv",
                "text/csv",
                use_container_width=True,
            )

with tab3:
    st.header("📖 About This Model")

    st.markdown("""
    ### Model Information
    
    This fraud detection system uses **XGBoost Classifier**, an ensemble machine learning algorithm that combines 
    multiple decision trees to make accurate predictions.
    
    #### Features Used:
    - **V1 - V28**: Principal Component Analysis (PCA) transformed features that capture transaction patterns
    - **Amount**: Transaction amount in USD
    - **Time**: Seconds elapsed since the first transaction in the dataset
    
    #### How It Works:
    1. **Scaling**: Amount and Time are standardized using the training set statistics
    2. **Feature Selection**: Top 20 most important features are selected for prediction
    3. **Prediction**: XGBoost model calculates fraud probability
    4. **Threshold**: Transactions with probability ≥ 50% are flagged as fraudulent
    
    #### Model Performance:
    - Trained on the Kaggle Credit Card Fraud Detection dataset
    - Handles class imbalance using appropriate techniques
    - Optimized for high recall (catching most frauds)
    """)

    st.markdown("### Top 10 Most Important Features")
    feat_imp = get_feature_importance()
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10 = feat_imp.tail(10)
    colors = [
        "#ff4757" if "V14" in f or "V17" in f else "#667eea" for f in top_10["Feature"]
    ]
    ax.barh(top_10["Feature"], top_10["Importance"], color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Features for Fraud Detection", fontweight="bold", fontsize=14)
    st.pyplot(fig)

    st.markdown("""
    ### Interpretation Guide:
    
    | Feature | High Negative Value | High Positive Value |
    |---------|---------------------|---------------------|
    | V14, V17 | Likely Fraud ⚠️ | Likely Legit ✓ |
    | V12, V10 | Likely Fraud ⚠️ | Likely Legit ✓ |
    | Amount | High Amount | Normal Amount |
    | Time | Unusual Hours | Business Hours |
    """)

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #888; margin-top: 2rem;">
    <p>🔒 Powered by XGBoost ML Model | Credit Card Fraud Detection System</p>
    <p style="font-size: 0.8rem;">Trained on 284,807 transactions | Accuracy optimized for fraud detection</p>
</div>
""",
    unsafe_allow_html=True,
)
