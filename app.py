import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model and scaler
MODEL_PATH = os.path.join('.', 'Models', 'best_model.pkl')
SCALER_PATH = os.path.join('.', 'Models', 'scaler_ann.pkl')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(page_title="Credit Card Default Prediction", layout="centered")
st.title("Credit Card Default Prediction")
st.markdown("""
This app predicts the probability of credit card default using a trained machine learning model.\
Fill in the features below and click **Predict**.\

*Model: Trained on anonymized credit card transaction data.*
""")

with st.expander("ℹ️ About this app"):
    st.write("""
    - **Features:** V1–V28, Amount, Time (technical features from PCA)
    - **Model:** Your trained model (e.g., XGBoost, RandomForest, etc.)
    - **Scaler:** StandardScaler or similar
    - **Explainability:** SHAP plots (if available)
    """)

# Input fields
def user_input_features():
    amount = st.number_input('Amount', min_value=0.0, value=100.0)
    time = st.number_input('Time', min_value=0.0, value=50000.0)
    v_features = []
    for i in range(1, 29):
        v = st.number_input(f'V{i}', value=0.0)
        v_features.append(v)
    features = [amount, time] + v_features
    return features

features = user_input_features()

if st.button('Predict'):
    # Arrange features in correct order for model
    # (Adjust order if your model expects differently)
    X = scaler.transform([features])
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None
    st.success(f'Prediction: {"Default" if pred else "No Default"}')
    if proba is not None:
        st.info(f'Probability of Default: {proba:.2%}')

    # Example: Show a simple plot (replace with SHAP or other explainability plots if available)
    st.subheader("Feature Values")
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.barplot(x=[f'V{i}' for i in range(1, 29)], y=features[2:], ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Placeholder for SHAP or other explainability plots
    st.subheader("Model Explainability (Coming Soon)")
    st.write("Add SHAP or other explainability plots here.")

st.markdown("---")
st.caption("Built with Streamlit. [GitHub Repo](https://github.com/yourusername/yourrepo)")
