from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")

MODEL_PATH = os.path.join("..", "Models", "best_model_xgb.pkl")
SCALER_PATH = os.path.join("..", "Models", "scaler.pkl")
TOP_FEATURES_PATH = os.path.join("..", "Models", "top_features.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
top_features = joblib.load(TOP_FEATURES_PATH)

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    raw_row = {k: float(data.get(k, 0.0)) for k in RAW_FEATURES}
    scaled_row = {
        **{k: raw_row[k] for k in RAW_FEATURES if k not in ["Amount", "Time"]},
        "scaled_Amount": (raw_row["Amount"] - AMOUNT_MEAN) / AMOUNT_STD,
        "scaled_Time": (raw_row["Time"] - TIME_MEAN) / TIME_STD,
    }
    scaler_input = [scaled_row[k] for k in SCALED_FEATURES]
    X_scaled = scaler.transform([scaler_input])
    model_input = {top_features[i]: X_scaled[0][i] for i in range(len(top_features))}
    X_model = np.array([[model_input[f] for f in top_features]])
    proba = float(model.predict_proba(X_model)[0][1])
    pred = int(proba >= 0.5)
    return jsonify({"prediction": pred, "probability": proba})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
