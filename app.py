from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
model = joblib.load(os.path.join('Models', 'best_model.pkl'))
scaler = joblib.load(os.path.join('Models', 'scaler_ann.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # expects a list of features
    X = scaler.transform([data])
    pred = int(model.predict(X)[0])
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
