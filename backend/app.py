from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='../frontend', template_folder='./frontend')
model = joblib.load(os.path.join('../models', 'best_model.pkl'))
scaler = joblib.load(os.path.join('../models', 'scaler_ann.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # expects a list of features
    X = scaler.transform([data])
    pred = int(model.predict(X)[0])
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
