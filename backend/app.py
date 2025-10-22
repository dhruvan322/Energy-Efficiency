from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # allow frontend access

# Set up correct file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'heating_load_model.joblib')
scaler_path = os.path.join(current_dir, '..', 'models', 'scaler.joblib')

# Load model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError as e:
    print(f"Error: Model or scaler file not found. Please check paths:\n{model_path}\n{scaler_path}")
    raise e

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "Energy Efficiency Prediction API is running!"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing features in request'
            }), 400
            
        features = np.array(data['features'])
        
        # Validate feature count
        if features.shape[0] != 8:
            return jsonify({
                'error': 'Expected 8 features'
            }), 400
            
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_heating_load': float(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Ensure the model files exist before starting the server
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "Model files not found. Please run train_model.py first to generate the models."
        )
    app.run(debug=True, host='0.0.0.0', port=5000)