# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os, numpy as np, json

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

# -- load scaler
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

# -- discover saved models
def load_models():
    reg_models_h = {}
    reg_models_c = {}
    clf_models_h = {}
    clf_models_c = {}
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith('model_heating_reg_'):
            name = fname.replace('model_heating_reg_', '').replace('.pkl', '')
            reg_models_h[name] = joblib.load(os.path.join(MODEL_DIR, fname))
        if fname.startswith('model_cooling_reg_'):
            name = fname.replace('model_cooling_reg_', '').replace('.pkl', '')
            reg_models_c[name] = joblib.load(os.path.join(MODEL_DIR, fname))
        if fname.startswith('model_heating_clf_'):
            name = fname.replace('model_heating_clf_', '').replace('.pkl', '')
            clf_models_h[name] = joblib.load(os.path.join(MODEL_DIR, fname))
        if fname.startswith('model_cooling_clf_'):
            name = fname.replace('model_cooling_clf_', '').replace('.pkl', '')
            clf_models_c[name] = joblib.load(os.path.join(MODEL_DIR, fname))
    return reg_models_h, reg_models_c, clf_models_h, clf_models_c

REG_H, REG_C, CLF_H, CLF_C = load_models()

# load metrics if present
metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "ok",
        "available_regressors_heating": list(REG_H.keys()),
        "available_regressors_cooling": list(REG_C.keys()),
        "available_classifiers_heating": list(CLF_H.keys()),
        "available_classifiers_cooling": list(CLF_C.keys())
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics or {"error": "metrics not found"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    JSON payload:
    {
      "features": [f1,..,f8],
      "reg_model": "<name>" (optional, defaults to 'LinearRegression'),
      "clf_model": "<name>" (optional, defaults to 'LogisticRegression'),
      "return_all": false (optional, if true returns predictions from all saved models)
    }
    """
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON"}), 400

    features = payload.get('features')
    if features is None:
        return jsonify({"error": "Missing 'features' list"}), 400

    try:
        arr = np.array(features, dtype=float).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Invalid feature values: {e}"}), 400

    arr_scaled = scaler.transform(arr)

    reg_model_name = payload.get('reg_model', 'LinearRegression')
    clf_model_name = payload.get('clf_model', 'LogisticRegression')
    return_all = bool(payload.get('return_all', False))

    response = {'input': features, 'results': {}}

    # If return_all True: include outputs from all saved regressors & classifiers
    if return_all:
        response['results']['heating_regression'] = {}
        response['results']['cooling_regression'] = {}
        response['results']['heating_classification'] = {}
        response['results']['cooling_classification'] = {}

        for name, m in REG_H.items():
            response['results']['heating_regression'][name] = float(m.predict(arr_scaled)[0])
        for name, m in REG_C.items():
            response['results']['cooling_regression'][name] = float(m.predict(arr_scaled)[0])
        for name, m in CLF_H.items():
            response['results']['heating_classification'][name] = str(m.predict(arr_scaled)[0])
        for name, m in CLF_C.items():
            response['results']['cooling_classification'][name] = str(m.predict(arr_scaled)[0])
        return jsonify(response)

    # Otherwise return from chosen models (if name not found -> error)
    if reg_model_name not in REG_H or reg_model_name not in REG_C:
        return jsonify({"error": f"reg_model '{reg_model_name}' not available. Use / to list models."}), 400
    if clf_model_name not in CLF_H or clf_model_name not in CLF_C:
        return jsonify({"error": f"clf_model '{clf_model_name}' not available. Use / to list models."}), 400

    heat_reg_pred = float(REG_H[reg_model_name].predict(arr_scaled)[0])
    cool_reg_pred = float(REG_C[reg_model_name].predict(arr_scaled)[0])
    heat_clf_pred = str(CLF_H[clf_model_name].predict(arr_scaled)[0])
    cool_clf_pred = str(CLF_C[clf_model_name].predict(arr_scaled)[0])

    response['results'] = {
        'heating': {
            'regression_model': reg_model_name,
            'regression_value': round(heat_reg_pred, 3),
            'classification_model': clf_model_name,
            'class_label': heat_clf_pred
        },
        'cooling': {
            'regression_model': reg_model_name,
            'regression_value': round(cool_reg_pred, 3),
            'classification_model': clf_model_name,
            'class_label': cool_clf_pred
        }
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
