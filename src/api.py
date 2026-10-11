from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cybersecurity_model.joblib"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "duration",
    "source_bytes",
    "destination_bytes",
    "failed_logins",
    "connection_count",
    "port_risk"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Cybersecurity Threat Detection API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send valid JSON data"}), 400

    missing = [f for f in FEATURES if f not in data]

    if missing:
        return jsonify({
            "error": "Missing features",
            "missing_features": missing
        }), 400

    try:
        values = [float(data[f]) for f in FEATURES]

        if any(v < 0 for v in values):
            return jsonify({"error": "Values must be non-negative"}), 400

        input_data = pd.DataFrame([values], columns=FEATURES)
        prediction = int(model.predict(input_data)[0])

        return jsonify({
            "prediction": prediction,
            "threat_status": "THREAT DETECTED" if prediction == 1 else "NORMAL"
        })

    except (TypeError, ValueError):
        return jsonify({"error": "Feature values must be numeric"}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)