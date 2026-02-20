from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# ==========================
# App Setup
# ==========================
app = Flask(__name__)
CORS(app)

# ==========================
# Load Models
# ==========================
try:
    severity_model = joblib.load("models/severity_model.pkl")
    priority_model = joblib.load("models/priority_model.pkl")

    le_area = joblib.load("models/le_area.pkl")
    le_access = joblib.load("models/le_access.pkl")
    le_severity = joblib.load("models/le_severity.pkl")
    le_priority = joblib.load("models/le_priority.pkl")

    print("✅ Models loaded successfully")

except Exception as e:
    print("❌ Model loading failed:", e)
    raise e


# ==========================
# Health Check
# ==========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ML Reporting API is live 🚀",
        "status": "running"
    })


# ==========================
# Single Prediction
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON body provided"
            }), 400

        required_fields = [
            "latitude",
            "longitude",
            "area_type",
            "population",
            "nearby_houses",
            "road_accessibility"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "error",
                    "message": f"Missing field: {field}"
                }), 400

        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        area_type = str(data["area_type"]).strip()
        population = int(data["population"])
        nearby_houses = int(data["nearby_houses"])
        road_access = str(data["road_accessibility"]).strip()

        # Encode
        area_encoded = le_area.transform([area_type])[0]
        access_encoded = le_access.transform([road_access])[0]

        features = np.array([[
            latitude,
            longitude,
            area_encoded,
            population,
            nearby_houses,
            access_encoded
        ]])

        # Predict
        severity_pred = severity_model.predict(features)[0]
        priority_pred = priority_model.predict(features)[0]

        severity = le_severity.inverse_transform([severity_pred])[0]
        priority = le_priority.inverse_transform([priority_pred])[0]

        return jsonify({
            "status": "success",
            "result": {
                "severity": severity,
                "priority": priority
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ==========================
# Batch Prediction (NEW)
# ==========================
@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    try:
        reports = request.get_json()

        if not isinstance(reports, list):
            return jsonify({
                "status": "error",
                "message": "Expected a list of reports"
            }), 400

        results = []

        for data in reports:
            latitude = float(data["latitude"])
            longitude = float(data["longitude"])
            area_type = str(data["area_type"]).strip()
            population = int(data["population"])
            nearby_houses = int(data["nearby_houses"])
            road_access = str(data["road_accessibility"]).strip()

            area_encoded = le_area.transform([area_type])[0]
            access_encoded = le_access.transform([road_access])[0]

            features = np.array([[
                latitude,
                longitude,
                area_encoded,
                population,
                nearby_houses,
                access_encoded
            ]])

            severity_pred = severity_model.predict(features)[0]
            priority_pred = priority_model.predict(features)[0]

            severity = le_severity.inverse_transform([severity_pred])[0]
            priority = le_priority.inverse_transform([priority_pred])[0]

            results.append({
                "severity": severity,
                "priority": priority
            })

        return jsonify({
            "status": "success",
            "results": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ==========================
# Railway Port Binding
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)