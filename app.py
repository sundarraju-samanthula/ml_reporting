from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# ==============================
# App Setup
# ==============================
app = Flask(__name__)
CORS(app)

# ==============================
# Load Models Safely
# ==============================
try:
    severity_model = joblib.load("severity_model.pkl")
    priority_model = joblib.load("priority_model.pkl")

    le_area = joblib.load("le_area.pkl")
    le_access = joblib.load("le_access.pkl")
    le_severity = joblib.load("le_severity.pkl")
    le_priority = joblib.load("le_priority.pkl")

    print("✅ Models loaded successfully")

except Exception as e:
    print("❌ Model loading failed:", e)
    raise e


# ==============================
# Health Check Route
# ==============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ML Reporting API is live 🚀",
        "status": "running"
    })


# ==============================
# Prediction Logic
# ==============================
def predict_waste(data):

    required_fields = [
        "latitude",
        "longitude",
        "area_type",
        "population",
        "nearby_houses",
        "road_accessibility"
    ]

    # Validate required fields
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing field: {field}")

    # Clean + convert values
    latitude = float(data["latitude"])
    longitude = float(data["longitude"])
    area_type = str(data["area_type"]).strip()
    population = int(data["population"])
    nearby_houses = int(data["nearby_houses"])
    road_access = str(data["road_accessibility"]).strip()

    # Encode categorical values
    area_encoded = le_area.transform([area_type])[0]
    access_encoded = le_access.transform([road_access])[0]

    # Create feature array
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

    # Decode
    severity = le_severity.inverse_transform([severity_pred])[0]
    priority = le_priority.inverse_transform([priority_pred])[0]

    return {
        "severity": severity,
        "priority": priority
    }


# ==============================
# Predict Endpoint
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON body provided"
            }), 400

        result = predict_waste(data)

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ==============================
# Render Production Port Binding
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
