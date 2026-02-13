from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ==============================
# Load Models & Encoders Once
# ==============================
severity_model = joblib.load("severity_model.pkl")
priority_model = joblib.load("priority_model.pkl")

le_area = joblib.load("le_area.pkl")
le_access = joblib.load("le_access.pkl")
le_severity = joblib.load("le_severity.pkl")
le_priority = joblib.load("le_priority.pkl")


# ==============================
# Prediction Function
# ==============================
def predict_waste(data):
    area = le_area.transform([data["area_type"]])[0]
    access = le_access.transform([data["road_accessibility"]])[0]

    features = np.array([[
        data["latitude"],
        data["longitude"],
        area,
        data["population"],
        data["nearby_houses"],
        access
    ]])

    severity_pred = severity_model.predict(features)[0]
    priority_pred = priority_model.predict(features)[0]

    severity = le_severity.inverse_transform([severity_pred])[0]
    priority = le_priority.inverse_transform([priority_pred])[0]

    return {
        "severity": severity,
        "priority": priority
    }


# ==============================
# API Endpoint
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        result = predict_waste(data)

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
