from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models
severity_model = joblib.load("severity_model.pkl")
priority_model = joblib.load("priority_model.pkl")

le_area = joblib.load("le_area.pkl")
le_access = joblib.load("le_access.pkl")
le_severity = joblib.load("le_severity.pkl")
le_priority = joblib.load("le_priority.pkl")


def predict_waste(data):
    try:
        # Normalize inputs (IMPORTANT FIX)
        area_type = data["area_type"].strip()
        road_access = data["road_accessibility"].strip()

        # Encode
        area = le_area.transform([area_type])[0]
        access = le_access.transform([road_access])[0]

        features = np.array([[ 
            float(data["latitude"]),
            float(data["longitude"]),
            area,
            int(data["population"]),
            int(data["nearby_houses"]),
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

    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")


@app.route("/")
def home():
    return jsonify({
        "message": "ML Reporting API is live 🚀",
        "status": "running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"status": "error", "message": "No JSON received"}), 400

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
