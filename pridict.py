import joblib
import numpy as np

# ==============================
# Load Saved Models
# ==============================
severity_model = joblib.load("severity_model.pkl")
priority_model = joblib.load("priority_model.pkl")

# Load Encoders
le_area = joblib.load("le_area.pkl")
le_access = joblib.load("le_access.pkl")
le_severity = joblib.load("le_severity.pkl")
le_priority = joblib.load("le_priority.pkl")

# ==============================
# Prediction Function
# ==============================
def predict_waste(data):
    try:
        # Encode categorical features
        area = le_area.transform([data["area_type"]])[0]
        access = le_access.transform([data["road_accessibility"]])[0]

        # Prepare feature array
        features = np.array([[
            data["latitude"],
            data["longitude"],
            area,
            data["population"],
            data["nearby_houses"],
            access
        ]])

        # Predict
        severity_pred = severity_model.predict(features)[0]
        priority_pred = priority_model.predict(features)[0]

        # Decode results
        severity = le_severity.inverse_transform([severity_pred])[0]
        priority = le_priority.inverse_transform([priority_pred])[0]

        return {
            "severity": severity,
            "priority": priority
        }

    except Exception as e:
        return {"error": str(e)}


# ==============================
# Example Test
# ==============================
if __name__ == "__main__":
    sample_data = {
        "latitude": 17.05,
        "longitude": 81.72,
        "area_type": "Urban",
        "population": 2500,
        "nearby_houses": 400,
        "road_accessibility": "Easy"
    }

    result = predict_waste(sample_data)
    print("\nPrediction Result:")
    print(result)
