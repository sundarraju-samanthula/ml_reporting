import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("waste_dataset.csv")

# ==============================
# Encode Categorical Columns
# ==============================
le_area = LabelEncoder()
le_access = LabelEncoder()
le_severity = LabelEncoder()
le_priority = LabelEncoder()

df["area_type"] = le_area.fit_transform(df["area_type"])
df["road_accessibility"] = le_access.fit_transform(df["road_accessibility"])
df["severity"] = le_severity.fit_transform(df["severity"])
df["priority"] = le_priority.fit_transform(df["priority"])

# ==============================
# Define Features & Targets
# ==============================
X = df[[
    "latitude",
    "longitude",
    "area_type",
    "population",
    "nearby_houses",
    "road_accessibility"
]]

y_severity = df["severity"]
y_priority = df["priority"]

# ==============================
# Split Data
# ==============================
X_train, X_test, y_train_s, y_test_s = train_test_split(
    X, y_severity, test_size=0.2, random_state=42
)

X_train2, X_test2, y_train_p, y_test_p = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

# ==============================
# Train Models
# ==============================
severity_model = RandomForestClassifier(random_state=42)
priority_model = RandomForestClassifier(random_state=42)

severity_model.fit(X_train, y_train_s)
priority_model.fit(X_train2, y_train_p)

# ==============================
# Evaluate
# ==============================
print("\nSeverity Model Report:")
print(classification_report(y_test_s, severity_model.predict(X_test)))

print("\nPriority Model Report:")
print(classification_report(y_test_p, priority_model.predict(X_test2)))

# ==============================
# Save Models & Encoders
# ==============================
joblib.dump(severity_model, "severity_model.pkl")
joblib.dump(priority_model, "priority_model.pkl")

joblib.dump(le_area, "le_area.pkl")
joblib.dump(le_access, "le_access.pkl")
joblib.dump(le_severity, "le_severity.pkl")
joblib.dump(le_priority, "le_priority.pkl")

print("\nModels and encoders saved successfully.")
