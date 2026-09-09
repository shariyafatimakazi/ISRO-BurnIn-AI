import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# MODULE B: EARLY-STAGE RANDOM FOREST ANOMALY PREDICTION
# ============================================================

# Locate dataset relative to this Python file
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic_burnin_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 1. Create Early Drift Feature
# ------------------------------------------------------------

# Early drift measures the change between 0h and 24h
df["Early_Drift"] = df["Value_24h"] - df["Value_0h"]


# ------------------------------------------------------------
# 2. Select Early-Stage Input Features
# ------------------------------------------------------------

features = [
    "Value_0h",
    "Value_24h",
    "Early_Drift"
]

X = df[features]
y = df["Ground_Truth"]


# ------------------------------------------------------------
# 3. Split Data into Training and Testing Sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 4. Train Random Forest Model
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Module B model trained successfully!")


# ------------------------------------------------------------
# 5. Make Predictions on Unseen Test Data
# ------------------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModule B Accuracy:", round(accuracy * 100, 2), "%")


# ------------------------------------------------------------
# 6. Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ------------------------------------------------------------
# 7. Confusion Matrix
# ------------------------------------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ------------------------------------------------------------
# 8. Calculate Anomaly Risk Score
# ------------------------------------------------------------

probabilities = model.predict_proba(X_test)

anomaly_index = list(model.classes_).index("Anomaly")

risk_score = probabilities[:, anomaly_index] * 100


# ------------------------------------------------------------
# 9. Create Results Table
# ------------------------------------------------------------

results = X_test.copy()

results["Actual"] = y_test.values
results["Prediction"] = y_pred
results["Risk_Score"] = risk_score.round(2)


# ------------------------------------------------------------
# 10. Convert Risk Score into Risk Category
# ------------------------------------------------------------

def risk_category(score):

    if score < 30:
        return "Low Risk"

    elif score < 70:
        return "Warning"

    else:
        return "High Risk"


results["Risk_Category"] = results["Risk_Score"].apply(
    risk_category
)


# ------------------------------------------------------------
# 11. Display Sample Results
# ------------------------------------------------------------

print("\nSample Risk Results:")

print(
    results[
        [
            "Value_0h",
            "Value_24h",
            "Early_Drift",
            "Actual",
            "Prediction",
            "Risk_Score",
            "Risk_Category"
        ]
    ].head(15)
)


# ------------------------------------------------------------
# 12. Example Prediction Function
# ------------------------------------------------------------

def predict_component(value_0h, value_24h):

    # Calculate early drift
    early_drift = value_24h - value_0h

    # Create input data
    input_data = pd.DataFrame([{
        "Value_0h": value_0h,
        "Value_24h": value_24h,
        "Early_Drift": early_drift
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability-based risk score
    probabilities = model.predict_proba(input_data)

    anomaly_index = list(model.classes_).index("Anomaly")

    risk_score = probabilities[0][anomaly_index] * 100

    # Risk category
    category = risk_category(risk_score)

    print("\n----- MODULE B RESULT -----")
    print("Value at 0h:", value_0h)
    print("Value at 24h:", value_24h)
    print("Early Drift:", round(early_drift, 3))
    print("Prediction:", prediction)
    print("Risk Score:", round(risk_score, 2), "%")
    print("Risk Category:", category)


# Example
predict_component(10.1, 12.2)
