import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, precision_score, recall_score

# ============================================================
# MODULE A: LOT-WISE Z-SCORE ANOMALY DETECTION
# ============================================================

# Locate dataset relative to this Python file
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic_burnin_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Total components:", len(df))

# ------------------------------------------------------------
# 1. Calculate lot-wise statistics
# ------------------------------------------------------------

columns = [
    "Value_0h",
    "Value_24h",
    "Value_96h",
    "Value_168h"
]

lot_stats = df.groupby("Lot_ID")[columns].agg(["mean", "std"])


# ------------------------------------------------------------
# 2. Calculate Z-score for 168h reading
# ------------------------------------------------------------

def calculate_z_score(row):
    lot = row["Lot_ID"]

    mean = lot_stats.loc[lot, ("Value_168h", "mean")]
    std = lot_stats.loc[lot, ("Value_168h", "std")]

    if std == 0:
        return 0

    return (row["Value_168h"] - mean) / std


df["Z_Score_168h"] = df.apply(calculate_z_score, axis=1)


# ------------------------------------------------------------
# 3. Test different Z-score thresholds
# ------------------------------------------------------------

print("\nThreshold Testing:")

for cutoff in [1, 1.5, 2, 2.5, 3]:

    flags = df["Z_Score_168h"].abs() > cutoff

    predictions = flags.map({
        True: "Anomaly",
        False: "Normal"
    })

    precision = precision_score(
        df["Ground_Truth"],
        predictions,
        pos_label="Anomaly"
    )

    recall = recall_score(
        df["Ground_Truth"],
        predictions,
        pos_label="Anomaly"
    )

    print(
        f"Cutoff {cutoff}: "
        f"Precision={precision:.2f}, "
        f"Recall={recall:.2f}"
    )


# ------------------------------------------------------------
# 4. Final threshold
# ------------------------------------------------------------

FINAL_CUTOFF = 1.0

df["ModuleA_Flag"] = (
    df["Z_Score_168h"].abs() > FINAL_CUTOFF
)

df["ModuleA_Prediction"] = df["ModuleA_Flag"].map({
    True: "Anomaly",
    False: "Normal"
})


# ------------------------------------------------------------
# 5. Evaluate Module A
# ------------------------------------------------------------

precision = precision_score(
    df["Ground_Truth"],
    df["ModuleA_Prediction"],
    pos_label="Anomaly"
)

recall = recall_score(
    df["Ground_Truth"],
    df["ModuleA_Prediction"],
    pos_label="Anomaly"
)

print("\n========================================")
print("MODULE A RESULTS")
print("========================================")
print("Final Z-score cutoff:", FINAL_CUTOFF)
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))

print("\nClassification Report:")
print(
    classification_report(
        df["Ground_Truth"],
        df["ModuleA_Prediction"]
    )
)


# ------------------------------------------------------------
# 6. Display sample predictions
# ------------------------------------------------------------

print("\nSample Results:")

print(
    df[
        [
            "Component_ID",
            "Lot_ID",
            "Value_168h",
            "Z_Score_168h",
            "ModuleA_Prediction",
            "Ground_Truth"
        ]
    ].head(20)
)
