import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    precision_score,
    average_precision_score,
    classification_report
)

# Load dataset
df = pd.read_csv("Data/feature_engineered_data.csv")

print("Dataset shape:", df.shape)
print("\nOriginal target distribution:")
print(df["Status"].value_counts())

# Create an imbalanced dataset: 95% Normal, 5% Anomaly
normal_data = df[df["Status"] == "Normal"]
anomaly_data = df[df["Status"] == "Anomaly"]

normal_sample = normal_data.sample(
    n=475,
    replace=True,
    random_state=42
)

anomaly_sample = anomaly_data.sample(
    n=25,
    random_state=42
)

imbalanced_df = pd.concat(
    [normal_sample, anomaly_sample]
).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nImbalanced dataset:")
print(imbalanced_df["Status"].value_counts())
print("\nImbalanced percentages:")
print(imbalanced_df["Status"].value_counts(normalize=True) * 100)

# Separate features (X) and target (y)
X = imbalanced_df.drop(columns=["Status"])
y = imbalanced_df["Status"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.value_counts())

# Convert target labels to numbers
y = y.map({
    "Normal": 0,
    "Anomaly": 1
})

print("\nEncoded target:")
print(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Build preprocessing + Random Forest pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

# Train the pipeline
# Train the pipeline
pipeline.fit(X_train, y_train)

print("\nPipeline training completed successfully.")

# Save the trained pipeline
joblib.dump(pipeline, "telecom_rf_pipeline.joblib")

print("Model saved successfully!")

# Save the trained pipeline
joblib.dump(pipeline, "telecom_rf_pipeline.joblib")

print("Model saved successfully!")

# Make predictions on the test data
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Default threshold = 0.50
y_pred_05 = (y_prob >= 0.50).astype(int)

# Calculate evaluation metrics
recall_05 = recall_score(y_test, y_pred_05)
precision_05 = precision_score(y_test, y_pred_05, zero_division=0)
pr_auc = average_precision_score(y_test, y_prob)

print("\n--- Evaluation at Threshold 0.50 ---")
print("Recall:", round(recall_05, 4))
print("Precision:", round(precision_05, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_05,
    target_names=["Normal", "Anomaly"],
    zero_division=0
))
# Threshold Moving
thresholds = [0.50, 0.35, 0.30]

print("\n--- Threshold Comparison ---")

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)

    print(
        f"Threshold: {threshold} | "
        f"Recall: {recall:.2f} | "
        f"Precision: {precision:.2f}"
    )
    # Feature Importance Analysis

# Get the trained Random Forest from the pipeline
rf_model = pipeline.named_steps["classifier"]

# Get feature importance values
importance = rf_model.feature_importances_

# Create a feature importance table
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\n--- Feature Importance ---")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Network Metric")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

# Save the plot
plt.savefig("shap_summary.png", dpi=300)
plt.show()

print("\nFeature importance plot saved as shap_summary.png")


