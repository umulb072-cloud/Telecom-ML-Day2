# Task 1 — Production Pipeline, Imbalance Handling & Explainability

## 1. Overview

A Scikit-Learn Pipeline was developed to combine data preprocessing and a Random Forest classifier. The pipeline includes median imputation, feature scaling, and a balanced Random Forest model.

## 2. Class Imbalance

A realistic telecom failure scenario was simulated with 95% Normal samples and 5% Anomaly samples. The Random Forest was trained using `class_weight="balanced"` to give greater importance to the minority class.

## 3. Evaluation

The model was evaluated using Recall, Precision, and PR-AUC instead of relying mainly on Accuracy. At the default threshold of 0.50, the model achieved high precision but lower anomaly recall.

Thresholds of 0.50, 0.35, and 0.30 were compared to investigate whether lowering the decision threshold could improve the detection of network failures.

## 4. Why Recall is Important

High Recall is more important than high Accuracy in telecom fault prediction because network failures are rare but can have serious operational consequences. A False Negative means that an actual network failure was missed by the model.

Missing a network failure can result in service degradation, downtime, and poor user experience. Therefore, telecom operators generally prefer a model that detects as many real failures as possible, even if this produces some additional false alarms.

## 5. Explainability

Random Forest feature importance was used to identify which network metrics contributed most to anomaly predictions. The resulting feature importance visualization was saved as:

`shap_summary.png`

The most important network metrics can help operators understand which measurements are associated with detected network faults.

## 6. Deliverables

- `pipeline_and_xai.py`
- `shap_summary.png`
- `Task1_Summary.md`