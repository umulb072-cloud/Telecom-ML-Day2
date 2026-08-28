# Day 2 — Train Your First ML Model

## Objective

The goal of Day 2 was to train a baseline machine learning model using the feature-engineered telecom dataset.

## Dataset

The dataset contains 500 telecom/network records with features such as:

- RSRP
- RSRQ
- SINR
- Latency
- Packet Drop Rate
- CPU Utilization
- Throughput
- Signal Quality
- High Latency
- High Packet Drop

The target variable is `Status`, which contains:

- Normal
- Anomaly

## Data Preparation

The following steps were performed:

1. Loaded the feature-engineered dataset.
2. Separated the input features (`X`) from the target (`y`).
3. Checked and handled missing values.
4. Converted the target labels:
   - Normal = 0
   - Anomaly = 1
5. Split the dataset into training and testing sets using an 80/20 split.
6. Applied StandardScaler for feature scaling.

## Machine Learning Model

A Logistic Regression model was selected as the baseline classification model.

The model was trained using the training dataset and then used to predict the classes of the test dataset.

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

## Results

Add your actual results below:

- Accuracy: 0.89
- Precision: 0.862
- Recall: 0.781
- F1-score: 0.82

## Predictions

The model predictions were saved as:

`Data/model_predictions.csv`

## Conclusion

A baseline Logistic Regression classification model was successfully trained on the telecom dataset. The model was evaluated using standard classification metrics and its predictions were saved for further analysis.