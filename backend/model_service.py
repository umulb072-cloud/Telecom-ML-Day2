
import joblib
import pandas as pd

from pathlib import Path


FEATURE_NAMES = [
    "RSRP",
    "RSRQ",
    "SINR",
    "Latency",
    "Packet_Drop_Rate",
    "CPU_Utilization",
    "Throughput",
    "Signal_Quality",
    "High_Latency",
    "High_Packet_Drop"
]


class ModelService:

    def __init__(self):
        self.model = None
        self.model_name = None

    def load_model(self):

        project_root = Path(__file__).resolve().parent.parent

        model_path = project_root / "telecom_rf_pipeline.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        self.model = joblib.load(model_path)
        self.model_name = model_path.name

        print("Model loaded successfully!")

    def predict(self, features):

        if self.model is None:
            raise RuntimeError("Model is not loaded")

        data = pd.DataFrame(
            [features],
            columns=FEATURE_NAMES
        )

        probabilities = self.model.predict_proba(data)[0]

        classes = list(self.model.classes_)

        if 1 not in classes:
            raise ValueError("Anomaly class 1 not found")

        anomaly_index = classes.index(1)

        anomaly_probability = float(
            probabilities[anomaly_index]
        )

        threshold = 0.35

        prediction = (
            "Anomaly"
            if anomaly_probability >= threshold
            else "Normal"
        )

        return prediction, anomaly_probability


model_service = ModelService()
