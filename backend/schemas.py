
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    request_id: str

    RSRP: float
    RSRQ: float
    SINR: float
    Latency: float
    Packet_Drop_Rate: float
    CPU_Utilization: float
    Throughput: float
    Signal_Quality: float
    High_Latency: float
    High_Packet_Drop: float


class PredictionResponse(BaseModel):
    request_id: str
    prediction: str
    anomaly_probability: float
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
