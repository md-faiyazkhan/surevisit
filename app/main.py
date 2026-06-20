from fastapi import FastAPI, HTTPException
from app.schemas import AppointmentRequest, PredictionResponse
from app.predictor import predict

app = FastAPI(title="SureVisit API", version="1.0")


@app.get("/")
def health_check():
    return {"status": "healthy", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict_no_show(request: AppointmentRequest):
    try:
        result = predict(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))