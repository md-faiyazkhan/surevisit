import joblib
from datetime import datetime
from app.features import engineer_features

MODEL = joblib.load("models/final_pipeline.joblib")

RISK_THRESHOLDS = {
    "low_max": 0.30,
    "medium_max": 0.60
}

RECOMMENDATIONS = {
    "Low": "Standard workflow, no action needed",
    "Medium": "Send additional SMS reminder",
    "High": "Schedule manual confirmation call"
}


def get_risk_level(probability: float) -> str:
    if probability <= RISK_THRESHOLDS["low_max"]:
        return "Low"
    elif probability <= RISK_THRESHOLDS["medium_max"]:
        return "Medium"
    else:
        return "High"


def predict(request_data: dict) -> dict:
    features_df = engineer_features(request_data)
    probability = MODEL.predict_proba(features_df)[0][1]
    risk_level = get_risk_level(probability)
    recommended_action = RECOMMENDATIONS[risk_level]

    return {
        "no_show_probability": round(float(probability), 4),
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "prediction_timestamp": datetime.utcnow().isoformat()
    }