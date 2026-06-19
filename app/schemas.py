from pydantic import BaseModel, Field
from typing import Optional


class AppointmentRequest(BaseModel):
    gender: str = Field(..., description="Patient gender: 'F' or 'M'")
    age: int = Field(..., ge=0, le=115, description="Patient age")
    scheduled_day: str = Field(..., description="Date appointment was scheduled, format: YYYY-MM-DD")
    appointment_day: str = Field(..., description="Date of the appointment, format: YYYY-MM-DD")
    neighbourhood: str = Field(..., description="Neighbourhood name")
    scholarship: int = Field(..., ge=0, le=1)
    hipertension: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    alcoholism: int = Field(..., ge=0, le=1)
    handcap: int = Field(..., ge=0, description="0 = no disability, 1+ = has disability")
    sms_received: int = Field(..., ge=0, le=1)
    previous_attendance_rate: Optional[float] = Field(None, ge=0, le=1, description="Patient's historical no-show rate, if known")
    risk_history: Optional[int] = Field(None, ge=0, description="Count of past no-shows, if known")


class PredictionResponse(BaseModel):
    no_show_probability: float
    risk_level: str
    recommended_action: str
    prediction_timestamp: str