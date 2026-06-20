def test_predict_valid_request(client):
    payload = {
        "gender": "F",
        "age": 45,
        "scheduled_day": "2024-05-10",
        "appointment_day": "2024-05-15",
        "neighbourhood": "JARDIM DA PENHA",
        "scholarship": 0,
        "hipertension": 1,
        "diabetes": 0,
        "alcoholism": 0,
        "handcap": 0,
        "sms_received": 1,
        "previous_attendance_rate": None,
        "risk_history": None
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "no_show_probability" in data
    assert "risk_level" in data
    assert "recommended_action" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_high_risk_profile(client):
    payload = {
        "gender": "M",
        "age": 25,
        "scheduled_day": "2024-04-01",
        "appointment_day": "2024-06-15",
        "neighbourhood": "ILHA DO BOI",
        "scholarship": 0,
        "hipertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handcap": 0,
        "sms_received": 0,
        "previous_attendance_rate": 0.9,
        "risk_history": 8
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["risk_level"] == "High"


def test_predict_invalid_age(client):
    payload = {
        "gender": "F",
        "age": -5,
        "scheduled_day": "2024-05-10",
        "appointment_day": "2024-05-15",
        "neighbourhood": "JARDIM DA PENHA",
        "scholarship": 0,
        "hipertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handcap": 0,
        "sms_received": 1,
        "previous_attendance_rate": None,
        "risk_history": None
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field(client):
    payload = {
        "age": 45
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422