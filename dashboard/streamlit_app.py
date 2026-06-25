import streamlit as st
import sys
import joblib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app.predictor import predict

st.set_page_config(
    page_title="SureVisit | No-show Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .stButton button {
        width: 100%;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏥 SureVisit</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Appointment No-show Risk Predictor</p>', unsafe_allow_html=True)

neighbourhood_freq = joblib.load(str(ROOT_DIR / "models" / "neighbourhood_freq.joblib"))
neighbourhood_list = sorted(neighbourhood_freq.keys())

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Patient & Appointment Details")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            gender = st.selectbox("Gender", ["F", "M"])
            age = st.number_input("Age", min_value=0, max_value=115, value=30)
            scheduled_day = st.date_input("Scheduled Day")
            appointment_day = st.date_input("Appointment Day")
            neighbourhood = st.selectbox("Neighbourhood", neighbourhood_list)
            sms_received = st.selectbox("SMS Reminder Sent?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        with c2:
            scholarship = st.selectbox("Scholarship", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            hipertension = st.selectbox("Hipertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            alcoholism = st.selectbox("Alcoholism", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            handcap = st.selectbox("Handicap", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            risk_history = st.number_input(
                "Number of previous missed appointments (leave 0 if unknown)",
                min_value=0, value=0
            )

        submitted = st.form_submit_button("Predict Risk", type="primary")

with right_col:
    st.subheader("Prediction Result")

    if submitted:
        request_data = {
            "gender": gender,
            "age": age,
            "scheduled_day": str(scheduled_day),
            "appointment_day": str(appointment_day),
            "neighbourhood": neighbourhood,
            "scholarship": scholarship,
            "hipertension": hipertension,
            "diabetes": diabetes,
            "alcoholism": alcoholism,
            "handcap": handcap,
            "sms_received": sms_received,
            "previous_attendance_rate": None,
            "risk_history": risk_history
        }

        try:
            result = predict(request_data)
            risk_level = result['risk_level']

            st.metric("No-show Probability", f"{result['no_show_probability'] * 100:.2f}%")

            if risk_level == "Low":
                st.success(f"**Risk Level: {risk_level}**")
            elif risk_level == "Medium":
                st.warning(f"**Risk Level: {risk_level}**")
            else:
                st.error(f"**Risk Level: {risk_level}**")

            st.markdown("**Recommended Action**")
            st.info(result['recommended_action'])

            st.caption("⚠️ This tool provides predictive risk estimates based on historical data and is intended for decision support only. It should not be used as a substitute for clinical or administrative judgment.")

            st.markdown("")

            utc_time = datetime.fromisoformat(result['prediction_timestamp'])

            readable_time = utc_time.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Kolkata")).strftime("%d %B %Y, %I:%M %p")
            st.caption(f"Prediction generated on {readable_time}")

            with st.expander("Prediction metadata"):
                st.json(result)

        except Exception as e:
            st.error(f"Error generating prediction: {e}")
    else:
        st.markdown(
            "<div style='padding: 3rem 1rem; text-align: center; color: #9ca3af; border: 1px dashed #d1d5db; border-radius: 8px;'>"
            "Fill the form and click <b>Predict Risk</b> to see results here."
            "</div>",
            unsafe_allow_html=True
        )

st.divider()
st.caption("SureVisit helps healthcare teams identify at-risk appointments early and take proactive action to reduce missed visits.")