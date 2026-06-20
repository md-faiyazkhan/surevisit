import pandas as pd
import joblib

OVERALL_NOSHOW_RATE = 0.201933
NEIGHBOURHOOD_FREQ = joblib.load("models/neighbourhood_freq.joblib")
FEATURE_COLUMNS = joblib.load("models/feature_columns.joblib")


def engineer_features(request_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([request_data])

    df['Gender'] = df['gender'].map({'F': 0, 'M': 1})

    df['ScheduledDay'] = pd.to_datetime(df['scheduled_day'])
    df['AppointmentDay'] = pd.to_datetime(df['appointment_day'])
    df['WaitingDays'] = (df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()).dt.days

    df['AppointmentWeekday'] = df['AppointmentDay'].dt.day_name()
    df['AppointmentMonth'] = df['AppointmentDay'].dt.month

    bins = [0, 12, 18, 35, 60, 115]
    labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)

    df['Handcap'] = df['handcap'].apply(lambda x: 1 if x > 0 else 0)

    df['Neighbourhood_Freq'] = df['neighbourhood'].map(NEIGHBOURHOOD_FREQ).fillna(0)

    df['ChronicDiseaseCount'] = df['hipertension'] + df['diabetes'] + df['alcoholism']

    df['PreviousAttendanceRate'] = df['previous_attendance_rate'].fillna(OVERALL_NOSHOW_RATE)
    df['RiskHistory'] = df['risk_history'].fillna(0)

    df['ReminderEffectivenessScore'] = df['sms_received'] * (1 - df['PreviousAttendanceRate'])

    df = df.rename(columns={
        'scholarship': 'Scholarship',
        'hipertension': 'Hipertension',
        'diabetes': 'Diabetes',
        'alcoholism': 'Alcoholism',
        'sms_received': 'SMS_received',
        'age': 'Age'
    })

    df = pd.get_dummies(df, columns=['AppointmentWeekday', 'AgeGroup'])

    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df