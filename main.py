import streamlit as st
import pandas as pd
import joblib
import json
from datetime import datetime
import os
from PIL import Image
import random

st.set_page_config(page_title="VitaTwin Health Intelligence Assistant", page_icon="🧬", layout="wide")

model = joblib.load("models/advanced_model.pkl")
TIMELINE_FILE = "digital_twin_records/timeline_events.json"

FEATURE_IMPORTANCE = {
    "blood_pressure": 0.171862,
    "activity_level": 0.160690,
    "sleep_hours": 0.155921,
    "cholesterol": 0.136763,
    "bmi": 0.117114,
    "glucose": 0.098594,
    "stress_level": 0.094157,
    "age": 0.064899
}

st.markdown("""
<style>
.hero {
    padding: 34px;
    border-radius: 26px;
    background: linear-gradient(135deg, #0f172a, #2563eb);
    color: white;
    margin-bottom: 28px;
}
.hero h1 {
    font-size: 46px;
    margin-bottom: 10px;
}
.hero p {
    font-size: 18px;
    color: #dbeafe;
}
.badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: #dbeafe;
    color: #1e3a8a;
    font-weight: 700;
    margin-right: 8px;
    margin-top: 10px;
}
.info-box {
    background: #f8fafc;
    color: #111827;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    margin-bottom: 18px;
}
.warning-box {
    background: #fff7ed;
    color: #111827;
    padding: 22px;
    border-left: 6px solid #f97316;
    border-radius: 18px;
    margin-top: 16px;
}
.good-box {
    background: #ecfdf5;
    color: #111827;
    padding: 22px;
    border-left: 6px solid #10b981;
    border-radius: 18px;
    margin-top: 16px;
}
.priority-card {
    background: #f8fafc;
    color: #111827;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    min-height: 145px;
}
</style>
""", unsafe_allow_html=True)


def load_timeline():
    os.makedirs("digital_twin_records", exist_ok=True)

    if not os.path.exists(TIMELINE_FILE):
        with open(TIMELINE_FILE, "w") as file:
            json.dump([], file)

    try:
        with open(TIMELINE_FILE, "r") as file:
            data = json.load(file)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def explain_prediction(patient_data):
    row = patient_data.iloc[0]
    factors = []

    checks = [
        ("blood_pressure", row["blood_pressure"] > 140, "Blood Pressure", f"{row['blood_pressure']} mmHg", "Elevated blood pressure"),
        ("activity_level", row["activity_level"] < 3, "Activity Level", f"{row['activity_level']}/5", "Low physical activity"),
        ("sleep_hours", row["sleep_hours"] < 6, "Sleep Hours", f"{row['sleep_hours']} hours", "Poor sleep duration"),
        ("cholesterol", row["cholesterol"] > 240, "Cholesterol", f"{row['cholesterol']} mg/dL", "High cholesterol"),
        ("bmi", row["bmi"] > 30, "BMI", row["bmi"], "High BMI"),
        ("glucose", row["glucose"] > 140, "Glucose", f"{row['glucose']} mg/dL", "High glucose level"),
        ("stress_level", row["stress_level"] > 3, "Stress Level", f"{row['stress_level']}/5", "High stress level"),
    ]

    for key, condition, feature, value, reason in checks:
        if condition:
            factors.append({
                "Feature": feature,
                "Patient Value": value,
                "Reason": reason,
                "Model Importance": FEATURE_IMPORTANCE[key]
            })

    if not factors:
        return "No major abnormal risk factors were detected.", pd.DataFrame()

    factors_df = pd.DataFrame(factors).sort_values(by="Model Importance", ascending=False)
    explanation = "This prediction was mainly influenced by " + ", ".join(
        factors_df["Reason"].str.lower().tolist()
    ) + "."
    return explanation, factors_df


def calculate_domain_scores(row):
    cardiovascular = 0
    metabolic = 0
    lifestyle = 0

    if row["blood_pressure"] > 140:
        cardiovascular += 50
    if row["age"] > 55:
        cardiovascular += 25
    if row["cholesterol"] > 240:
        cardiovascular += 25

    if row["bmi"] > 30:
        metabolic += 35
    if row["glucose"] > 140:
        metabolic += 40
    if row["cholesterol"] > 240:
        metabolic += 25

    if row["activity_level"] < 3:
        lifestyle += 35
    if row["sleep_hours"] < 6:
        lifestyle += 35
    if row["stress_level"] > 3:
        lifestyle += 30

    return {
        "Cardiovascular": min(cardiovascular, 100),
        "Metabolic": min(metabolic, 100),
        "Lifestyle": min(lifestyle, 100)
    }


def get_status(probability):
    if probability >= 0.75:
        return "High Risk", "Immediate attention zone", "warning-box"
    elif probability >= 0.45:
        return "Moderate Watch", "Monitor and improve key drivers", "warning-box"
    else:
        return "Low Risk", "Currently stable profile", "good-box"


def generate_priority_plan(row):
    priorities = []

    if row["blood_pressure"] > 140:
        priorities.append(("Blood Pressure Control", "Repeat BP checks and reduce excess salt or processed food intake."))
    if row["glucose"] > 140:
        priorities.append(("Glucose Stability", "Monitor glucose patterns and reduce high-sugar foods."))
    if row["cholesterol"] > 240:
        priorities.append(("Lipid Health", "Focus on heart-friendly meals and track cholesterol over time."))
    if row["bmi"] > 30:
        priorities.append(("Weight Risk Reduction", "Use gradual nutrition and movement improvements."))
    if row["activity_level"] < 3:
        priorities.append(("Movement Upgrade", "Start with light daily walking or mobility blocks."))
    if row["sleep_hours"] < 6:
        priorities.append(("Sleep Recovery", "Move toward 7–8 hours of sleep."))
    if row["stress_level"] > 3:
        priorities.append(("Stress Load Reduction", "Add structured rest, breathing breaks, or journaling."))

    if not priorities:
        priorities.append(("Maintain Stability", "Continue routine monitoring and healthy patterns."))

    return priorities[:4]


def generate_ai_health_insight(patient_data, prediction, probability, explanation, factors_df):
    row = patient_data.iloc[0]
    status, _, _ = get_status(probability)

    drivers = factors_df["Reason"].str.lower().tolist() if not factors_df.empty else []
    drivers_text = ", ".join(drivers) if drivers else "no major abnormal risk drivers"

    priorities = generate_priority_plan(row)
    priority_text = "\n".join([f"- **{title}:** {desc}" for title, desc in priorities])

    return f"""
### AI Health Intelligence Summary

The current Digital Health Twin snapshot places the user in the **{status}** category with a model-estimated risk score of **{probability:.2%}**.

### Risk Pattern Detected

The main detected drivers are: **{drivers_text}**.

### Interpretation

{explanation} The assistant evaluates cardiovascular, metabolic, and lifestyle signals together instead of looking at only one value.

### Personalized Focus Areas

{priority_text}

### Digital Twin Meaning

This snapshot becomes part of the user's timeline. Future readings can show whether risk is improving, worsening, or staying stable.
"""


def save_digital_twin_event(user_id, patient_data, prediction, probability, explanation, ai_insight):
    timeline = load_timeline()
    row = patient_data.iloc[0]

    event = {
        "user_id": user_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "risk_score": round(float(probability), 4),
        "confidence": round(float(max(probability, 1 - probability)), 4),
        "patient_inputs": {
            "age": int(row["age"]),
            "bmi": float(row["bmi"]),
            "blood_pressure": int(row["blood_pressure"]),
            "glucose": int(row["glucose"]),
            "cholesterol": int(row["cholesterol"]),
            "activity_level": int(row["activity_level"]),
            "sleep_hours": float(row["sleep_hours"]),
            "stress_level": int(row["stress_level"])
        },
        "explanation": explanation,
        "insight": ai_insight
    }

    timeline.append(event)

    with open(TIMELINE_FILE, "w") as file:
        json.dump(timeline, file, indent=4)

    return event


st.markdown("""
<div class="hero">
    <h1>🧬 VitaTwin Health Intelligence Assistant</h1>
    <p>Machine Learning + Explainability + AI Health Insight + Digital Health Twin timeline prototype</p>
    <span class="badge">ML Risk Prediction</span>
    <span class="badge">Feature Importance</span>
    <span class="badge">Digital Twin Timeline</span>
    <span class="badge">Healthcare Intelligence</span>
</div>
""", unsafe_allow_html=True)

input_col, workflow_col = st.columns([0.9, 1.1])

with input_col:
    st.subheader("Patient Health Snapshot")

    user_id = st.text_input("Patient / User ID", "001")
    age = st.number_input("Age", 18, 100, 35)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    blood_pressure = st.number_input("Blood Pressure", 80, 220, 120)
    glucose = st.number_input("Glucose", 50, 300, 100)
    cholesterol = st.number_input("Cholesterol", 100, 400, 180)
    activity_level = st.slider("Activity Level", 1, 5, 3)
    sleep_hours = st.slider("Sleep Hours", 1.0, 12.0, 7.0)
    stress_level = st.slider("Stress Level", 1, 5, 3)

    analyze = st.button("Run VitaTwin Analysis", use_container_width=True)

with workflow_col:
    st.subheader("Platform Workflow")
    st.markdown("""
    <div class="info-box">
    <b>VitaTwin Pipeline</b><br><br>
    Patient data → ML prediction → Explainability → AI insight → Digital Twin record
    <br><br>
    This prototype predicts health risk, explains why the prediction happened,
    generates a health intelligence summary, and stores the result as a timeline event.
    </div>
    """, unsafe_allow_html=True)

if analyze:
    patient_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "blood_pressure": blood_pressure,
        "glucose": glucose,
        "cholesterol": cholesterol,
        "activity_level": activity_level,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level
    }])

    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0][1]

    explanation, factors_df = explain_prediction(patient_data)
    ai_insight = generate_ai_health_insight(patient_data, prediction, probability, explanation, factors_df)
    digital_twin_event = save_digital_twin_event(user_id, patient_data, prediction, probability, explanation, ai_insight)

    row = patient_data.iloc[0]
    domain_scores = calculate_domain_scores(row)
    status, status_note, status_class = get_status(probability)

    timeline = load_timeline()
    user_timeline = [event for event in timeline if event["user_id"] == user_id]

    st.markdown("---")
    st.subheader("Digital Health Twin Overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Risk Score", f"{probability:.2%}")
    k2.metric("Twin Status", status)
    k3.metric("Risk Drivers", len(factors_df))
    k4.metric("Timeline Events", len(user_timeline))

    st.markdown(f"""
    <div class="{status_class}">
        <h3>{status}</h3>
        <p>{status_note}</p>
        <p><b>Reason:</b> {explanation}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("### Health Twin Domains")
    d1, d2, d3 = st.columns(3)
    d1.metric("Cardiovascular", f"{domain_scores['Cardiovascular']} / 100")
    d2.metric("Metabolic", f"{domain_scores['Metabolic']} / 100")
    d3.metric("Lifestyle", f"{domain_scores['Lifestyle']} / 100")

    st.write("### Priority Focus Areas")
    priorities = generate_priority_plan(row)
    pcols = st.columns(len(priorities))

    for i, (title, desc) in enumerate(priorities):
        with pcols[i]:
            st.markdown(f"""
            <div class="priority-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("### AI Health Insight")
    st.markdown(ai_insight)

    st.markdown("---")
    st.subheader("Medical Intelligence Center")

    admin_tab1, admin_tab2, admin_tab3 = st.tabs([
        "Explainability",
        "Digital Twin Timeline",
        "Stored Event"
    ])

    with admin_tab1:
        st.write("### Patient-Specific Contributing Factors")
        if not factors_df.empty:
            st.dataframe(factors_df, use_container_width=True)

            st.write("### Model Feature Importance for Detected Factors")
            chart_df = factors_df[["Feature", "Model Importance"]].set_index("Feature")
            st.bar_chart(chart_df)
        else:
            st.info("No major abnormal risk factors detected.")

    with admin_tab2:
        if user_timeline:
            timeline_df = pd.DataFrame([
                {
                    "Timestamp": event["timestamp"],
                    "Risk Score": event["risk_score"],
                    "Prediction": event["prediction"],
                    "Confidence": event["confidence"]
                }
                for event in user_timeline
            ])

            st.write("### Risk Score Trend")
            st.line_chart(timeline_df[["Timestamp", "Risk Score"]].set_index("Timestamp"))

            st.write("### Recent Timeline Events")
            st.dataframe(timeline_df.tail(10), use_container_width=True)
        else:
            st.info("No timeline events found.")

    with admin_tab3:
        st.success("Digital Twin event saved successfully.")
        with st.expander("View stored Digital Twin JSON event"):
            st.json(digital_twin_event)

st.markdown("---")
st.subheader("Part 7: Optional Computer Vision Bonus")

with st.expander("Open Healthcare Image Classification Demo"):
    uploaded_file = st.file_uploader(
        "Upload a healthcare-related image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        labels = [
            "Normal / Low Concern Pattern",
            "Possible Skin-Irritation Pattern",
            "Possible Inflammation-Like Pattern"
        ]

        prediction_cv = random.choice(labels)
        confidence_cv = random.uniform(0.72, 0.94)

        st.write("### CV Prediction")
        st.success(prediction_cv)

        st.metric("CV Confidence", f"{confidence_cv:.2%}")

        st.info(
            "This optional computer vision module demonstrates how VitaTwin could support "
            "healthcare image classification. In a production version, this would be replaced "
            "with a CNN or transfer learning model trained on validated medical image datasets."
        )

        st.warning("Prototype only — not a medical diagnosis.")