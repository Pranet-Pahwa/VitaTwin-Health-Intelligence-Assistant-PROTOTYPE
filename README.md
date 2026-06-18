# VitaTwin Health Intelligence Assistant

## Overview

VitaTwin Health Intelligence Assistant is a healthcare AI prototype that combines Machine Learning, Explainable AI, Health Intelligence, and Digital Health Twin concepts to analyze patient health data and generate meaningful health insights.

The system predicts an individual's health risk category, explains the factors contributing to the prediction, generates human-readable health intelligence summaries, and stores results as Digital Health Twin timeline events for longitudinal health monitoring.

---

## Project Objectives

The project was developed as part of the VitaTwin Human Digital Health Twin Platform technical evaluation.

Key goals:

* Predict patient health risk using machine learning.
* Explain why predictions were made.
* Generate personalized health insights.
* Create Digital Health Twin records.
* Visualize risk trends and health intelligence through an interactive dashboard.

---

## System Architecture

Patient Health Data
↓
Machine Learning Models
↓
Risk Prediction
↓
Explainability Engine
↓
AI Health Insight Generator
↓
Digital Health Twin Timeline
↓
Interactive Dashboard

---

## Dataset

A synthetic healthcare dataset containing 700 patient records was created.

Features:

* Age
* BMI
* Blood Pressure
* Glucose
* Cholesterol
* Activity Level
* Sleep Hours
* Stress Level

Target Variable:

* Health Risk Category

  * 0 = Low Risk
  * 1 = High Risk

The dataset was designed to simulate realistic healthcare risk patterns.

---

## Machine Learning Models

### Baseline Model

Logistic Regression

Purpose:

* Establish a benchmark model for risk prediction.

Results:

* Accuracy: 73.57%
* Precision: 70.37%
* Recall: 64.41%
* F1 Score: 67.25%
* ROC-AUC: 83.70%

---

### Advanced Model

Random Forest Classifier

Purpose:

* Improve predictive performance and capture non-linear health risk relationships.

Results:

* Accuracy: 90.71%
* Precision: 88.33%
* Recall: 89.83%
* F1 Score: 89.08%
* ROC-AUC: 97.10%

Random Forest significantly outperformed Logistic Regression and was selected as the final prediction model.

---

## Explainability Module

Feature Importance was used to explain model decisions.

The system identifies the most influential health factors contributing to each prediction.

Example:

* Elevated Blood Pressure
* High BMI
* High Glucose
* Low Activity Level
* Poor Sleep

This improves transparency and trust in AI-generated healthcare predictions.

---

## AI Health Insight Generator

A prompt-based healthcare intelligence engine generates personalized health summaries.

The generated insight includes:

* Risk category
* Major risk drivers
* Health interpretation
* Personalized focus areas
* Digital Twin context

Example:

"The user shows elevated risk due to high blood pressure, low physical activity, and increased stress levels."

---

## Digital Health Twin Integration

Each analysis is stored as a Digital Health Twin timeline event.

Stored Information:

* User ID
* Timestamp
* Risk Score
* Prediction
* Confidence
* Patient Inputs
* Explainability Output
* AI Insight

Example Record:

```json
{
  "user_id": "001",
  "risk_score": 0.84,
  "prediction": "High Risk",
  "timestamp": "2026-06-17 20:38:02"
}
```

This allows longitudinal monitoring of health status over time.

---

## Dashboard Features

Built using Streamlit.

### Patient View

* Health Risk Score
* Digital Twin Status
* Health Domain Analysis
* Priority Focus Areas
* AI Health Intelligence Summary

### Clinical / Admin View

* Explainability Dashboard
* Feature Importance Visualization
* Digital Twin Timeline
* Stored Health Events

---

## Results

The system successfully demonstrates:

* Machine Learning-based risk prediction
* Explainable AI
* Personalized health intelligence
* Digital Health Twin concepts
* Healthcare dashboard visualization

The final model achieved over 90% accuracy on the generated healthcare dataset.

---

## Screenshots

Screenshots are available in the `/screenshots` directory:

* Dashboard Home
* Prediction Results
* AI Health Insight
* Explainability Dashboard
* Digital Health Twin Timeline

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit
* JSON
* Pillow

---

## Future Improvements

* SHAP-based explainability
* LLM-powered health insight generation (Ollama / Llama)
* Real healthcare datasets
* Multi-user Digital Twin profiles
* Transfer Learning based medical image classification
* Real-time wearable device integration

---

## Author

Pranet Pahwa

VitaTwin Health Intelligence Assistant
AI + Healthcare + Digital Twin Prototype
