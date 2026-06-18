import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("data/vitatwin_health_dataset.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

X = df.drop("health_risk", axis=1)
y = df["health_risk"]

print("Input features shape:", X.shape)
print("Target shape:", y.shape)
print("Target distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

baseline_model = LogisticRegression(max_iter=1000)

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)
baseline_probabilities = baseline_model.predict_proba(X_test)[:, 1]

print("\nBaseline Model: Logistic Regression")
print("Accuracy:", accuracy_score(y_test, baseline_predictions))
print("Precision:", precision_score(y_test, baseline_predictions))
print("Recall:", recall_score(y_test, baseline_predictions))
print("F1 Score:", f1_score(y_test, baseline_predictions))
print("ROC-AUC:", roc_auc_score(y_test, baseline_probabilities))

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

print("\nAdvanced Model: Random Forest")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Precision:", precision_score(y_test, rf_predictions))
print("Recall:", recall_score(y_test, rf_predictions))
print("F1 Score:", f1_score(y_test, rf_predictions))
print("ROC-AUC:", roc_auc_score(y_test, rf_probabilities))

feature_importance = rf_model.feature_importances_

features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance_df)

joblib.dump(baseline_model, "models/baseline_model.pkl")
joblib.dump(rf_model, "models/advanced_model.pkl")

print("\nModels saved successfully in the models folder!")