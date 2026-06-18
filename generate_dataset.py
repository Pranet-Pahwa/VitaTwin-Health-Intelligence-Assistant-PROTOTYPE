import pandas as pd
import numpy as np

np.random.seed(42)

records = 700

data = {
    "age": np.random.randint(18, 80, records),
    "bmi": np.random.normal(27, 5, records).clip(18, 40).round(1),
    "blood_pressure": np.random.normal(130, 20, records).clip(90, 200).astype(int),
    "glucose": np.random.normal(115, 35, records).clip(70, 300).astype(int),
    "cholesterol": np.random.normal(210, 40, records).clip(120, 350).astype(int),
    "activity_level": np.random.randint(1, 6, records),
    "sleep_hours": np.random.uniform(4, 9, records).round(1),
    "stress_level": np.random.randint(1, 6, records)
}

df = pd.DataFrame(data)

risk_score = (
    (df["bmi"] > 30).astype(int)
    + (df["blood_pressure"] > 140).astype(int)
    + (df["glucose"] > 140).astype(int)
    + (df["cholesterol"] > 240).astype(int)
    + (df["activity_level"] < 3).astype(int)
    + (df["sleep_hours"] < 6).astype(int)
    + (df["stress_level"] > 3).astype(int)
)


df["health_risk"] = (risk_score >= 3).astype(int)
df.to_csv("data/vitatwin_health_dataset.csv", index=False)

print("Dataset saved successfully!")
print(df.head())
