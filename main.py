# main.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/salaries.csv")

# Drop rows with missing target (Salary) or any other null values
df.dropna(subset=['Salary'], inplace=True)
df.dropna(inplace=True)  # Optional: remove rows with any nulls


# Encode categorical columns
categorical_cols = ['Gender', 'Education Level', 'Job Title']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and Target
X = df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = df['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/salary_model.pkl")
joblib.dump(encoders, "models/label_encoders.pkl")

print("✅ Model and encoders saved to models/")
