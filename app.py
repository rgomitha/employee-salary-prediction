# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("models/salary_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("💼 Employee Salary Prediction")

st.markdown("This app predicts employee salary based on inputs like age, gender, education, job title, and experience.")

# User inputs
age = st.slider("Age", 18, 65, 25)
gender = st.selectbox("Gender", encoders['Gender'].classes_)
education = st.selectbox("Education Level", encoders['Education Level'].classes_)
job_title = st.selectbox("Job Title", encoders['Job Title'].classes_)
experience = st.slider("Years of Experience", 0, 40, 1)

# Encode categorical inputs
gender_enc = encoders['Gender'].transform([gender])[0]
education_enc = encoders['Education Level'].transform([education])[0]
job_enc = encoders['Job Title'].transform([job_title])[0]

# Prepare input DataFrame
input_df = pd.DataFrame([{
    'Age': age,
    'Gender': gender_enc,
    'Education Level': education_enc,
    'Job Title': job_enc,
    'Years of Experience': experience
}])

# Predict salary
if st.button("Predict Salary 💰"):
    salary = model.predict(input_df)[0]
    st.success(f"Estimated Salary: ₹{int(salary):,}")
