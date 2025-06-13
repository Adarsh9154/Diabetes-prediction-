import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load and prepare data
@st.cache_data
def load_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns="Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = svm.SVC(kernel='linear')
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = load_model()

# Title and description
st.title("🩺 Diabetes Prediction App")
st.write("Enter your medical details below to predict if you're diabetic.")

# Input form
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Prediction logic
if st.button("Predict Diabetes"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)

    if result[0] == 0:
        st.success("✅ You are **not diabetic**.")
    else:
        st.error("⚠️ You are **diabetic**. Please consult a doctor.")

