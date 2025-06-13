import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import joblib

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes.csv')
    X = data.drop(columns='Outcome', axis=1)
    Y = data['Outcome']
    return X, Y

X, Y = load_data()

# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Model Training
model = svm.SVC(kernel='linear')
model.fit(X_scaled, Y)

# Streamlit UI
st.title("Diabetes Prediction Web App")

st.header("Enter the following values:")

# Input fields for all 8 features
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=846, value=80)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
Age = st.number_input("Age", min_value=1, max_value=120, value=33)

input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                        BMI, DiabetesPedigreeFunction, Age]])
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 0:
        st.success("The person is **not diabetic**.")
    else:
        st.error("The person is **diabetic**.")

