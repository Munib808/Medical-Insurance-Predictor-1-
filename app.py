import streamlit as st
import numpy as np
import pickle

# Load trained stacking regressor and scaler
model = pickle.load(open("Insurance_stacking_reg.pkl", "rb"))
scaler = pickle.load(open("ScaleInsurance_stacking_reg.pkl", "rb"))

# App Title and Description
st.title("🏥 Medical Insurance Charges Prediction")
st.markdown("""
Welcome! This app predicts **medical insurance claim amount** based on personal information using a powerful machine learning model (stacking regressor).  
Fill in the details below to estimate your expected charges. 💰
""")

# User Inputs
age = st.slider("👤 Age", 18, 100, 30)
bmi = st.number_input("⚖️ BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("👶 Number of Children", min_value=0, max_value=10, step=1, value=0)
sex = st.selectbox("🚻 Gender", ["Male", "Female"])
smoker = st.selectbox("🚬 Smoker?", ["Yes", "No"])
region = st.selectbox("🌍 Region", ["Southeast", "Other"])

# Encode Categorical Inputs
sex_numeric = 1 if sex == "Male" else 0
smoker_numeric = 1 if smoker == "Yes" else 0
region_southeast = 1 if region == "Southeast" else 0

# Input Order Must Match Training
input_data = np.array([[age, bmi, children, sex_numeric, smoker_numeric, region_southeast]])

# Scale Input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("🔍 Predict Charges"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"💵 Estimated Insurance Charges: ₹{prediction:,.2f}")
