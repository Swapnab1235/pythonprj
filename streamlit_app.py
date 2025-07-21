import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("LogR_Model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit UI
st.title("Credit Card Default Prediction")
st.markdown("Enter the customer financial details to predict whether they will default on credit card payment.")

# Input form
with st.form("prediction_form"):
    LIMIT_BAL = st.number_input("Limit Balance", min_value=0.0)
    AGE = st.number_input("Age", min_value=18, max_value=100)
    PAY_0 = st.number_input("Repayment Status (Sept)")
    BILL_AMT1 = st.number_input("Bill Amount (Sept)")
    PAY_AMT1 = st.number_input("Payment Amount (Sept)")
    
    # You can add more inputs here as required for your model
    
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Replace below list with all features in the order expected by your model
        features = [LIMIT_BAL, AGE, PAY_0, BILL_AMT1, PAY_AMT1]
        final_features = np.array(features).reshape(1, -1)
        final_features_scaled = scaler.transform(final_features)
        prediction = model.predict(final_features_scaled)

        result = "🟥 Will Default" if prediction[0] == 1 else "🟩 Will Not Default"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
