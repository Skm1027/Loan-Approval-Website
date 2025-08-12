import streamlit as st
from predict import predict_loan

st.title("🏦 Loan Approval Prediction App")

st.markdown("Enter the applicant details to predict loan approval:")

# Streamlit form inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Term (in months)", [360, 180, 240, 120])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Categorical encoding (same format as in model training)
def encode_inputs():
    return [
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents],
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
    ]

# Model selection
model_choice = st.radio("Select Model", ["Random Forest", "Logistic Regression"])
model_code = 'rf' if model_choice == "Random Forest" else 'lr'

# Predict button
if st.button("Predict Loan Approval"):
    features = encode_inputs()
    result, confidence = predict_loan(features, model_type=model_code)
    status = "✅ Approved" if result == 1 else "❌ Not Approved"
    st.success(f"Loan Status: {status}")
    st.info(f"Confidence Score: {confidence:.2f}")
# This code is the main Streamlit app for loan approval prediction.
# It allows users to input applicant details and get predictions using either a Random Forest or Logistic Regression model.
# The inputs are encoded in the same way as during model training, and the prediction results are displayed with a confidence score.
# The app uses the `predict_loan` function from the `predict` module to perform the prediction.
# The user can select the model type and view the prediction results interactively.
# The app is designed to be user-friendly and provides clear feedback on the loan approval status and confidence level.
# The app is built using Streamlit, a popular framework for