import streamlit as st
import pandas as pd
from predict import predict_loan
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Apply custom CSS for styling
st.markdown("""
<style>
    /* Center the header and add some spacing */
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #34495e;
        margin-bottom: 2rem;
    }
    
    /* Style for the prediction result */
    .result {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Style for the form container */
    .form-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar custom styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("<div class='main-header'>Loan Approval Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Enter the details in the sidebar to predict loan approval</div>", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("Applicant Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=500)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, step=500)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=100)
loan_term = st.sidebar.selectbox("Loan Term (in months)", [360, 180, 240, 120])
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

model_choice = st.sidebar.radio("Select Model", ["Random Forest", "Logistic Regression"])
model_code = 'rf' if model_choice == "Random Forest" else 'lr'

# Define a container for form inputs (optional)
with st.container():
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    st.write("Use the sidebar to input applicant details and then click the **Predict Loan Approval** button.")
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction button in the main section
if st.button("Predict Loan Approval"):
    # Prepare features (the order must match what the model expects)
    features = [
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

    # Get prediction and confidence using predict.py
    result, confidence = predict_loan(features, model_type=model_code)
    status = "✅ Approved" if result == 1 else "❌ Not Approved"
    
    # Display the result using a styled container
    st.markdown(f"<div class='result'>Loan Status: {status}<br>Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)
