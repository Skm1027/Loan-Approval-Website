# File: predict.py
# This file contains the prediction logic for the loan approval model.
# It loads the trained model and uses it to predict loan approval based on user inputs.
import joblib
import numpy as np
import pandas as pd

def predict_loan(features, model_type='rf'):
    model = joblib.load('random_forest_model.pkl' if model_type == 'rf' else 'logistic_model.pkl')

    # Feature names used during training (must match model.py)
    feature_names = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]

    # Wrap features in a DataFrame
    X = pd.DataFrame([features], columns=feature_names)

    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][prediction]
    return prediction, confidence
