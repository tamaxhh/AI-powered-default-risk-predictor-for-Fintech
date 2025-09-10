import sys
import os

# go one level up (from Notebook/ to project root)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

#%%writefile ../dashboard/streamlit_app.py

import streamlit as st
import pandas as pd
import shap
from joblib import load
from src.config import MODEL_DIR, PROCESSED_DATA_PATH
from src.utils import log_info

# Load model and data
xgb_model = load(os.path.join(MODEL_DIR, 'xgb_model.sav'))
data = pd.read_csv(PROCESSED_DATA_PATH.replace('.csv', '_engineered.csv'))
features = data.drop('Loan_Status', axis=1).columns

st.title('Loan Default Risk Predictor')

# Input form
st.header('Enter Applicant Details')
input_data = {}
for feature in features:
    if feature in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        options = sorted(data[feature].unique())
        input_data[feature] = st.selectbox(feature, options)
    else:
        input_data[feature] = st.number_input(feature, min_value=0.0, value=float(data[feature].mean()))

# Predict
if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    pred = xgb_model.predict(input_df)[0]
    prob = xgb_model.predict_proba(input_df)[0][1]
    st.header('Prediction')
    st.write(f'Loan Default Risk: {"High" if pred == 1 else "Low"}')
    st.write(f'Probability of Default: {prob:.2%}')

    # SHAP explanation
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_df)
    st.header('SHAP Explanation')
    shap.initjs()
    st.pyplot(shap.force_plot(explainer.expected_value, shap_values[0], input_df, matplotlib=True))

# Data insights
st.header('Data Insights')
st.write('Default Rate by Income Level')
data['Income_Bin'] = pd.qcut(data['ApplicantIncome'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
default_by_income = data.groupby('Income_Bin')['Loan_Status'].mean()
st.bar_chart(default_by_income)
log_info('Streamlit dashboard prototyped')