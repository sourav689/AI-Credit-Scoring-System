import streamlit as st
import joblib
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="Bank Credit Portal", layout="wide")

@st.cache_resource
def load_model():
    # Loads your 97% accuracy XGBoost model
    return joblib.load('credit_risk_model_v1.pkl')

model = load_model()

# --- FRONTEND UI ---
st.title("🏛️ Indian Bank: Credit Risk Assessment Portal")
st.subheader("Employee Data Entry Module (INR ₹)")

with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("Primary Financials")
        monthly_income = st.number_input("Monthly Salary (₹)", min_value=0, value=50000)
        monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=0, value=20000)
        existing_emi = st.number_input("Current Loan EMIs (₹)", min_value=0, value=0)

    with col2:
        st.info("Loan Details")
        loan_amount = st.number_input("Requested Loan (₹)", min_value=0, value=200000)
        duration = st.number_input("Duration (Months)", min_value=1, value=24)
        interest_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0) / 100

    with col3:
        st.info("Stability Factors")
        dependents = st.number_input("Number of Dependents", 0, 10, 2)
        emp_years = st.slider("Employment Years", 0, 40, 5)
        digital_score = st.slider("Digital Behavior Score", 0.0, 1.0, 0.85)

    submitted = st.form_submit_button("RUN CREDIT ANALYSIS")

# --- BACKEND LOGIC (FEATURE ENGINEERING) ---
if submitted:
    # 1. Calculate the 'Smart' features from raw Rupee inputs
    estimated_emi = (loan_amount * (1 + interest_rate)) / duration
    surplus = monthly_income - monthly_expenses - existing_emi
    emi_coverage = surplus / estimated_emi if estimated_emi > 0 else 0
    loan_to_income = loan_amount / (monthly_income * 12) if monthly_income > 0 else 0
    
    # 2. Build the exact 22-column DataFrame for the model
    # Note: Column order must match your training data!
    input_df = pd.DataFrame([{
        'banked_flag': 1, 'job_category': 2, 'housing_type': 1, 
        'dependents': dependents, 'employment_years': emp_years, 'residence_years': 3,
        'loan_amount': loan_amount, 'duration_months': duration, 'interest_rate': interest_rate,
        'existing_monthly_obligation': existing_emi, 'average_monthly_inflow': monthly_income,
        'average_monthly_outflow': monthly_expenses, 'emi': estimated_emi,
        'net_monthly_surplus': surplus, 'emi_coverage_ratio': emi_coverage,
        'loan_to_income_ratio': loan_to_income, 
        'existing_obligation_ratio': existing_emi / monthly_income if monthly_income > 0 else 0,
        'dependent_ratio': dependents / 5, 'employment_stability_score': 0.8,
        'income_variance_index': 0.05, 'residence_stability_score': 0.7,
        'digital_behavior_score': digital_score
    }])

    # 3. Model Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0].max()

    # --- DISPLAY RESULTS ---
    st.divider()
    if prediction == 0:
        st.success(f"### RESULT: ✅ SAFE (Confidence: {prob:.2%})")
        st.balloons()
    else:
        st.error(f"### RESULT: ⚠️ RISK (Confidence: {prob:.2%})")
    
    st.write(f"**Analysis:** Net surplus of ₹{surplus:,.2f} provides a coverage ratio of {emi_coverage:.2f}x.")