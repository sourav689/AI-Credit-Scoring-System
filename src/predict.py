import joblib
import pandas as pd

# 1. Load the trained model
MODEL_PATH = '../models/credit_risk_model_v1.pkl'
model = joblib.load(MODEL_PATH)

def assess_credit_risk(customer_data):
    """
    Takes a dictionary of customer details and returns Risk/Safe.
    """
    # Convert input to DataFrame
    df_input = pd.DataFrame([customer_data])
    
    # Get Prediction
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0]
    
    status = "⚠️ RISK" if prediction == 1 else "✅ SAFE"
    confidence = probability[prediction] * 100
    
    return status, confidence

# --- TEST IT ---
# Example: A customer with high surplus but low income variance
test_customer = {
    'banked_flag': 1, 'job_category': 2, 'housing_type': 1, 'dependents': 1,
    'employment_years': 4, 'residence_years': 2, 'loan_amount': 3000,
    'duration_months': 12, 'interest_rate': 0.11, 'existing_monthly_obligation': 100,
    'average_monthly_inflow': 2500, 'average_monthly_outflow': 1200, 'emi': 200,
    'net_monthly_surplus': -500, 'emi_coverage_ratio': 6.5, 'loan_to_income_ratio': 0.12,
    'existing_obligation_ratio': 0.04, 'dependent_ratio': 0.5, 'employment_stability_score': 0.85,
    'income_variance_index': 0.02, 'residence_stability_score': 0.8, 'digital_behavior_score': 0.9
}

status, conf = assess_credit_risk(test_customer)
print(f"Final Assessment: {status} ({conf:.2f}% Confidence)")