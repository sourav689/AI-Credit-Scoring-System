from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the brain (Your 97% accuracy model)
try:
    model = joblib.load('models/credit_risk_model_v1.pkl')
except Exception as e:
    print(f"Error: Model file not found. Check 'models/' folder. {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # --- 1. CAPTURE ALL FRONTEND INPUTS ---
        # Financials
        income = float(data.get('income', 0))
        expenses = float(data.get('expenses', 0))
        loan_amt = float(data.get('loan_amt', 0))
        tenure = float(data.get('tenure', 12))
        existing_emi = float(data.get('existing_emi', 0))
        
        # Stability & Demographics
        age = float(data.get('age', 30))
        emp_years = float(data.get('emp_years', 2))
        res_years = float(data.get('res_years', 2))
        dependents = float(data.get('dependents', 0))
        
        # Categorical Inputs (Mapping Text to Numbers)
        banked = 1 if data.get('banked') == 'Yes' else 0
        digital_score = float(data.get('digital_score', 0.5))
        
        # --- 2. FEATURE ENGINEERING (Math & Logic) ---
        est_monthly_emi = (loan_amt * 1.15) / tenure
        surplus = income - expenses - existing_emi
        emi_coverage = surplus / est_monthly_emi if est_monthly_emi > 0 else 0
        
        # --- 3. FINAL 22-FEATURE DATASET ASSEMBLY ---
        features = pd.DataFrame([{
            'banked_flag': banked,
            'job_category': 1 if data.get('purpose') == 'Business' else 2,
            'housing_type': 1 if data.get('res_type') == 'Owned' else 2,
            'dependents': dependents,
            'employment_years': emp_years,
            'residence_years': res_years,
            'loan_amount': loan_amt,
            'duration_months': tenure,
            'interest_rate': 0.12,
            'existing_monthly_obligation': existing_emi,
            'average_monthly_inflow': income,
            'average_monthly_outflow': expenses,
            'emi': est_monthly_emi,
            'net_monthly_surplus': surplus,
            'emi_coverage_ratio': emi_coverage,
            'loan_to_income_ratio': loan_amt / (income * 12 + 1),
            'existing_obligation_ratio': existing_emi / (income + 1),
            'dependent_ratio': dependents / 5,
            'employment_stability_score': min(1.0, emp_years / 10),
            'income_variance_index': 0.05,
            'residence_stability_score': min(1.0, res_years / 10),
            'digital_behavior_score': digital_score
        }])

        # --- 4. PREDICTION ---
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0].max()
        
        return jsonify({
            "status": "APPROVED ✅" if prediction == 0 else "REJECTED ❌",
            "confidence": f"{prob:.2%}",
            "surplus": f"₹{surplus:,.0f}",
            "score_impact": "Positive" if digital_score > 0.7 else "Neutral/Negative"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)