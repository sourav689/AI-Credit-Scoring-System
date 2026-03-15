from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('models/credit_risk_model_v1.pkl')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error: Model file not found. {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # --- RAW INPUTS ---
        age           = float(data.get('age', 25))
        income        = float(data.get('income', 0))
        expenses      = float(data.get('expenses', 0))
        loan_amt      = float(data.get('loan_amt', 0))
        tenure        = float(data.get('tenure', 12))
        existing_emi  = float(data.get('existing_emi', 0))
        emp_years     = float(data.get('emp_years', 2))
        res_years     = float(data.get('res_years', 2))
        dependents    = float(data.get('dependents', 0))
        banked        = 1 if data.get('banked') == 'Yes' else 0
        digital_score = float(data.get('digital_score', 0.5))
        location      = 1 if data.get('location') == 'Urban' else 0
        digital_risk  = float(data.get('digital_risk', 0.05))
        purpose       = 1 if data.get('purpose') == 'Business' else 0  # 1=Business, 0=Personal
        housing_raw   = data.get('res_type', 'Rented')
        job_cat       = int(data.get('job_category', 2))

        # Housing encoding: own=0, rent=1, family=2 (matches LabelEncoder alphabetical order)
        housing_map = {'Family': 2, 'Owned': 0, 'Rented': 1}
        housing_encoded = housing_map.get(housing_raw, 1)

        # --- DERIVED FEATURES ---
        r             = 0.12 / 12  # monthly interest rate
        est_emi       = loan_amt * r * (1 + r)**tenure / ((1 + r)**tenure - 1) if tenure > 0 else 0
        surplus       = income - expenses - existing_emi - est_emi
        usable        = income - expenses - existing_emi
        emi_coverage  = usable / est_emi if est_emi > 0 else 0
        lti           = loan_amt / (income + 1)
        eor           = existing_emi / (income + 1)
        dep_ratio     = min(1.0, dependents / max(age - 17, 1))
        emp_stab      = min(1.0, emp_years / 10)
        res_stab      = min(1.0, res_years / 10)

        # --- FEATURE DATAFRAME (must match exact training column order) ---
        features = pd.DataFrame([{
            'age':                        age,
            'banked_flag':                banked,
            'job_category':               job_cat,
            'housing_type':               housing_encoded,
            'dependents':                 dependents,
            'employment_years':           emp_years,
            'residence_years':            res_years,
            'loan_amount':                loan_amt,
            'duration_months':            tenure,
            'interest_rate':              0.12,
            'existing_monthly_obligation': existing_emi,
            'average_monthly_inflow':     income,
            'average_monthly_outflow':    expenses,
            'emi':                        round(est_emi, 2),
            'net_monthly_surplus':        round(surplus, 2),
            'emi_coverage_ratio':         round(emi_coverage, 4),
            'loan_to_income_ratio':       round(lti, 4),
            'existing_obligation_ratio':  round(eor, 4),
            'dependent_ratio':            round(dep_ratio, 4),
            'employment_stability_score': round(emp_stab, 2),
            'income_variance_index':      0.05,
            'residence_stability_score':  round(res_stab, 2),
            'digital_behavior_score':     digital_score,
            'location_type':              location,
            'digital_risk_score':         digital_risk,
            'purpose':                    purpose,
        }])

        prediction = model.predict(features)[0]
        proba      = model.predict_proba(features)[0]
        confidence = proba.max()
        risk_prob  = proba[1]

        # Risk tier based on probability
        if risk_prob < 0.35:
            tier = "Low Risk 🟢"
        elif risk_prob < 0.65:
            tier = "Medium Risk 🟡"
        else:
            tier = "High Risk 🔴"

        return jsonify({
            "status":      "APPROVED ✅" if prediction == 0 else "REJECTED ❌",
            "confidence":  f"{confidence:.2%}",
            "risk_tier":   tier,
            "risk_prob":   f"{risk_prob:.2%}",
            "surplus":     f"₹{surplus:,.0f}",
            "emi":         f"₹{est_emi:,.0f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)