# AI-Powered Credit Risk Intelligence System

An end-to-end Fintech solution designed to predict loan default risk with **97% accuracy**. The system utilizes an XGBoost classifier and features a real-time banking dashboard allowing officers to input customer financial data in **Indian Rupees (INR)** for instant credit decisions.

## Project Overview
This repository contains a full-stack machine learning application that bridges the gap between complex risk modeling and day-to-day banking operations. The core engine evaluates 22 distinct features ranging from debt-to-income ratios to behavioral stability metrics.

## Folder Structure
```text
credit-risk-intelligence/
├── app.py                  # Flask Backend & API Logic
├── requirements.txt        # Python Dependencies
├── .gitignore              # Version Control Exclusion File
├── models/
│   └── credit_risk_model_v1.pkl   # Pre-trained XGBoost Model
├── templates/
│   └── index.html          # Web Interface (HTML5/CSS3/JS)
└── README.md               # Project Documentation

Core Features
Predictive Modeling: Gradient Boosted Decision Trees (XGBoost) optimized for imbalanced financial datasets.

Automated Feature Engineering: Real-time conversion of raw currency inputs into derived analytical metrics such as EMI Coverage Ratio and Stability Scores.

Risk Assessment Dashboard: A responsive, dark-themed interface for streamlined data entry and decision visualization.

Full-Stack Integration: Robust communication between a JavaScript-driven frontend and a Python-based RESTful API.


Technical Stack
Machine Learning: Python, Scikit-Learn, XGBoost, Joblib

Data Processing: Pandas, NumPy

Backend Infrastructure: Flask (REST API)

Frontend Development: HTML5, CSS3 (Inter UI), JavaScript (Fetch API)

Installation and Deployment
Prerequisites
Python 3.8 or higher

Pip (Python Package Manager)

Setup Instructions
1.Clone the Repository: 
git clone [https://github.com/your-username/credit-risk-intelligence.git](https://github.com/your-username/credit-risk-intelligence.git)
cd credit-risk-intelligence

2.Install Dependencies:
pip install -r requirements.txt

3.Execute the Application
python app.py

4.Access the Dashboard
Open your browser and navigate to: http://127.0.0.1:5000

Model Performance
The model was evaluated using standard classification metrics to ensure reliability in a banking context:

Accuracy: 97%

Optimization: Focused on minimizing False Negatives to reduce the risk of Non-Performing Assets (NPAs).
