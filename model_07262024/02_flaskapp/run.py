from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle

app = Flask(__name__)

# Load the model components from the pickle file
with open('baf_model_sm_0726_2.pkl', 'rb') as f:
    kmeans, model, threshold = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract and convert data from the form
    form_data = [
        request.form.get('income', type=float) or 0.0,
        request.form.get('name_email_similarity', type=float) or 0.0,
        int(request.form.get('living_length', 0)),
        int(request.form.get('age', 0)),
        request.form.get('days_since_submission', type=int) or 0,
        request.form.get('applications_zip', type=int) or 0,
        request.form.get('applications_bank_branch', type=int) or 0,
        request.form.get('distinct_emails', type=int) or 0,
        request.form.get('risk_score', type=int) or 0,
        request.form.get('credit_limit', type=int) or 0,
        request.form.get('session_length', type=int) or 0,
        request.form.get('emails_dob', type=int) or 0,
        request.form.get('month', type=int) or 0,
        int(request.form.get('email', 0)),  # Assuming 'email' is binary as well
        1 if request.form.get('home_phone') else 0,  # 1 if valid, 0 if not valid
        1 if request.form.get('other_cards') else 0,  # 1 if yes, 0 if no
        1 if request.form.get('location_check', '0') == '1' else 0,  # 1 if foreign request, 0 otherwise
        1 if request.form.get('sign_out', '0') == '0' else 0,  # 1 if keep session alive, 0 if not
        1 if request.form.get('previous_address') else 0,  # 1 if has previous address, 0 if not
        1 if request.form.get('payment_type') else 0,  # 1 if payment_type is AC, 0 if not
        1 if request.form.get('employment_status') else 0,  # 1 if employment_status is CA, 0 if not
        1 if request.form.get('housing_status') else 0,  # 1 if housing_status is BA, 0 if not
        1 if request.form.get('device_os') else 0,  # 1 if device_os is windows, 0 if not
    ]

    # Convert to numpy array and reshape for prediction
    input_data = np.array(form_data).reshape(1, -1)

    # Predict with the model
    prediction = model.predict(input_data)[0]
    fraud_proba = model.predict_proba(input_data)[0][1]

    result = 'Fraud' if fraud_proba > threshold else 'Not Fraud'

    return jsonify(result=result, probability=fraud_proba)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
