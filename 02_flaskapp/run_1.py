from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle

app = Flask(__name__)

# Load the model components from the pickle file
with open('model_pipeline.pkl', 'rb') as f:
        kmeans, model, threshold = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

        # Extracting data from the form
    form_data = [
            int(request.form['email']),
            request.form.get('home_phone', type=float),
            request.form.get('mobile_phone', type=float),
            request.form.get('other_cards', type=float),
            int(request.form['location_check']),
            int(request.form['sign_out']),
            request.form['previous_address'],
            request.form.get('income', type=float),
            request.form.get('name_email_similarity', type=float),
            int(request.form['living_length']),
            int(request.form['age']),
            request.form.get('days_since_submission', type=int),
            request.form.get('applications_zip', type=int),
            request.form.get('applications_6_hours', type=int),
            request.form.get('applications_24_hours', type=int),
            request.form.get('applications_bank_branch', type=int),
            request.form.get('emails_dob', type=int),
            request.form.get('risk_score', type=int),
            request.form.get('credit_limit', type=int),
            request.form.get('session_length', type=int),
            request.form.get('distinct_emails', type=int),
            request.form.get('month', type=int),
            request.form['payment_type'],
            request.form['employment_status'],
            request.form['housing_status'],
            int(request.form['internet']),
            request.form['device_os']]

    # Prepare the input for the model
    input_data = []
    input_data.extend(form_data[:21])
    input_data.extend([
            1 if request.form['payment_type'] == 'AA' else 0,
            1 if request.form['payment_type'] == 'AB' else 0,
            1 if request.form['payment_type'] == 'AC' else 0,
            1 if request.form['payment_type'] == 'AD' else 0,
            1 if request.form['payment_type'] == 'AE' else 0,
            1 if request.form['employment_status'] == 'CA' else 0,
            1 if request.form['employment_status'] == 'CB' else 0,
            1 if request.form['employment_status'] == 'CC' else 0,
            1 if request.form['employment_status'] == 'CD' else 0,
            1 if request.form['employment_status'] == 'CE' else 0,
            1 if request.form['employment_status'] == 'CF' else 0,
            1 if request.form['employment_status'] == 'CG' else 0,
            1 if request.form['housing_status'] == 'BA' else 0,
            1 if request.form['housing_status'] == 'BB' else 0,
            1 if request.form['housing_status'] == 'BC' else 0,
            1 if request.form['housing_status'] == 'BD' else 0,
            1 if request.form['housing_status'] == 'BE' else 0,
            1 if request.form['housing_status'] == 'BF' else 0,
            1 if request.form['housing_status'] == 'BG' else 0,
            1 if request.form['device_os'] == 'Linux' else 0,
            1 if request.form['device_os'] == 'Macintosh' else 0,
            1 if request.form['device_os'] == 'Other' else 0,
            1 if request.form['device_os'] == 'Windows' else 0,
            1 if request.form['device_os'] == 'X11' else 0,])

    # Convert to numpy array and reshape for prediction
    input_data = np.array(input_data).reshape(1, -1)
        
    # Since there's no transformer, you can directly predict with the model
    prediction = model.predict(input_data)[0]
    fraud_proba = model.predict_proba(input_data)[0][1]
        
    result = 'Fraud' if fraud_proba > threshold else 'Not Fraud'

    return jsonify(result=result, probability=fraud_proba)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
