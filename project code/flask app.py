```python
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('payments.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        features = [
            float(request.form['step']),
            float(request.form['type']),
            float(request.form['amount']),
            float(request.form['oldbalanceOrg']),
            float(request.form['newbalanceOrig']),
            float(request.form['oldbalanceDest']),
            float(request.form['newbalanceDest'])
        ]

        features_array = np.array([features])
        fraud_prob = model.predict_proba(features_array)[0][1]

        if fraud_prob > 0.20:
            result = f"⚠️ Fraud Detected (Probability: {fraud_prob:.2f})"
        else:
            result = f"✅ Legitimate Transaction (Fraud Probability: {fraud_prob:.2f})"

        return render_template('submit.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
