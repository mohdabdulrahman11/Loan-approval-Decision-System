from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
import os
import io

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'loan_model.pkl'))

NUMERIC_COLS = [
    'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents',
    'Credit_Score', 'Existing_Loans', 'DTI_Ratio', 'Savings',
    'Collateral_Value', 'Loan_Amount', 'Loan_Term'
]
CATEGORICAL_COLS = [
    'Gender', 'Education_Level', 'Employment_Status',
    'Marital_Status', 'Loan_Purpose', 'Property_Area', 'Employer_Category'
]

def engineer_features(df):
    df['DTI_Ratio_sq']    = df['DTI_Ratio'] ** 2
    df['Credit_Score_sq'] = df['Credit_Score'] ** 2
    return df

def get_decision(prob):
    if prob >= 0.65:
        return 'Approved', 'success'
    elif prob >= 0.35:
        return 'Manual Review', 'warning'
    else:
        return 'Rejected', 'danger'

def get_risk_factors(row):
    factors = []
    if float(row['Credit_Score']) < 600:
        factors.append('Low Credit Score')
    if float(row['DTI_Ratio']) > 0.5:
        factors.append('High Debt-to-Income Ratio')
    if float(row['Existing_Loans']) > 3:
        factors.append('Too Many Existing Loans')
    if float(row['Savings']) < 5000:
        factors.append('Insufficient Savings')
    if float(row['Loan_Amount']) > float(row['Collateral_Value']):
        factors.append('Loan Exceeds Collateral Value')
    return factors[:3]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    row = {col: float(form[col]) for col in NUMERIC_COLS}
    row.update({col: form[col] for col in CATEGORICAL_COLS})

    df = pd.DataFrame([row])
    df = engineer_features(df)

    all_cols = NUMERIC_COLS + ['DTI_Ratio_sq', 'Credit_Score_sq'] + CATEGORICAL_COLS
    prob = model.predict_proba(df[all_cols])[0][1]
    decision, style = get_decision(prob)
    risk_factors = get_risk_factors(row) if decision != 'Approved' else []

    result = {
        'decision': decision,
        'style': style,
        'confidence': round(prob * 100, 1),
        'risk_factors': risk_factors
    }
    return render_template('index.html', result=result, active_tab='single')

@app.route('/batch', methods=['POST'])
def batch():
    file = request.files.get('csvfile')
    if not file:
        return render_template('index.html', batch_error='No file uploaded.', active_tab='batch')
    try:
        df = pd.read_csv(file, encoding='latin1')

        # Drop unnecessary cols if present
        df = df.drop(columns=['Applicant_ID'], errors='ignore')
        df = df.dropna(subset=['Applicant_Income'])

        for col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = engineer_features(df)
        all_cols = NUMERIC_COLS + ['DTI_Ratio_sq', 'Credit_Score_sq'] + CATEGORICAL_COLS

        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            return render_template('index.html', batch_error=f'Missing columns: {missing}', active_tab='batch')

        probs = model.predict_proba(df[all_cols])[:, 1]
        decisions = [get_decision(p) for p in probs]

        df['Decision']      = [d[0] for d in decisions]
        df['Confidence_%']  = (probs * 100).round(1)

        output = df.to_csv(index=False)
        preview = df[['Applicant_Income', 'Credit_Score', 'Loan_Amount', 'Decision', 'Confidence_%']].head(20).to_dict('records')

        total    = len(df)
        approved = int((df['Decision'] == 'Approved').sum())
        rejected = int((df['Decision'] == 'Rejected').sum())
        review   = int((df['Decision'] == 'Manual Review').sum())

        return render_template('index.html',
            batch_results=preview, batch_csv=output,
            total=total, approved=approved, rejected=rejected, review=review,
            active_tab='batch'
        )
    except Exception as e:
        return render_template('index.html', batch_error=str(e), active_tab='batch')

@app.route('/download', methods=['POST'])
def download():
    csv_data = request.form['csv_data']
    return send_file(
        io.BytesIO(csv_data.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='loan_predictions.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)