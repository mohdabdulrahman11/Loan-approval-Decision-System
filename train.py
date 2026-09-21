import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score

data = pd.read_csv('loan_approval_data.csv')
print(f'Loaded. Shape: {data.shape}')

# --- Clean
data = data.drop(columns=['Applicant_ID'])
data = data.dropna(subset=['Loan_Approved'])
print(f"After dropping nulls: {data.shape}")

# --- Feature Engineering
data['DTI_Ratio_sq']    = data['DTI_Ratio'] ** 2
data['Credit_Score_sq'] = data['Credit_Score'] ** 2

# --- Target
data['Loan_Approved'] = data['Loan_Approved'].map({'Yes': 1, 'No': 0})
print(f"Class distribution: {Counter(data['Loan_Approved'])}")

# --- Features
numeric_features = [
    'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents',
    'Credit_Score', 'Existing_Loans', 'DTI_Ratio', 'Savings',
    'Collateral_Value', 'Loan_Amount', 'Loan_Term',
    'DTI_Ratio_sq', 'Credit_Score_sq'
]
categorical_features = [
    'Gender', 'Education_Level', 'Employment_Status',
    'Marital_Status', 'Loan_Purpose', 'Property_Area', 'Employer_Category'
]

X = data[numeric_features + categorical_features]
y = data['Loan_Approved']

# --- Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# --- Train
clf_pipeline.fit(X_train, y_train)
print("✅ Model trained!")

# --- Evaluate
y_pred = clf_pipeline.predict(X_test)
y_prob = clf_pipeline.predict_proba(X_test)[:, 1]
print(confusion_matrix(y_test, y_pred))
print(f"Precision: {precision_score(y_test, y_pred):.4f} | Recall: {recall_score(y_test, y_pred):.4f} | ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred))

# --- Save
joblib.dump(clf_pipeline, 'loan_model.pkl')
print("✅ Saved as loan_model.pkl")