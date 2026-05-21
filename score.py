import os
import json
import pickle
import pandas as pd
import numpy as np

# Global variables for loaded artifacts
model = None
label_encoders = None
threshold = None
feature_columns = None
medians = None

def init():
    """
    Initializes the model and loading artifacts.
    This function is run once when the endpoint is deployed.
    """
    global model, label_encoders, threshold, feature_columns, medians
    
    # Azure ML sets AZUREML_MODEL_DIR environment variable pointing to the model directory
    model_dir = os.getenv('AZUREML_MODEL_DIR', '.')
    
    # Try finding files inside an 'artifacts' directory first, then fallback to model_dir
    artifacts_dir = os.path.join(model_dir, 'artifacts')
    if not os.path.exists(artifacts_dir):
        artifacts_dir = model_dir
        
    print(f"Loading model artifacts from: {artifacts_dir}")
    
    # Load serialized objects
    with open(os.path.join(artifacts_dir, 'lgbm_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(artifacts_dir, 'label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
        
    with open(os.path.join(artifacts_dir, 'threshold.pkl'), 'rb') as f:
        threshold = pickle.load(f)
        
    with open(os.path.join(artifacts_dir, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
        
    # Load medians for robust imputation, with fallbacks to training statistics
    medians_path = os.path.join(artifacts_dir, 'medians.pkl')
    if os.path.exists(medians_path):
        with open(medians_path, 'rb') as f:
            medians = pickle.load(f)
    else:
        medians = {
            'loan_amnt': 12800.0,
            'term': 36.0,
            'int_rate': 12.62,
            'grade': 3.0,
            'emp_length': 6.0,
            'annual_inc': 65000.0,
            'dti': 17.82,
            'delinq_2yrs': 0.0,
            'inq_last_6mths': 0.0,
            'revol_util': 50.3,
            'pub_rec': 0.0,
            'income_to_loan_ratio': 4.9995833680526625,
            'interest_burden': 1558.8
        }
        
    print("Inference service initialized successfully.")

def assign_risk(prob):
    """
    Categorizes the borrower into Low, Medium, or High risk tier based on probability of default.
    """
    if prob < 0.3:
        return 'Low'
    elif prob < 0.6:
        return 'Medium'
    else:
        return 'High'

def run(raw_data):
    """
    Preprocesses JSON data, generates predictions, and returns results.
    Called for each client request sent to the endpoint.
    """
    try:
        # 1. Parse JSON input data
        payload = json.loads(raw_data)
        
        # Support common formats: {"data": [...]}, {"input_data": {"data": [...]}}, or direct list/dict
        if isinstance(payload, dict):
            if "data" in payload:
                records = payload["data"]
            elif "input_data" in payload:
                input_data = payload["input_data"]
                if isinstance(input_data, dict) and "columns" in input_data and "data" in input_data:
                    records = pd.DataFrame(data=input_data["data"], columns=input_data["columns"])
                else:
                    records = input_data
            else:
                # Single record dict
                records = [payload]
        elif isinstance(payload, list):
            records = payload
        else:
            return json.dumps({"error": "Invalid format. Expected JSON list of records or dictionary under key 'data'."})
            
        # Convert to pandas DataFrame
        if not isinstance(records, pd.DataFrame):
            df_input = pd.DataFrame(records)
        else:
            df_input = records.copy()
            
        # 2. Check and prepare expected inputs (15 base features)
        base_features = [
            'loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade',
            'emp_length', 'home_ownership', 'annual_inc',
            'verification_status', 'purpose', 'dti',
            'delinq_2yrs', 'inq_last_6mths', 'revol_util', 'pub_rec'
        ]
        
        # Re-initialize missing columns as NaN
        for col in base_features:
            if col not in df_input.columns:
                df_input[col] = np.nan
                
        # 3. Apply cleaning conversions (mirroring training pipeline)
        # Interest Rate
        if df_input['int_rate'].dtype == object:
            df_input['int_rate'] = df_input['int_rate'].astype(str).str.replace('%', '').str.strip().astype(float)
            
        # Revolving Line Utilization Rate
        if df_input['revol_util'].dtype == object:
            df_input['revol_util'] = df_input['revol_util'].astype(str).str.replace('%', '').str.strip().astype(float)
            
        # Employment Length (map string values to numeric ordinal values)
        emp_length_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
        }
        if df_input['emp_length'].dtype == object:
            df_input['emp_length'] = df_input['emp_length'].map(emp_length_map)
            
        # Loan Grade (map to numeric ordinal values)
        grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        if df_input['grade'].dtype == object:
            df_input['grade'] = df_input['grade'].map(grade_map)
            
        # Term (extract numeric months from string)
        if df_input['term'].dtype == object:
            df_input['term'] = df_input['term'].astype(str).str.strip().str.extract('(\\d+)').astype(float)
            
        # 4. Apply categorical label encoders (robust to unseen inputs)
        cat_cols = ['sub_grade', 'home_ownership', 'verification_status', 'purpose']
        for col in cat_cols:
            le = label_encoders[col]
            df_input[col] = df_input[col].fillna('missing').astype(str)
            # Map categories present in fit, default to 0 otherwise
            df_input[col] = df_input[col].apply(lambda x: int(le.transform([x])[0]) if x in le.classes_ else 0)
            
        # 5. Compute Engineered Features
        df_input['income_to_loan_ratio'] = df_input['annual_inc'] / (df_input['loan_amnt'] + 1)
        df_input['interest_burden'] = df_input['int_rate'] * df_input['loan_amnt'] / 100
        
        # 6. Impute missing values with medians
        for col in feature_columns:
            if col in medians:
                df_input[col] = df_input[col].fillna(medians[col])
            else:
                df_input[col] = df_input[col].fillna(0.0)
                
        # 7. Sort/reindex features to match model expectations
        df_inference = df_input[feature_columns]
        
        # 8. Generate Predictions
        pred_probas = model.predict_proba(df_inference)[:, 1]
        pred_classes = (pred_probas > threshold).astype(int)
        
        # 9. Format outputs
        results = []
        for prob, pred_class in zip(pred_probas, pred_classes):
            results.append({
                "default_probability": float(prob),
                "default_prediction": int(pred_class),
                "risk_tier": assign_risk(prob)
            })
            
        return json.dumps({"predictions": results})
        
    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        })
