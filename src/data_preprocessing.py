import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import RAW_DATA_PATH
from utils import log_info

def load_data():
    """
    Load data from CSV as in notebook.
    """
    data = pd.read_csv(RAW_DATA_PATH)
    log_info(f"Data loaded with shape: {data.shape}")
    return data

def clean_data(data):
    """
    Handle missing values as in notebook (mode for categorical, median for numerical).
    """
    data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])
    data["Married"] = data["Married"].fillna(data["Married"].mode()[0])
    data["Dependents"] = data["Dependents"].fillna(data["Dependents"].mode()[0])
    data["Self_Employed"] = data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])
    data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
    data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median())
    data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode()[0])
    log_info("Missing values filled")
    return data

def encode_data(data):
    """
    Encode categorical variables using LabelEncoder as imported in notebook.
    """
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = le.fit_transform(data[col])
    log_info("Categorical variables encoded")
    return data

def preprocess():
    """
    Full preprocessing: load, clean, encode, drop ID, split X/y, train_test_split.
    No scaling in notebook, but placeholder if needed.
    """
    data = load_data()
    data = clean_data(data)
    data = encode_data(data)
    data = data.drop('Loan_ID', axis=1)  # Implied, as Loan_ID not used in models
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    log_info(f"Data preprocessed and split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
    return X, y, X_train, X_test, y_train, y_test  # Return full X/y for CV