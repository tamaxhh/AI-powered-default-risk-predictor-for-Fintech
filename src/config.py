import os

# Data path from notebook (adjusted to relative; update if needed)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'Loan Status Prediction.csv')
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, 'Data', 'processed', 'processed_loan_data.csv')
MODEL_DIR = r"C:\Users\anxaa\OneDrive\Documents\Github Project\AI-Powered-Loan-Approval-Classifier-main\AI-Powered-Loan-Approval-Classifier-main\models"

# Model parameters and thresholds from notebook/implied
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SMOTE_RANDOM_STATE = 42