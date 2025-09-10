import os

# Data path from notebook (adjusted to relative; update if needed)
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data', 'Loan Status Prediction.csv')

# Model parameters and thresholds from notebook/implied
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5