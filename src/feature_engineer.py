import numpy as np
from utils import log_info

def engineer_features(data):
    """
    Create new features: TotalIncome (common in such datasets), DTI (Debt-to-Income = LoanAmount / TotalIncome).
    Credit length not possible (no data). Notebook has no features, but added as per structure comment.
    """
    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
    data['DTI'] = data['LoanAmount'] / data['TotalIncome']  # Debt-to-Income ratio
    data['LoanAmount_Log'] = np.log(data['LoanAmount'] + 1)  # Log transform to handle skew (common)
    log_info("New features added: TotalIncome, DTI, LoanAmount_Log")
    return data