import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

from src.data_preprocessing import preprocess
from src.feature_engineer import engineer_features
from src.train_model import train_models
from src.evaluate_model import evaluate_models
from src.interpret_model import interpret_model

# Run pipeline
X_full, y_full, X_train, X_test, y_train, y_test = preprocess()

# Apply feature engineering (on full data, then resplit if needed; for simplicity, apply to pre-split)
X_train = engineer_features(X_train)
X_test = engineer_features(X_test)
X_full = engineer_features(X_full)  # For CV

models = train_models(X_train, y_train)
evaluate_models(models, X_full, y_full, X_test, y_test)

# Interpret best model (Logistic Regression)
interpret_model(models[0], X_full)