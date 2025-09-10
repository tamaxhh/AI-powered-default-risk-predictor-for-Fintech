import shap  # Requires pip install shap (not in notebook, but added for structure)
from utils import log_info

def interpret_model(model, X):
    """
    Explain the best model (Logistic Regression from notebook) using SHAP.
    Notebook has no interpretation, but added basic as per structure.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)
    log_info("Model interpreted using SHAP summary plot")