import logging
import matplotlib.pyplot as plt
import pandas as pd

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(message):
    logging.info(message)

def plot_cv_results(cv_results):
    """
    Plot cross-validation mean accuracies from notebook.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(cv_results.keys(), cv_results.values(), color=["skyblue", "orange", "lightgreen"])
    plt.title("Cross-Validation Mean Accuracy (5-Fold)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()

def create_comparison_df(accuracies):
    """
    Create comparison dataframe from notebook.
    """
    return pd.DataFrame({
        'Model': ['Logistic Regression', 'Support Vector Machine', 'Decision Tree'],
        'Accuracy_Score': accuracies
    }).sort_values(by='Accuracy_Score', ascending=False)