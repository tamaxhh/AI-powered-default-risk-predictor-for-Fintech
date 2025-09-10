from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from config import CV_FOLDS
from utils import log_info, plot_cv_results, create_comparison_df

def evaluate_models(models, X, y, X_test, y_test):
    """
    Evaluate using cross_val_score and accuracy_score from notebook.
    """
    log_model, svm_model, dt_model = models

    # Cross-validation from notebook
    log_cv_scores = cross_val_score(log_model, X, y, cv=CV_FOLDS).mean()
    svm_cv_scores = cross_val_score(svm_model, X, y, cv=CV_FOLDS).mean()
    dt_cv_scores = cross_val_score(dt_model, X, y, cv=CV_FOLDS).mean()

    cv_results = {
        "Logistic Regression": log_cv_scores,
        "SVM": svm_cv_scores,
        "Decision Tree": dt_cv_scores
    }
    plot_cv_results(cv_results)  # Plot from notebook

    # Predictions and accuracy from notebook
    log_pred = log_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)

    log_acc = accuracy_score(y_test, log_pred) * 100
    svm_acc = accuracy_score(y_test, svm_pred) * 100
    dt_acc = accuracy_score(y_test, dt_pred) * 100

    comparison_df = create_comparison_df([log_acc, svm_acc, dt_acc])
    log_info(f"Comparison:\n{comparison_df}")

    return comparison_df