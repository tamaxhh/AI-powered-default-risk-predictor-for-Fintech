from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils import log_info

def train_models(X_train, y_train):
    """
    Train models from notebook: LogisticRegression, SVC, DecisionTreeClassifier.
    """
    log_model = LogisticRegression()
    svm_model = SVC()
    dt_model = DecisionTreeClassifier()

    log_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)

    log_info("Models trained: LogisticRegression, SVC, DecisionTreeClassifier")
    return log_model, svm_model, dt_model