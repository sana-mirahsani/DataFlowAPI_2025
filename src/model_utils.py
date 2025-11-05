# ==================================================
# This file contains all functions for training model 
# ==================================================

# =============================================================================
# 0. Libraries
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =============================================================================
# 1. LogisticRegression model
# =============================================================================
def logistic_regression_model_training(X_train, y_train, num_iter=300):
    """
    Load and fit the model.

    Args:
        X_train : Sparse matrix
            data.
        y_train : pd.sereis
            Target column 0, 1
    
    Returns: model
        A trained model.
    """

    model = LogisticRegression(max_iter=num_iter)
    model.fit(X_train, y_train)
    return model

# =============================================================================
# 2. Evaluation for LogisticRegression
# =============================================================================
def eval_lr_model(X_test, y_test, model):
    """
    Evaluate LogisticRegression model
    by classifcation report.

    Args:
        X_test : Sparse matrix
            Unseen data.
        y_test : pd.sereis
            Unseen target column 0, 1
        model : LogisticRegression model
    Returns: 
        A trained model.
    """

    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)