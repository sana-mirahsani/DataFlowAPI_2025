# ==================================================
# This file contains all functions for training model 
# ==================================================

# =============================================================================
# 0. Libraries
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
from pathlib import Path
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
    # create LR model
    model = LogisticRegression(max_iter=num_iter)
    model.fit(X_train, y_train)

    # save LR model
    save_model(model, "logistic_regression")

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

# =============================================================================
# 3. Save model
# =============================================================================
def save_model(model, name:str):
    """
    Save the model to read it later in API.

    Args:
        model : ML model.
        name : str
            The name for model
    Returns: 
        A trained model.
    """
    # path of saving model
    path = "../models/" + name + ".pkl"
    data_path = os.path.abspath(path)
    
    try:
        joblib.dump(model, data_path)
        print("Model saved successfully!")
    except:
        raise Exception("Can't save the model!!")
    
# =============================================================================
# 4. Save vocabulary + transformer
# =============================================================================
def save_vocab(vectorizer, name:str):
    """
    Save the model to read it later in API.

    Args:
        model : ML model.
        name : str
            The name for model
    Returns: 
        A trained model.
    """
    # path of saving model
    path = "../models/" + name + ".pkl"
    data_path = os.path.abspath(path)
    
    try:
        joblib.dump(vectorizer, data_path)
        print("Vocabulary saved successfully!")
    except:
        raise Exception("Can't save the Vocabulary!!")