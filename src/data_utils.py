# ==================================================
# This file contains all functions for data like 
# exploration, cleaning 
# ==================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from model_utils import save_vocab
# =============================================================================
# 1. Save /Load data
# =============================================================================
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV file into a pandas DataFrame.

    Args:
        file_path : str
            Directory of data file (csv).
    
    Returns: pd.DataFrame
        A dataframe of csv file.
    """
    return pd.read_csv(file_path, sep=";")

def save_data(df: pd.DataFrame, file_path: str):
    """Sava dataframe into a CSV file.

    Args:
        df : pd.DataFrame
            A (cleaned) dataframe.
        file_path: str
            Directory to save data.
    
    Returns: True of False.
        If data is saved successfuly -> True, if not -> False.
    """
    df.to_csv(file_path, index=False)

# =============================================================================
# 2. Explore data
# =============================================================================
def explore_data(df: pd.DataFrame):
    """
    Quick exploration: print shape, head, summary stats, missing values.

    Args:
        file_path : pd.DataFrame
            A dataframe.
    
    Returns: None
        print the values.
    """
    print("Shape:", df.shape)
    print("Head:\n", df.head())
    print("Info:")
    print(df.info())
    print("Missing values:\n", df.isna().sum())
    print("Describe:\n", df.describe())

# =============================================================================
# 3. Clean data
# =============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame:
    - Drop duplicates if any, showing counts before and after
    - Fill missing values if any, showing counts before and after

    Args:
        file_path : pd.DataFrame
            A raw dataframe.
    
    Returns: 
        pd.DataFrame : A cleaned dataframe.
    """
    # Check duplicates before removing any
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"Found {dup_count} duplicate rows. Removing them...")
        df = df.drop_duplicates()
        print(f"After removing duplicates: {df.shape[0]} rows")
    else:
        print("No duplicate rows found.")
    
    # Check missing values before removing any
    missing_count = df.isna().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Filling them with 0...")
        df = df.fillna(0)
        print(f"After filling missing values: {df.isna().sum().sum()} missing values remaining")
    else:
        print("No missing values found.")
    
    return df

# =============================================================================
# 4. create text column
# =============================================================================
def create_text_column(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Create a column text by combining 
    content and title review column.

    Args:
        df : pd.DataFrame
            A cleaned dataframe.
        columns :
            title and content review columns.
    
    Returns: pd.DataFrame
        A new dataframe with text column.
    """
    df["text"] = df[columns[0]].fillna('').astype(str) + " " + df[columns[1]].fillna('').astype(str)

    return df

# =============================================================================
# 5. clean text
# =============================================================================
def clean_text(text:str) -> str:
    """Clean the content of data.
    
    Args:
        text : str
            A row of data frame, only the text column.
    
    Returns: str
        A cleaned text.
    """

    text = text.lower()                            # lowercase everything
    text = re.sub(r"http\S+", "", text)             # remove URLs
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)     # keep only letters (with accents)
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text

# =============================================================================
# 6. Convert to numerical vectors by TF-IDF
# =============================================================================
def convert_by_TF_IDF(df: pd.DataFrame) -> tuple:
    """Convert text to numerical vectors by TF-IDF.

    Args:
        df : pd.DataFrame
            A dataframe.
    
    Returns: X, y
        Data as numerical vectors and target column.
    """
    # create vocabulary
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["text"])
    y = df["Target"]

    # save vocabulary
    save_vocab(vectorizer, "TF_IDF_vocab")
    
    return X, y

# =============================================================================
# 7. Scikit-learn train_test_split
# =============================================================================
def split_data(X, y, test_size=0.2, random_state=42):
    """First split data to X and y
        Second, split X and y to train_test sets.

    Args:
        X : Sparse matrix of 
            numerical vectors.
        y: pd.pandas
            Target column.
        test_size : float
            Test size as a float number.
        random_state : int
            Random state value for splitting data.
    
    Returns: 4 values:
        X_train, y_train, X_test, y_test.
    """

    return train_test_split(X, y, test_size=test_size, random_state=random_state)