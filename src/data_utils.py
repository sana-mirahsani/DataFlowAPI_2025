# ==================================================
# This file contains all functions for data like 
# exploration, cleaning 
# ==================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
# 4. Rescale data
# =============================================================================
def rescale_data(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Standardize numerical columns using StandardScaler.

    Args:
        file_path : pd.DataFrame
            A cleaned dataframe.
    
    Returns: pd.DataFrame
        A normalized dataframe.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# =============================================================================
# 5. Drop some columns
# =============================================================================
def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop unnecessary columns from the DataFrame.
    
    Args:
        df : pd.DataFrame
            A dataframe.
        columns : list[str]
            columns to remove.
    
    Returns: pd.DataFrame
        A dataframe without some columns.
    """
    return df.drop(columns=columns, errors='ignore')

# =============================================================================
# 6. Encoding categorical variables
# =============================================================================
def encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert categorical columns to numeric using one-hot encoding.

    Args:
        df : pd.DataFrame
            A dataframe.
        columns : list[str]
            Categorical columns.
    
    Returns: pd.DataFrame
        A dataframe with numeric columns.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)

# =============================================================================
# 7. Scikit-learn train_test_split
# =============================================================================
def split_data(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42):
    """First split data to X and y
        Second, split X and y to train_test sets.

    Args:
        df : pd.DataFrame
            A dataframe.
        target_col: str
            Target column.
        test_size : float
            Test size as a float number.
        random_state : int
            Random state value for splitting data.
    
    Returns: 4 values:
        X_train, y_train, X_test, y_test.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)