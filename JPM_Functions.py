"""ML PreProcessing and Model Performance Functions

outlier_removal: Removes rows with outliers in a specified column
my_train_test_split: Splits data into a testing and training by specified split percentage
my_mse: Returns mean square error between predicted and actual
my_rmse: Returns root mean square error between predicted and actual
my_mae: Returns mean absolute error between predicted and actual 
my_r2: Returns r2 (coefficient of determination) between predicted and actual
"""

import random
import math
import pandas as pd
import numpy as np

# Outlier Removal
def outlier_removal(df: pd.DataFrame, column: str):
    """Removes rows with outliers in a particular column

    Args:
        df (pandas DataFrame): The DataFrame we want the values removed from
        column (str): The column we're checking for outliers

    Returns:
        df (pandas DataFrame): The DataFrame minus the outliers
    """
    # Calculate IQR
    upper_qr = df[column].quantile(0.75)
    lower_qr = df[column].quantile(0.25)
    inter_qr = upper_qr-lower_qr
    # Filter for outliers
    df = df[df[column]<(upper_qr+(1.5*inter_qr))] # Gets all values below upper outlier limits
    df = df[df[column]>(lower_qr-(1.5*inter_qr))] # Gets all values above lower limits
    # Reset indexes
    df.reset_index(drop=True)

    return df

# Train, Test Split
def my_train_test_split(X: pd.DataFrame,y: pd.DataFrame, test_size: float = 0.2, random_state: int = None) -> pd.DataFrame:
    """Splits data into training and testing DataFrames

    Args:
        X (pd.DataFrame): Input DataFrame to be split
        y (pd.DataFrame): Output DataFrame to be split
        test_size (float, optional): Percentage of data in test DataFrame. 
            Defaults to 0.2.
        random_state (int, optional): Saves the split configuration for reusability. 
            Defaults to None.

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame): Split DataFrames 
    """
    #get random seed to allow reproduceability
    random.seed(random_state)
    #select test indexes from range of total indexes
    test_ixs= random.sample(range(len(y)), math.floor(len(y)*test_size))
    #return X_train, X_test, y_train, y_test (train sets are total sets - test sets)
    return X.drop(test_ixs), X.iloc[test_ixs], y.drop(test_ixs), y.iloc[test_ixs]

# Performance Metric Functions: MSE, RMSE, MAE, R2
def my_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Find Mean Square Error (MSE) between predicted and observed data.

    Args:
        a (np.ndarray): Observed Data
        b (np.ndarray): Predicted Data

    Returns:
        mse (float): The final MSE
    """
    return np.mean((a-b)**2)

def my_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Finds root mean square error (RMSE) between predicted and observered data.

    Args:
        a (np.ndarray): Observed Data
        b (np.ndarray): Predicted Data

    Returns:
        math.sqrt(MSE) (float): Final root mean square error (RMSE) in eq format.
    """
    return math.sqrt(my_mse(a,b))

def my_mae(a: np.ndarray,b: np.ndarray) -> float:
    """Finds mean absolute error (MAE) between predicted and observed data.

    Args:
        a (np.ndarray): Observed Data
        b (np.ndarray): Predicted Data

    Returns:
        mae (float): Final mean absolute error (MAE)
    """
    return np.mean(np.abs(a-b))

def my_r2(a: np.ndarray,b: np.ndarray) -> float:
    """Finds R2 (Coeff of Determination) score between predicted and observed data

    Args:
        a (np.ndarray): Observed Data
        b (np.ndarray): Predicted Data

    Returns:
        r2 (float): The final R2 score (in eq format)
    """
    y_bar = sum(a)/len(a)
    ss_res = np.mean((a-b)**2)
    ss_tot = np.mean((a-y_bar)**2)
    return 1-(ss_res/ss_tot)

def my_performance(a: pd.DataFrame, b: pd.DataFrame):
    """Turns predicted and observed data to numpy ndarrays and carries out MSE, RMSE, MAE, and R2

    Args:
        a (pd.DataFrame): Observed Data
        b (pd.DataFrame): Predicted Data
    """
    if not isinstance(a, np.ndarray):
        a=a.to_numpy()
    if not isinstance(b, np.ndarray):
        b=b.to_numpy()
    print(f'Mean Squared Error: {my_mse(a,b)}')
    print(f'Root Mean Squared Error: {my_rmse(a,b)}')
    print(f'Mean Absolute Error: {my_mae(a,b)}')
    print(f'R2 Score: {my_r2(a,b)}')
