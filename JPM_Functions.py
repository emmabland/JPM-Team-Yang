import random
import math
import pandas as pd

# Outlier Removal
def outlier_removal(df, column: str):
    """Removes rows with outliers in a particular column

    Args:
        df (pandas DataFrame): The DataFrame we want the values removed from
        column (str): The column we're checking for outliers

    Returns:
        df (pandas DataFrame): The DataFrame minus the outliers
    """
    # Calculate IQR
    upper_QR = df[column].quantile(0.75)
    lower_QR = df[column].quantile(0.25)
    inter_QR = upper_QR-lower_QR
    # Filter for outliers
    df = df[df[column]<(upper_QR+(1.5*inter_QR))] # Gets all values below upper outlier limits
    df = df[df[column]>(lower_QR-(1.5*inter_QR))] # Gets all values above lower limits
    # Reset indexes 
    df.reset_index(drop=True)
 
    return df

# Performance Metric Functiosn
# Below are our version of the performance metric functions, and the train, test, split function

def my_train_test_split(*arrays, test_size = 0.2, training_size = 0.8, random_state = None):
    if test_size+training_size!=1:
        raise 'Bad training/test split size'
    for i in arrays:
        if len(i)!=len(arrays[0]):
            raise 'Bad input object size'
    random.seed(random_state)
    array_size = len(arrays[0])
    num_test = math.floor(array_size*test_size)
    num_train = math.floor(array_size*training_size)
    num_train+=(array_size-num_train-num_test)
    return_list = []
    test_list = random.sample(range(array_size),num_test, )
    for i in arrays:
        if type(i)==pd.DataFrame:
            test_array = pd.DataFrame(columns = i.columns)
            train_array = pd.DataFrame(columns = i.columns)
        elif type(i)==pd.Series:
            test_array = pd.Series()
            train_array = pd.Series()
        test_array = i.iloc[test_list]
        train_array = i.drop(test_list)
        return_list.append(train_array)
        return_list.append(test_array)
    return return_list

def my_mse(a, b):
    a = a.to_numpy()
    sum = 0
    for i in range(len(a)):
        sum+=(a[i]-b[i])**2
    sum/=len(a)
    return sum

def my_rmse(a, b):
    return math.sqrt(my_mse(a,b))

def my_mae(a,b):
    a = a.to_numpy()
    sum = 0
    for i in range(len(a)):
        sum+=abs(a[i]-b[i])
    sum/=len(a)
    return sum

def my_r2(a,b):
    a = a.to_numpy()
    y_bar = sum(a)/len(a)
    ss_res = 0
    ss_tot = 0
    for i in range(len(a)):
        ss_res+=(a[i]-b[i])**2
    for j in range(len(a)):
        ss_tot+=(a[j]-y_bar)**2
    return 1-(ss_res/ss_tot)
