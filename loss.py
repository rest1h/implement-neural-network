import numpy as np

def mean_squared_error(y_pred, y_true):
    return np.square(np.subtract(y_true, y_pred)).mean()
    
def cross_entropy_error(y_pred, y_true):
    delta = 1e-7
    return -np.sum(y_true * np.log(y_pred + delta))