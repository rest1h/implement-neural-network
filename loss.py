import numpy as np


def mean_squared_error(y_pred, y_true):
    return np.mean(np.square(np.subtract(y_true, y_pred)))
