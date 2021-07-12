import numpy as np

def check_1d_array(array):
    if array.ndim == 1:
        return array
    elif array.ndim == 2 and array.shape[1] == 1:
        return array.flatten()
    else:
        raise TypeError('Unexpected array dimensions')

def check_2d_array(array):
    if array.ndim == 2 and array.shape[1] == 1:
        return array
    elif array.ndim == 1:
        return array.reshape(array.size, 1)
    else:
        raise TypeError('Unexpected array dimensions')

def trapezoid_area(x1, x2, y1, y2):
    return 0.5 * abs(x1 - x2) * (y1 + y2) 