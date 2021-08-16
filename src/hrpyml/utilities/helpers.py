import numpy as np

def cast_to_1d_array(array):
    if array.ndim == 1:
        return array
    elif array.ndim == 2 and array.shape[1] == 1:
        return array.flatten()
    else:
        raise TypeError('Unexpected array dimensions')


def cast_to_2d_array(array):
    if array.ndim == 2 and array.shape[1] == 1:
        return array
    elif array.ndim == 1:
        return array.reshape(array.size, 1)
    else:
        raise TypeError('Unexpected array dimensions')


def trapezoid_area(x1, x2, y1, y2):
    return 0.5 * abs(x1 - x2) * (y1 + y2)


def binary_to_one_hot(targets, model_output):

    if targets.ndim == 1:
        targets = np.stack((targets, 1-targets), axis=1)
        model_output = np.stack((model_output, 1-model_output), axis=1)
    elif targets.shape[1] == 1:
        targets = np.concatenate((targets, 1-targets), axis=1)
        model_output = np.concatenate((model_output, 1-model_output), axis=1)

    return targets, model_output