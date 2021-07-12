import numpy as np
import pandas as pd
from math import floor

# Splits pandas dataframe into two numpy arrays of input and target data
def dataframe_to_inputs_targets(dataframe, input_column_indexes, target_column_indexes=-1):
    """
    dataframe_to_inputs_targets(dataframe, input_column_indexes, target_column_indexes=-1)

    arguments:  dataframe               - pandas dataframe containing all data
                input_column_indexes    - tuple of (start, end) integer column indexes for input data
                target_column_indexes   - tuple of (start, end) integer column indexes for target data, if left blank assumes only last column

    returns: inputs, targets
    """

    input_slice = np.s_[:,input_column_indexes[0]:input_column_indexes[1]]
    if isinstance(target_column_indexes, int):
        target_slice =  np.s_[:,target_column_indexes]
    else:
        target_slice = np.s_[:,target_column_indexes[0]:target_column_indexes[1]]

    return dataframe.iloc[input_slice].to_numpy(copy=True), dataframe.iloc[target_slice].to_numpy(copy=True)


# Normalise data
def normalise_data(data, *args):

    # If just data is provided, compute mean and standard deviation from data and return normalised data plus a tuple of (mean, standard deviation)
    if len(args) == 0:
        mean = np.mean(data, 0)
        standard_deviation = np.std(data, 0)

        return (data - mean)/standard_deviation, (mean, standard_deviation)

    # If data and a tuple of (mean, standard deviation) is provided, return data normalised using (mean, standard deviation)
    elif len(args) == 1:
        mean = args[0][0]
        standard_deviation = args[0][1]

        return (data - mean)/standard_deviation
    
    else:
        raise TypeError('Wrong number of arguments supplied')


# Compute principle components analysis
def principle_components_analysis(data):

    eigenvalues, eigenvectors = np.linalg.eig(data.transpose() @ data)
    sort_indicies = np.argsort(eigenvalues, axis=None)
    eigenvalues_sorted = eigenvalues[sort_indicies]
    eigenvectors_sorted = eigenvectors[:, sort_indicies]

    return eigenvectors_sorted, eigenvalues_sorted


def data_preprocessing(training_data, test_data, preprocessors=[]):
    """
    data_preprocessing(training_data, test_data, preprocessors)

    Function to automatically perform preprocessing on training and test data.

    Arguments:  training_data       - numpy array of training data
                test_data           - numpy array of test data
                preprocessors       - list of strings indicating desired preprocessors, options:
                                            - normalise, pca

    Returns:    training_data, test_data, preprocessing_params
    """

    preprocessing_params = {}

    if 'normalise' in preprocessors:
        training_data, normalisation_parameters = normalise_data(training_data)
        test_data = normalise_data(test_data, normalisation_parameters)
        preprocessing_params['normalisation'] = normalisation_parameters


    if 'pca' in preprocessors:
        pca_eigenvectors, pca_eigenvalues = principle_components_analysis(training_data)
        training_data = training_data @ pca_eigenvectors
        test_data = test_data @ pca_eigenvectors
        preprocessing_params['pca'] = (pca_eigenvectors, pca_eigenvalues)

    return training_data, test_data, preprocessing_params