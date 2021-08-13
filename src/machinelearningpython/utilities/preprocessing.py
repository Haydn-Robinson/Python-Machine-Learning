import numpy as np
import pandas as pd
from math import floor

# Splits pandas dataframe into two numpy arrays of input and target data
def dataframe_to_inputs_targets(dataframe, input_count, target_count=1, input_start=0, target_start=None):
    """
    dataframe_to_inputs_targets(dataframe, input_column_indexes, target_column_indexes=-1)

    arguments:  dataframe               - Pandas dataframe containing all data
                input_count             - The number of input variables (integer)
                target_count            - The number of target variables (integer - defaults to 1)
                input_start             - Starting index of the input variables (integer - defaults to 0)

    returns:    inputs                  - numpy array containing the inputs
                targets                 - numpy array containing the targets
    """

    input_slice = np.s_[:, input_start:input_start + input_count]

    if target_start == None:
        target_start = input_start + input_count
    
    if target_count == 1:
        target_slice =  np.s_[:,target_start]
    else:
        target_slice = np.s_[:,target_start:target_start + target_count]

    return dataframe.iloc[input_slice].to_numpy(copy=True), dataframe.iloc[target_slice].to_numpy(copy=True)


# Function to one-hot encode an array of integer target labels
def int_labels_to_one_hot(targets):
    """
    int_labels_to_one_hot(targets, label_start=0)

    arguments:  targets                 - Target vector of integer encoded class labels with shape (n,) or (n,1)
                label_start             - The integer at which the class labels starts, defaults to 0


    returns:    one_hot                 - One-hot encoded target array of shape (n,c) where c is the number of classes
                classes                 - Numpy array containing the c unique classes in the order in which they are encoded
    """

    if targets.ndim == 2 and targets.shape[1] == 1:
        targets.flatten()
    elif targets.ndim != 1:
        raise ValueError('Target must be of shape (n,) or (n,1) for this operation')

    classes = np.unique(targets)
    one_hot = np.zeros((targets.size, classes.size), dtype=np.int32)
    for ii in range(0, classes.size):
        one_hot[targets == classes[ii], ii] = 1
    
    return one_hot, classes


# Function to one-hot encode an array of string target labels
def string_labels_to_one_hot(targets, string_lables=None):
    """
    string_labels_to_one_hot(targets, label_start=0)

    arguments:  targets                 - Target array of string encoded class labels with shape (n,) or (n,1)
                string_lables           - Iterable containing all unique class labels in the order desired for the one-hot encoding,
                                          if none is provided classes will be encoded in alphabetical order.


    returns:    one_hot                 - One-hot encoded target array of shape (n,c) where c is the number of classes
    """

    if targets.ndim == 2 and targets.shape[1] == 1:
        targets.flatten()
    elif targets.ndim != 1:
        raise ValueError('Target must be of shape (n,) or (n,1) for this operation')

    if string_labels != None:
        class_count = len(string_labels)
    else:
        string_labels = np.unique(targets)
        class_count = len(classes)

    one_hot = np.zeros((targets.size, class_count))

    for ii in range(0, class_count):
        class_indicies = np.arange(0, targets.shape[0])[targets[ii] == string_labels[ii]]
        one_hot[class_indicies, ii] = 1

    return one_hot


# Normalise data
def normalise_data(data, *args):
    """
    normalise_data(data, normalise_params=None)

    Function to automatically perform preprocessing on training and test data.

    Arguments:  data                    - Numpy array of data to be normalised
                normalise_params        - Optional argument to specify tuple of (mean, standard deviation). If this is provided,
                                          data will be normalised using the provided parameters, otherwise (mean, standard deviation)
                                          will be computed from the dataset


    Returns:    normalised_data         - array of normalised training data
    """

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
    """
    principle_components_analysis(data)

    Function to obtain arrays of eigenvectors and eigenvalues, sorted by eigenvalue, to use in PCA.

    Arguments:  data                    - Numpy array of data on which to perform PCA.

    Returns:    eigenvectors_sorted     - array of eigenvectors, sorted in decreasing order according to eigenvalue
                eigenvalues_sorted      - vector of eigenvalues sorted in decreasing order
    """

    eigenvalues, eigenvectors = np.linalg.eig(data.transpose() @ data)
    sort_indicies = np.argsort(eigenvalues, axis=None)[::-1]
    eigenvalues_sorted = eigenvalues[sort_indicies]
    eigenvectors_sorted = eigenvectors[:, sort_indicies]

    return eigenvectors_sorted, eigenvalues_sorted


# Wrapper function to perform several types of preprocessing in one go 
def data_preprocessing(training_data, test_data, preprocessors=None):
    """
    data_preprocessing(training_data, test_data, preprocessors)

    Function to automatically perform preprocessing on training and test data.

    Arguments:  training_data           - numpy array of training data
                test_data               - numpy array of test data
                preprocessors           - list of strings indicating desired preprocessors, options:
                                            - normalise, pca
                                            - defaults to empty list (no preprocessing)

    Returns:    training_data           - array of preprocessed training data
                test_data               - array of preprocessed test data
                preprocessing_params    - dictionary containing the preprocessing parameters for each applied method of preprocessing
    """

    if preprocessors == None:
        preprocessors = []

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