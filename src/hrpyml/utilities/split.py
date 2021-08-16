import numpy as np
from math import floor

# Exception for wrong number of arguments
class DataSetError(ValueError):
    pass


# perform simple stratified test/train split
def stratified_split(targets, test_proportion = 0.2):

    """ 
    stratified_split(targets, test_proportion = 0.2)

    Function to get stratified test and train data sets for classification from a one-hot encoded target vector
    
    """

    # if binary classification extend target vector to include negative class
    if targets.ndim == 1:
        targets = np.stack((targets, 1-targets), axis=1)
    elif targets.shape[1] == 1:
        targets = np.concatenate((targets, 1-targets), axis=1)

    training_proportion = 1 - test_proportion   
    training_indicies = np.array([], dtype=np.int32)
    test_indicies = np.array([], dtype=np.int32)
    for ii in range(0, targets.shape[1]):
        class_indicies = np.arange(0, targets.shape[0])[targets[:, ii] == 1]
        if class_indicies.size < 2:
            raise DataSetError(f"The dataset can't be split. The supplied dataset has less than 2 examples for the class at index {ii}.")
        training_indicies = np.concatenate((training_indicies, class_indicies[0:floor(training_proportion*class_indicies.size)]))
        test_indicies = np.concatenate((test_indicies, class_indicies[floor(training_proportion*class_indicies.size):]))

    training_indicies.sort()
    test_indicies.sort()

    return training_indicies, test_indicies


def get_stratified_k_folds(targets, fold_count):

    """ 
    get_stratified_k_folds(targets, fold_count)

    Function to get k stratified folds for k-fold cross validation from a one-hot encoded target vector
    
    """

    # if binary classification extend target vector to include negative class
    if targets.ndim == 1:
        targets = np.stack((targets, 1-targets), axis=1)
    elif targets.shape[1] == 1:
        targets = np.concatenate((targets, 1-targets), axis=1)

    # Get number of samples in each class
    class_counts = np.sum(targets, axis=0)

    # Calculate the largest integer fold size and the remainder for each class
    class_count_quotient, class_count_remainder = np.divmod(class_counts, fold_count)

    folds = [np.array([], dtype=np.int32) for fold in range(0, fold_count)]
    
    for ii in range(0, targets.shape[1]):
        class_indicies = np.arange(0, targets.shape[0])[targets[:, ii] == 1]
        if class_indicies.size < fold_count:
            raise DataSetError(f"The data set can't be split into {fold_count} folds. The supplied data set only has {class_indicies.size} examples for the class at index {ii}.")
        current_index = 0
        fold_index = 0
        
        # Loop through each fold and add the correct number of samples from the current class
        for fold_index in range(0, fold_count):
            class_in_fold_count = class_count_quotient[ii] + int(fold_index < class_count_remainder[ii])
            folds[fold_index] = np.concatenate((folds[fold_index], class_indicies[current_index:current_index + class_in_fold_count]))
            current_index += class_in_fold_count

    # Sort to preserve the original order in the folds
    for fold_indicies in folds:
        fold_indicies.sort()

    return tuple(folds)
