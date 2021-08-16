import numpy as np
import itertools as it
from .network import Network
from ..utilities.split import get_stratified_k_folds
from ..utilities.preprocessing import data_preprocessing
from ..evaluate import classifier as evcls

REQUIRED_NETWORK_PARAMETERS = ['network_structure', 'hidden_layer_function', 'output_function']
TRAINING_SEARCH_PARAMETERS = ['l2_param']
DEFAULT_TRAINING_PARAMETERS = {'fold_count': 5, 'data_preprocessors': []}


def _generate_search_cases(network_parameters, training_parameters):

    search_cases ={}

    # Check whether any parameters only have a single case and if necessary convert it to a tuple so the iterator can be formed correctly
    for parameter in network_parameters:

        if not isinstance(network_parameters[parameter], (tuple, list)):
            search_cases[parameter] = (network_parameters[parameter],)

        elif not isinstance(network_parameters[parameter][0], tuple):
            search_cases[parameter] = (network_parameters[parameter],)
    
    # Check for training param search
    for parameter in TRAINING_SEARCH_PARAMETERS & training_parameters.keys():

        if isinstance(training_parameters[parameter], (tuple, list, np.ndarray)):
            search_cases[parameter] = training_parameters[parameter]
        else:
            search_cases[parameter] = (training_parameters[parameter],)

    # Use an iterator to generate the parameter combinations and return them as a list of dictionaries 
    return [dict(zip(search_cases.keys(), combination)) for combination in it.product(*search_cases.values())]



def hyperparameter_search(inputs, targets, network_parameters, training_parameters, optimiser_parameters, verbose=True):

    """
    hyperparameter_search(inputs, targets, search_parameters, training_parameters)

    Function to perform hyperparameter search

    Arguments:  inputs                  - numpy array of inputs

                targets                 - numpy array of targets
                
                network_parameters       - dictionary of {parameter: search cases} containing a tuple of search cases for each parameter
                                            - required parameters: network_structure, hidden_layer_function, output_function
                                            - optional parameters: l2_param

                training_parameters     - dictionary of {parameter: value} containing desired training parameters
                                            - required parameters: fold_count, optimiser
                                            - optional parameters: data_preprocessing

                optimiser_parameters    - dictionary of {parameter: value} containing parameters for optimisation algorithm
                                            - required parameters: mini_batch_size, epochs, learning_rate

    Returns:    scores                  - The evaluated scores obtained by cross validation for each combination of parameters, sorted in descending order

                combinations            - A list of dictionaries where the dictionary at combinations[i] contains the parameter combination which obtained a score of scores[i]

    """

    # Check that required network parameters been provided, and if not raise an error:
    if any([parameter not in network_parameters for parameter in REQUIRED_NETWORK_PARAMETERS]):
        raise ValueError(f'Must specify at least 1 case for each of the required parameters:\n{network_parameters}')

   # Generate the combinations
    combinations = _generate_search_cases(network_parameters, training_parameters)
    combination_count = len(combinations)

    # initialise an array to hold the scores
    scores = np.zeros((combination_count, 2))
    

    ii = 0
    for combination in combinations:
        network_parameters = {parameter: combination[parameter] for parameter in REQUIRED_NETWORK_PARAMETERS }
        for parameter in TRAINING_SEARCH_PARAMETERS & training_parameters.keys():
            training_parameters[parameter] = combination[parameter]
        
        if verbose:
            print(f'\n\n ------------------- Combination: {ii+1} of {combination_count} -------------------\n\n')
            for key, value in combination.items():
                print(f'{key}:  {value}\n')
            print(f' -------------------------------------------------------------\n')
        
        scores[ii, :] = cross_validation(inputs, targets, network_parameters, training_parameters, optimiser_parameters)
        ii += 1

    score_ranks = np.argsort(scores[:,0])[::-1]
    ranked_scores = scores[score_ranks, ...]
    ranked_combinations = [combinations[rank] for rank in score_ranks]

    return scores, ranked_combinations


  
def cross_validation(inputs, targets, network_parameters, training_parameters, optimiser_parameters):

    """
   cross_validation(inputs, targets, network_parameters, training_parameters, fold_count, optimiser_parameters)

    Function to perform (fold_count)-fold cross validation.

    Arguments:  inputs                  - numpy array of inputs

                targets                 - numpy array of targets
    
                network_parameters      - dictionary of {parameter: value} containing network parameters
                                            - required parameters: network_structure, hidden_layer_function, output_function

                training_parameters     - dictionary of {parameter: value} containing desired training parameters
                                            - required parameters: fold_count, optimiser
                                            - optional parameters: data_preprocessing, l2_param

                optimiser_parameters    - dictionary of {parameter: value} containing parameters for optimisation algorithm
                                            - required parameters: mini_batch_size, epochs, learning_rate

    """

    # Check that required network parameters been provided, and if not raise an error:
    if any([parameter not in network_parameters for parameter in REQUIRED_NETWORK_PARAMETERS]):
        raise ValueError(f'network_parameters must contain a value for each of the required parameters:\n{network_parameters}')

    # Check that required training parameters been provided, and if not use defaults:
    for parameter in DEFAULT_TRAINING_PARAMETERS:
        if parameter not in training_parameters:
            training_parameters[parameter] = DEFAULT_TRAINING_PARAMETERS[parameter]

    # Split data set into k folds and initialise array to hold scores
    folds = get_stratified_k_folds(targets, training_parameters['fold_count'])
    scores = np.zeros(training_parameters['fold_count'])
    ii = 0
    
    # Perform cross validation
    for fold_indicies in folds:

        # split data into test and train data sets
        training_inputs = np.delete(inputs, fold_indicies, axis=0)
        training_targets = np.delete(targets, fold_indicies, axis=0)
        test_inputs = inputs[fold_indicies, ...]
        test_targets = targets[fold_indicies, ...]

        # Preprocess data
        preprocessed_training_inputs, preprocessed_test_inputs, preprocessing_params = data_preprocessing(training_inputs, test_inputs, training_parameters['data_preprocessors'])

        # Unpack network params and provide to network constructor
        network = Network(**network_parameters)

        # if optimiser is set to verbose, print a header for the current fold to aid readibility
        #if 'verbose' in optimiser_parameters.keys():
            #if optimiser_parameters['verbose']:
        print(f'\n----- fold {ii+1} of {training_parameters["fold_count"]} -----\n')
        
                # Train network
        network.train(preprocessed_training_inputs, training_targets, training_parameters, optimiser_parameters)

        # Evaluate classifier
        test_model_outputs = network.feedforward(preprocessed_test_inputs)
        scores[ii] = evcls.vuroc_hand_till(test_model_outputs, test_targets)
        ii += 1

    print(f'\n----- Cross validation Summary -----\nScores:\t\t\t{scores}\nMean:\t\t\t{np.mean(scores)}\nStandard Deviation:\t{np.std(scores)}\n')

    return np.array([np.mean(scores), np.std(scores)])