import numpy as np
import math
from ._functions import COST_FUNCTIONS, COST_FUNCTION_SELECTION, REGULARISATION_COST_FUNCTIONS

DEFAULT_SGD_OPTIMISER_PARAMETERS = {'mini_batch_size': 10,
                                    'epochs': 50,
                                    'learning_rate': 0.001,
                                    'verbose': False,
                                    'learning_rate_decay_factor': 1,
                                    'learning_rate_decay_delay': 0,
                                    'learning_rate_decay_rate': 5}

DEFAULT_NAG_OPTIMISER_PARAMETERS = {'mini_batch_size': 10,
                                    'epochs': 50,
                                    'learning_rate': 0.001,
                                    'momentum_coefficient': 0.9,
                                    'verbose': False,
                                    'learning_rate_decay_factor': 1,
                                    'learning_rate_decay_delay': 0,
                                    'learning_rate_decay_rate': 5}


def stochastic_gradient_descent(network, inputs, targets, training_parameters, optimiser_parameters):

    # Check that all required optimiser parameters have been provided, and if not use defaults.
    for parameter in DEFAULT_SGD_OPTIMISER_PARAMETERS:
        if parameter not in optimiser_parameters:
            optimiser_parameters[parameter] = DEFAULT_SGD_OPTIMISER_PARAMETERS[parameter]

    # Initialise learning_rate and annealing factor
    annealing_factor = 1
    learning_rate = optimiser_parameters['learning_rate']

    # get size of supplied data and check that the mini batch size is smaller
    sample_size = inputs.shape[0]

    if optimiser_parameters['mini_batch_size'] > sample_size:
        raise ValueError('Mini-Batch exceeds size of data set')

    rng = np.random.default_rng()

    for ii in range(0, optimiser_parameters['epochs']):

        # Randomly permute the data set 
        epoch_indices = rng.permutation(sample_size)

        inputs_permuted = inputs[epoch_indices, ...]
        targets_permuted = targets[epoch_indices, ...]

        # Compute learning rate based on annealing schedule:
        if  optimiser_parameters['learning_rate_decay_factor'] != 1 and ii > optimiser_parameters['learning_rate_decay_delay']:
            annealing_factor = optimiser_parameters['learning_rate_decay_factor'] ** math.floor((ii - optimiser_parameters['learning_rate_decay_delay'])/optimiser_parameters['learning_rate_decay_rate'])
            learning_rate = optimiser_parameters['learning_rate'] * annealing_factor

        # Loop through each mini-batch in the current epoch
        for jj in range(0, math.ceil(sample_size/optimiser_parameters['mini_batch_size'])):            
            mini_batch_index = np.s_[optimiser_parameters['mini_batch_size'] * jj : min(optimiser_parameters['mini_batch_size'] * (jj + 1), sample_size)]

            # Compute gradients
            weights_gradients, biases_gradients = network._backpropagation(inputs_permuted[mini_batch_index,:], targets_permuted[mini_batch_index])

            # Update weights and biases
            weights, biases = network.get_weights()
            for kk in range(0, network.layer_count):

                # Add regularisation 
                weights_gradients[kk] += training_parameters['l2_param'] / inputs_permuted[mini_batch_index,...].shape[0] * weights[kk] 

                # Update weights
                weights[kk] -= learning_rate * weights_gradients[kk]
                biases[kk] -= learning_rate * biases_gradients[kk]

            # Repack into network
            network.update_weights(weights, biases)

        # If verbose, check cost and print summary at the end of each epoch
        if optimiser_parameters['verbose'] and ii % (math.floor(optimiser_parameters['epochs']/10)) == 0:
            outputs = network.feedforward(inputs)
            cost = COST_FUNCTIONS[COST_FUNCTION_SELECTION[network.layers[-1].output_function]](outputs, targets)
            reg_cost = REGULARISATION_COST_FUNCTIONS['l2'](network, training_parameters['l2_param'], targets)
            print(f'Epoch: {ii:>6}  --    Cost = {cost:<17.15f}    Regularisation Cost = {reg_cost:<17.15f}    Total Cost = {cost + reg_cost:<17.15f}')

    # If not verbose, check cost and print summary at the end of the optimisation process
    if not optimiser_parameters['verbose']:
        outputs = network.feedforward(inputs)
        cost = COST_FUNCTIONS[COST_FUNCTION_SELECTION[network.layers[-1].output_function]](outputs, targets)
        reg_cost = REGULARISATION_COST_FUNCTIONS['l2'](network, training_parameters['l2_param'], targets)
        print(f'Final Result    --    Cost = {cost}    Regularisation Cost = {reg_cost}    Total Cost = {cost + reg_cost}')

    return network


def nesterov_accelerated_gradient(network, inputs, targets, training_parameters, optimiser_parameters):

    # Check that all required optimiser parameters have been provided, and if not use defaults.
    for parameter in DEFAULT_NAG_OPTIMISER_PARAMETERS:
        if parameter not in optimiser_parameters:
            optimiser_parameters[parameter] = DEFAULT_SGD_OPTIMISER_PARAMETERS[parameter]

    # Initialise learning_rate and annealing factor
    annealing_factor = 1
    learning_rate = optimiser_parameters['learning_rate']
    momentum_coefficient = optimiser_parameters['momentum_coefficient']

    prev_weight_velocity = [np.zeros(layer.weights.shape) for layer in network.layers]
    weight_velocity = [np.zeros(layer.weights.shape) for layer in network.layers]
    prev_bias_velocity = [np.zeros(layer.biases.shape) for layer in network.layers]
    bias_velocity = [np.zeros(layer.biases.shape) for layer in network.layers]

    # get size of supplied data and check that the mini batch size is smaller
    sample_size = inputs.shape[0]

    if optimiser_parameters['mini_batch_size'] > sample_size:
        raise ValueError('Mini-Batch exceeds size of data set')

    rng = np.random.default_rng()

    for ii in range(0, optimiser_parameters['epochs']):

        # Randomly permute the data set 
        epoch_indices = rng.permutation(sample_size)

        inputs_permuted = inputs[epoch_indices, ...]
        targets_permuted = targets[epoch_indices, ...]

        # Compute learning rate based on annealing schedule:
        if  optimiser_parameters['learning_rate_decay_factor'] != 1 and ii > optimiser_parameters['learning_rate_decay_delay']:
            annealing_factor = optimiser_parameters['learning_rate_decay_factor'] ** math.floor((ii - optimiser_parameters['learning_rate_decay_delay'])/optimiser_parameters['learning_rate_decay_rate'])
            learning_rate = optimiser_parameters['learning_rate'] * annealing_factor

        # Loop through each mini-batch in the current epoch
        for jj in range(0, math.ceil(sample_size/optimiser_parameters['mini_batch_size'])):
            mini_batch_index = np.s_[optimiser_parameters['mini_batch_size'] * jj : min(optimiser_parameters['mini_batch_size'] * (jj + 1), sample_size)]

            # Compute gradients
            weights_gradients, biases_gradients = network._backpropagation(inputs_permuted[mini_batch_index,:], targets_permuted[mini_batch_index])

            # Update weights and biases
            weights, biases = network.get_weights()

            for kk in range(0, network.layer_count):
            
                # Add regularisation
                weights_gradients[kk] += training_parameters['l2_param'] / inputs_permuted[mini_batch_index,...].shape[0] * weights[kk]

                # Perform momentum update
                prev_weight_velocity[kk] = weight_velocity[kk]
                prev_bias_velocity[kk] = bias_velocity[kk]

                weight_velocity[kk] = momentum_coefficient * weight_velocity[kk] - learning_rate * weights_gradients[kk]
                bias_velocity[kk] = momentum_coefficient * bias_velocity[kk] - learning_rate * biases_gradients[kk]

                weights[kk] += (1 + momentum_coefficient) * weight_velocity[kk] - momentum_coefficient * prev_weight_velocity[kk]
                biases[kk] +=  (1 + momentum_coefficient) * bias_velocity[kk] - momentum_coefficient * prev_bias_velocity[kk]

            # Repack into network
            network.update_weights(weights, biases)

        # If verbose, check cost and print summary at the end of each epoch
        if optimiser_parameters['verbose'] and ii % (math.floor(optimiser_parameters['epochs']/10)) == 0:
            outputs = network.feedforward(inputs)
            cost = COST_FUNCTIONS[COST_FUNCTION_SELECTION[network.layers[-1].output_function]](outputs, targets)
            reg_cost = REGULARISATION_COST_FUNCTIONS['l2'](network, training_parameters['l2_param'], targets)
            print(f'Epoch: {ii:>6}  --    Cost = {cost:<17.15f}    Regularisation Cost = {reg_cost:<17.15f}    Total Cost = {cost + reg_cost:<17.15f}')

    # If not verbose, check cost and print summary at the end of the optimisation process
    if not optimiser_parameters['verbose']:
        outputs = network.feedforward(inputs)
        cost = COST_FUNCTIONS[COST_FUNCTION_SELECTION[network.layers[-1].output_function]](outputs, targets)
        reg_cost = REGULARISATION_COST_FUNCTIONS['l2'](network, training_parameters['l2_param'], targets)
        print(f'Final Result    --    Cost = {cost}    Regularisation Cost = {reg_cost}    Total Cost = {cost + reg_cost}')

    return network


def adam(network):
    pass


OPTIMISERS = {'sgd':stochastic_gradient_descent, 'nag': nesterov_accelerated_gradient, 'adam':adam}