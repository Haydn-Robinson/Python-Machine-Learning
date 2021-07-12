from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pathlib
import itertools as it
import csv
from math import sqrt
from ._functions import ACTIVATIONS, ACTIVATIONS_DERIVATIVES, COST_FUNCTIONS, COST_FUNCTION_DERIVATIVES, COST_FUNCTION_SELECTION, REGULARISATION_COST_FUNCTION_DERIVATIVES
from ._stochasticoptimisers import OPTIMISERS
from ..utilities.helpers import check_1d_array


# Exception for wrong number of arguments
class WrongNumberOfArguments(TypeError):
    pass


# Layer Class
class Layer:

    def __init__(self, input_count, neuron_count, output_function, random_generator):

        """
        Randomly inititialise a layer object

        """
       
        self.neuron_count = neuron_count
        self.output_function = output_function

        if output_function == 'relu':
            variance_factor = sqrt(2/input_count)
        else:
            variance_factor = 1/sqrt(input_count)

        self.weights = random_generator.standard_normal((neuron_count, input_count)) * variance_factor
        self.biases = np.zeros((neuron_count, 1))



    # Method to obtain network output for given input data
    def feedforward(self, inputs):
       
        # Perform feedforward computation
        self.weighted_sums = self.weights @ inputs + self.biases
        self.outputs = ACTIVATIONS[self.output_function](self.weighted_sums)
        return self.outputs
    

# Network Class
class Network:

    def __init__(self, network_structure, hidden_layer_function, output_function):

        """
        __init__(self, network_structure, hidden_layer_function, output_function)

        Inititialise a network 

            networkStructure, a tuple of the form (ni,nh1,...,nhm,no) where
                - ni is the number of neurons in the input layer
                - nh1 is the number of neurons in the first hidden layer
                - nhm is the number of neurons in the mth hidden layer
                - no is the number of neurons in the output layer.

            hiddenLayerFunction, a string specifying the desired type of neuron to use in the hidden layer, options are
                - sigmoid, tanh, ReLu

            outputFunction, a string specifying the desired type of neuron to use in the output layer, options are
                - sigmoid, tanh, ReLu

        """
     

        self._new_network(network_structure, hidden_layer_function, output_function)
       
       

    # Method to initialise new network
    def _new_network(self, network_structure, hidden_layer_function, output_function):

        # Get network dimensions
        self.input_count = network_structure[0]
        self.output_count = network_structure[-1]
        self.layer_count = len(network_structure) - 1
        self.layers = []
        
        # Initialise each layer
        random_generator = np.random.default_rng()
        for ii in range(0, self.layer_count - 1):
            self.layers.append(Layer(network_structure[ii], network_structure[ii+1], hidden_layer_function.lower(), random_generator))
        self.layers.append(Layer(network_structure[-2], self.output_count, output_function.lower(), random_generator))

        # Select cost function to match output function:
        self.cost_function = COST_FUNCTION_SELECTION[output_function]


    # Method to load network from file
    def load_from_file(self, load_directory):

        self.layers = []
        for layer_data in load_directory.iterdir():
            self.layers.append(Layer(layer_data))
                
        self.input_count = self.layers[0].weights.shape[1]
        self.output_count = self.layers[-1].weights.shape[0]            
        self.layer_count = len(self.layers)

        # Select cost function to match output function:
        self.cost_function = COST_FUNCTION_SELECTION[self.layers[-1].output_function]


    # Method to feedforward an input through the network   
    def feedforward(self, input):

        # Check input dimensions
        if input.ndim == 1 and input.size == self.input_count:
           layer_output = input.reshape(1, input.size)
        elif input.ndim == 2 and input.shape[1] == self.input_count:
            layer_output = input.transpose()
        else:
            raise WrongNumberOfArguments("The dimension of the provided input(s) do not match the network's input dimension")

        # Perform feedforward computation
        for ii in range(0,self.layer_count):
            layer_output = self.layers[ii].feedforward(layer_output)

        # Return correct output shape
        if layer_output.ndim == 1:
            return layer_output
        elif layer_output.ndim == 2 and layer_output.shape[0] == 1:
            return layer_output.flatten()
        else:
            return layer_output.transpose()


    # Method to feedforward an input through the network and return outputs for all layers  
    def _feedforward_with_layer_cache(self, input):
        """
        _feedforward_with_layer_cache(self, inputs)

        Perform feedforward computation and return a tuple of outputs for each layer, including input layer
        """

        # Check input dimensions
        if input.ndim == 1 and input.size == self.input_count:
           layer_output = input.reshape(1, input.size)
        elif input.ndim == 2 and input.shape[1] == self.input_count:
            layer_output = input.transpose()
        else:
            raise WrongNumberOfArguments("The dimension of the provided input(s) do not match the network's input dimension")

        # Perform feedforward computation and cache results
        layer_output = input.transpose()
        cache = (layer_output,)
        for ii in range(0,self.layer_count):
            layer_output = self.layers[ii].feedforward(layer_output)
            cache = cache + (layer_output,)

        return cache


    # Method to unpack weights and biases
    def get_weights(self):

        biases = []
        weights = []

        for layer in self.layers:
            weights.append(layer.weights)
            biases.append(layer.biases)

        return weights, biases


    # Method to update weights and biases
    def update_weights(self, weights, biases):

        # Raise an error if the incorrect number of weight or bias matrices are provided
        if len(weights) != len(self.layers) or len(biases) != len(self.layers):
            raise ValueError('The number of weight and biases matrices provided much match the number of layers in the network')

        # Raise an error if the shape of any of the provided weights/bias matrices do not match the shape of network's corresponding matrix
        if (any([weights[ii].shape != self.layers[ii].weights.shape for ii in range(0, len(self.layers))]) or
            any([biases[ii].shape != self.layers[ii].biases.shape for ii in range(0, len(self.layers))])):
            raise ValueError('Shape of weight/bias matrices does not match the shape of the network weight/bias matrices')

        for ii in range(0, len(weights)):
            self.layers[ii].weights = weights[ii]
            self.layers[ii].biases = biases[ii]

        return weights, biases



    # Perform backpropagation
    def _backpropagation(self, inputs, targets):

        """
        Method to compute the average partial derivatives of the cost function with respect to each weight and bias in the network for a batch of training examples.

        """

        errors = []
        weights_gradients = []
        biases_gradients = []

        # 1. Feedforward through layers and cache the outputs
        model_output = self._feedforward_with_layer_cache(inputs)

        # 2. compute error in the neurons of the output layer
        errors.append(model_output[-1] - targets.transpose())

        # 3. compute average partial derivatives with respect to cost function of weights and biases in output layer
        weights_gradients.append((errors[0] @ model_output[-2].transpose())/inputs.shape[0])
        biases_gradients.append(np.sum(errors[0], axis=1, keepdims=True)/inputs.shape[0])
    
        # 4. Loop backwards through remaining layers
        for ii in range(2, self.layer_count+1):

            # a. compute the error in the neurons of the current layer
            errors.insert(0, (self.layers[-ii+1].weights.transpose() @ errors[0]) * ACTIVATIONS_DERIVATIVES[self.layers[-ii].output_function](model_output[-ii]))

            # b. compute average partial derivatives with respect to cost function of weights in the current layer
            weights_gradients.insert(0, (errors[0] @ model_output[-ii-1].transpose())/inputs.shape[0])
            biases_gradients.insert(0, np.sum(errors[0], axis=1, keepdims=True)/inputs.shape[0])

        return weights_gradients, biases_gradients


    def train(self, inputs, targets, training_parameters, optimiser_parameters):

        # Check required training parameters are provided and restore defaults if not
        default_training_parameters = {'optimiser': 'sgd', 'l2_param': 0}
        for parameter in default_training_parameters:
            if parameter not in training_parameters:
                training_parameters[parameter] = default_training_parameters[parameter]

        # train network
        trained_network = OPTIMISERS[training_parameters['optimiser']](self, inputs, targets, training_parameters, optimiser_parameters)
        self.layers = trained_network.layers


# save network to file
def save_network(network, save_directory, file_name):

    if not isinstance(save_directory, pathlib.Path):
        save_directory = pathlib.Path(save_directory)

    save_directory.mkdir(parents=False, exist_ok=True)

    with open(save_directory/f'{file_name}.csv', mode='w', newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter =',')
        network_structure = [layer.neuron_count for layer in network.layers]
        network_structure.insert(0, network.input_count)
        csv_writer.writerow(network_structure)
        csv_writer.writerow([network.layers[-2].output_function, network.layers[-1].output_function])
        csv_writer.writerow('')
        for layer in network.layers:
            np.savetxt(output_file, np.concatenate((layer.biases.reshape(layer.biases.size, 1), layer.weights), axis=1), delimiter =',', newline='\n')
            csv_writer.writerow([''])


# Load network from file
def load_network(file_name, save_directory=pathlib.Path.cwd()):

    if not isinstance(save_directory, pathlib.Path):
        save_directory = pathlib.Path(save_directory)

    with open(save_directory/f'{file_name}.csv', mode='r') as read_file:
        csv_data = csv.reader(read_file)
        file_data = list(csv_data)

    network_structure = [int(neuron_count) for neuron_count in file_data[0]]
    hidden_function = file_data[1][0]
    output_function = file_data[1][1]
    network = Network(network_structure, hidden_function, output_function)

    list_index = 3
    for ii in range(0, len(network_structure)-1):
        layer_data = np.asarray(file_data[list_index:list_index + network_structure[ii+1]]).astype(float)
        network.layers[ii].biases = layer_data[:,0].copy()[:, np.newaxis]
        network.layers[ii].weights = layer_data[:,1:].copy()
        list_index += network_structure[ii+1] + 1

    return network
