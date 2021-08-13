import numpy as np

# Activation Functions
def identity(input):
    """Compute identity neuron activation """

    return input


def sigmoid(input):
    """Compute sigmoid neuron activation """

    return 1/(1 + np.exp(-input))
          
def tanh(input):
    """ Compute Tanh neuron activation """

    return np.tanh(input)
      
def relu(input):
    """ Compute ReLu neuron activation """

    return np.maximum(0, input)

def softmax(input):
    """ Compute softmax neuron activation with numerical stability protection"""

    exps = np.exp(input - np.amax(input, axis=0))
    return exps/np.sum(exps, axis=0)


ACTIVATIONS = {'identity': identity,
               'sigmoid': sigmoid,
               'tanh': tanh,
               'relu': relu,
               'softmax': softmax}


# Activation function derivatives
def derivative_identity(identity_output):
    """Compute derivative of identity function """

    return np.ones(identity_output.shape)


def derivative_sigmoid(sigmoid_output):
    """ Compute derivative of sigmoid function """

    return sigmoid_output*(1 - sigmoid_output)
          

def derivative_tanh(tanh_output):
    """ Compute derivative of Tanh function """

    return 1 - tanh_output**2

       
def derivative_relu(relu_output):
    """ Compute derivative of ReLu function """

    relu_output[relu_output <= 0] = 0
    relu_output[relu_output > 0] = 1

    return  relu_output


def derivative_softmax(softmax_output):
    """ Compute softmax neuron activation """
    pass


ACTIVATIONS_DERIVATIVES = {'identity': derivative_identity,
                           'sigmoid': derivative_sigmoid,
                           'tanh': derivative_tanh,
                           'relu': derivative_relu,
                           'softmax': derivative_softmax}


# Cost functions
def mean_square_error(model_output, target):
    """ Compute mean square error """

    if model_output.shape != target.shape:
        raise ValueError('Dimensions of model output and target do not match')

    return np.sum(np.square(target - model_output))/model_output.size/2


def binary_cross_entropy(model_output, target):
    """ Compute binary cross-entropy cost """

    if model_output.shape != target.shape:
        raise ValueError('Dimensions of model output and target do not match')

    # Limit the model output to the interval (0,1) to prevent divide by zero
    eps = np.finfo(model_output.dtype).eps
    model_output = np.clip(model_output, eps, 1- eps)

    return -np.sum(target * np.log(model_output) + (1 - target)*np.log(1 - model_output))/model_output.shape[0]


def cross_entropy(model_output, target):
    """ Compute cross-entropy cost """

    if model_output.shape != target.shape:
        raise ValueError('Dimensions of model output and target do not match')
    
    # Limit the model output to the interval (0,1) to prevent divide by zero
    eps = np.finfo(model_output.dtype).eps
    model_output = np.clip(model_output, eps, 1- eps)

    return -np.sum(target * np.log(model_output))/model_output.shape[0]


COST_FUNCTIONS = {'mse': mean_square_error, 'binary_cross_entropy': binary_cross_entropy, 'cross_entropy':cross_entropy}
COST_FUNCTION_SELECTION = {'identity': mean_square_error, 'sigmoid': 'binary_cross_entropy', 'softmax': 'cross_entropy'}


# Cost function derivatives
def derivative_mean_square_error(model_output, target):
    """ Compute derivative of mean square error cost function"""

    
def derivative_cross_entropy(model_output, target):
    """ Compute derivative of cross-entropy cost function"""

    return model_output - target


COST_FUNCTION_DERIVATIVES = {'mse': derivative_mean_square_error, 'cross_entropy': derivative_cross_entropy}


# Regularisation Cost functions
def l2_cost(network, l2_param, targets):

    # get weights
    weights = network.get_weights()[0]

    l2_cost = 0
    for layer_weights in weights:
        l2_cost += 0.5 * l2_param * np.sum(layer_weights**2)/targets.shape[0]

    return l2_cost

# Regularisation Cost function derivatives
def l2_cost_derivative(network, l2_param, targets):

    # get weights
    weights = network.get_weights()[0]

    l2_cost_derivative = 0
    for layer_weights in weights:
        l2_cost_derivative += l2_param * np.sum(layer_weights)/targets.size

    return l2_cost_derivative

REGULARISATION_COST_FUNCTIONS = {'l2': l2_cost}
REGULARISATION_COST_FUNCTION_DERIVATIVES = {'l2': l2_cost_derivative}