import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from math import inf, ceil, floor
from ..utilities.helpers import check_1d_array, trapezoid_area

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def get_confusion_matrix(predicted_classes, actual_classes):

    """
    get_confusion_matrix(predicted_classes, actual_classes)

    Obtain the confusion matrix as a numpy array for a data set by providing predicted and actual classes for each sample.
    Classes should be encoded as the positive integers 0,1,2,3,...
    """

    if predicted_classes.shape[0] != actual_classes.shape[0]:
        raise TypeError('Dimensions of predictions and targets must agree')

    classes = np.unique(actual_classes)
    class_count = classes.size

    confusion_matrix = np.zeros((class_count, class_count))

    for predicted_class, actual_class in zip(predicted_classes, actual_classes):
        confusion_matrix[actual_class, predicted_class] += 1

    return confusion_matrix


def get_tpr_fpr(confusion_matrix):
    """
    get_tpr_fpr(confusion_matrix)

    Given the confusion matrix for a binary classifier, return the tpr (true positive rate, aka sensitivity/recall) and the
    fpr (false positive rate).
    """

    if not isinstance(confusion_matrix, np.ndarray):
        raise TypeError('Please supply confusion matrix as numpy array')
    
    if not confusion_matrix.shape == (2,2):
        raise ValueError('Confusion matrix must be a 2x2 array')

    tpr = confusion_matrix[1,1]/(confusion_matrix[1,1] + confusion_matrix[1,0])
    fpr = confusion_matrix[0,1]/(confusion_matrix[0,1] + confusion_matrix[0,0])

    return tpr, fpr


def get_accuracy(confusion_matrix):
    """
    get_accuracy(confusion_matrix)

    Given the confusion matrix for a binary classifier, return the accuracy
    """

    return (confusion_matrix[1,1] + confusion_matrix[0,0])/np.sum(confusion_matrix)


def roc_curve(model_outputs, targets, plot=True):

    # check input dimensions
    model_outputs = check_1d_array(model_outputs)
    targets = check_1d_array(targets)

    # Get the number of instances in each class
    classes, class_counts = np.unique(targets, return_counts=True)

    if classes.size < 2:
        raise ValueError('Require at least one point in each class')

    negatives_count = class_counts[0]
    positives_count = class_counts[1]

    # Sort model data by decreasing score
    sorted_indicies = np.argsort(model_outputs)[::-1]
    sorted_model_outputs = model_outputs[sorted_indicies]
    sorted_targets = targets[sorted_indicies]

    # Initialise algorithm variables
    true_positives = 0
    false_positives = 0
    true_positive_rates = []
    false_positive_rates = []
    accuracies = []
    prev_model_output = -inf

    for ii in range(0, sorted_model_outputs.size):

        # Only push roc point to curve after all instances with the same score have been processed
        if sorted_model_outputs[ii] != prev_model_output:
            true_positive_rates.append(true_positives/positives_count)
            false_positive_rates.append(false_positives/negatives_count)
            prev_model_output = sorted_model_outputs[ii]
            accuracies.append((true_positives + (negatives_count - false_positives))/model_outputs.size)

        if sorted_targets[ii] == 1:
            true_positives += 1
        else:
            false_positives += 1

    # push the point (1,1) to the curve
    true_positive_rates.append(true_positives/positives_count)
    false_positive_rates.append(false_positives/negatives_count)
    accuracies.append((true_positives + (negatives_count - false_positives))/model_outputs.size)

    # Plot ROC curve
    if plot:
        fig, ax = plt.subplots()
        ax.plot(false_positive_rates, true_positive_rates)
        ax.plot([0,1], [0,1], 'k--')
        ax.set_xlabel('False Positive rate')
        ax.set_ylabel('True Positive rate')
        ax.set_title('ROC Curve')
        plt.minorticks_on()
        plt.grid(b=True, which='major')
        plt.grid(b=True, which='minor', linestyle='--')

    return true_positive_rates, false_positive_rates, accuracies, sorted_model_outputs


def auroc(model_outputs, targets):

    # check input dimensions
    model_outputs = check_1d_array(model_outputs)
    targets = check_1d_array(targets)

     # Get the number of instances in each class
    classes, class_counts = np.unique(targets, return_counts=True)

    if classes.size < 2:
        raise ValueError('Require at least one point in each class')

    negatives_count = class_counts[0]
    positives_count = class_counts[1]

    # Sort model data by decreasing score
    sorted_indicies = np.argsort(model_outputs)[::-1]
    sorted_model_outputs = model_outputs[sorted_indicies]
    sorted_targets = targets[sorted_indicies]

    # Initialise algorithm variables
    true_positives = 0
    false_positives = 0
    prev_true_positives = 0
    prev_false_positives = 0
    auroc = 0
    prev_model_output = -inf

    for ii in range(0, sorted_model_outputs.size):

        # Only update auroc after all instances with the same score have been processed
        if sorted_model_outputs[ii] != prev_model_output:
            auroc += trapezoid_area(false_positives, prev_false_positives, true_positives, prev_true_positives)
            prev_model_output = sorted_model_outputs[ii]
            prev_true_positives = true_positives
            prev_false_positives = false_positives

        # Increment true_positives or false_positives count by considering the true class of the current data point
        if sorted_targets[ii] == 1:
            true_positives += 1
        else:
            false_positives += 1

    # final auroc update
    auroc += trapezoid_area(negatives_count, prev_false_positives, positives_count, prev_true_positives)

    # scale auroc from area of (positives_count x negatives_count) to unit square:
    auroc = auroc/(positives_count * negatives_count)

    return auroc


def auroc_hand_till(model_output, targets):

    if model_output.shape[0] != targets.shape[0]:
        raise TypeError('Dimensions of predictions and targets must agree')

    positive_class_indicies = np.arange(0, targets.shape[0])[targets == 1]
    negative_class_indicies = np.arange(0, targets.shape[0])[targets == 0]

    positive_class_count = positive_class_indicies.shape[0]
    negative_class_count = negative_class_indicies.shape[0]

    adjusted_model_output = model_output.copy()
    #adjusted_model_output[negative_class_indicies] = 1 - adjusted_model_output[negative_class_indicies]

    sorted_outputs_indices = np.argsort(adjusted_model_output, axis=None)
    sorted_outputs = adjusted_model_output[sorted_outputs_indices]

    auroc = (np.sum(sorted_outputs_indices[positive_class_indicies] + 1) - positive_class_count*(positive_class_count + 1)/2)/positive_class_count/negative_class_count
    return auroc


def choose_threshold(tprs, fprs, accuracies, thresholds, tpr_select=-inf, fpr_select=inf):
    
    output = {}
    best_accuracy = 0
    min_fpr = 1
    max_tpr = 0

    for tpr, fpr, accuracy, threshold in zip(tprs, fprs, accuracies, thresholds):
        
        # Set tpr and minimise fpr
        if tpr_select != -inf and tpr >= tpr_select and fpr < min_fpr:
            actual_tpr = tpr
            min_fpr = fpr
            tpr_select_accuracy = accuracy
            tpr_select_threshold = threshold
            output['tpr_min_fpr'] = (tpr_select_threshold, actual_tpr, min_fpr, tpr_select_accuracy)

         # Set fpr and maximise tpr
        if fpr_select != inf and fpr <= fpr_select and tpr > max_tpr:
            actual_fpr = fpr
            max_tpr = tpr
            fpr_select_accuracy = accuracy
            fpr_select_threshold = threshold
            output['fpr_max_tpr'] = (fpr_select_threshold, max_tpr, actual_fpr, fpr_select_accuracy)

        # max accuracy:
        if accuracy > best_accuracy:
            accuracy_tpr = tpr
            accuracy_fpr = fpr
            best_accuracy = accuracy
            accuracy_threshold = threshold
            output['max_accuracy'] = (accuracy_threshold, accuracy_tpr, accuracy_fpr, best_accuracy)

    if len(output) == 1:
        return output['max_accuracy']
    else:
        return output
    

def plot_2d_decision_boundary(network, inputs, targets, boundary = 0.5):

    # Set min and max values and give it some padding
    x1_min, x1_max = inputs[:, 0].min() - 0.6, inputs[:, 0].max() + 0.6
    x2_min, x2_max = inputs[:, 1].min() - 0.6, inputs[:, 1].max() + 0.6
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    # Predict the function value for the whole grid
    model_outputs = network.feedforward(np.stack((xx.ravel(), yy.ravel()), 1))
    model_outputs = model_outputs.reshape(xx.shape)
    predicted_classes = model_outputs > boundary
    levels = np.array([boundary])

    # Plot the contour and training examples
    fig, ax = plt.subplots()
    #plt.contour(xx, yy, predicted_classes, 1, colors='black', linestyles='dashed')
    plt.contourf(xx, yy, predicted_classes, 1, cmap=plt.cm.RdBu, alpha=0.8, extend='both')
    plt.scatter(inputs[:,0], inputs[:,1], c=targets, cmap=plt.cm.RdBu)
    ax.set_xlim(0.5*ceil(xx.min()/0.5), 0.5*floor(xx.max()/0.5))
    ax.set_ylim(0.5*ceil(yy.min()/0.5), 0.5*floor(yy.max()/0.5))
    plt.title(f'Threshold: {boundary}')


