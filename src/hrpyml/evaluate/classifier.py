import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from math import inf, ceil, floor, comb
from ..utilities.helpers import cast_to_1d_array, trapezoid_area

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


def binary_roc_curve(model_outputs, targets):

    # check input dimensions
    model_outputs = cast_to_1d_array(model_outputs)
    targets = cast_to_1d_array(targets)

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

    return {'tprs': true_positive_rates, 'fprs': false_positive_rates, 'accuracies': accuracies, 'thresholds': sorted_model_outputs.tolist()}


def roc_curve(model_outputs, targets, strategy='ovo'):
    """
    Full multi-class roc-curve generator. TO-DO
    """
    
    pass

    #if model_outputs.shape[1] <= 2:
    #    roc_curve = binary_roc_curve(model_outputs, targets)
    #else:

    #    # One vs one
    #    if strategy == 'ovo':
    #        roc_curves = []
    #        for ii in range(0, model_outputs.shape[1]):
    #            curves
    #            for jj in range(0, ii):
    #                roc_ii_jj = binary_roc_curve(model_outputs)
    #                roc_curves.append()



def plot_roc(roc_curve):
    fig, ax = plt.subplots()
    ax.plot(roc_curve['false_positive_rates'], roc_curve['true_positive_rates'])
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('False Positive rate')
    ax.set_ylabel('True Positive rate')
    ax.set_title('ROC Curve')
    plt.minorticks_on()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor', linestyle='--')
    return fig, ax


def binary_auroc(model_outputs, targets):

    # check input dimensions
    model_outputs = cast_to_1d_array(model_outputs)
    targets = cast_to_1d_array(targets)

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


def vuroc_hand_till(model_output, targets):
    """
    Computes volume under the receiver operating characteristic curve for classification problems using the method of Hand and Till (2001).
    """

    if model_output.shape != targets.shape:
        raise TypeError('Dimensions of predictions and targets must agree')

    # if binary classification extend target vector to include negative class
    if targets.ndim == 1:
        targets = np.stack((targets, 1-targets), axis=1)
        model_output = np.stack((model_output, 1-model_output), axis=1)
    elif targets.shape[1] == 1:
        targets = np.concatenate((targets, 1-targets), axis=1)
        model_output = np.concatenate((model_output, 1-model_output), axis=1)

    class_count = targets.shape[1]
    pairwise_aurocs = np.zeros((targets.shape[1], targets.shape[1]))
    pairwise_aurocs1 = np.zeros((targets.shape[1], targets.shape[1]))

    for ii in range(0, class_count):
        for jj in range(0, class_count):

            class_ii_mask = targets[:, ii] == 1
            class_jj_mask = targets[:, jj] == 1

            targets_ii_jj = targets[class_ii_mask | class_jj_mask, :]
            sort_order = np.argsort(model_output[class_ii_mask | class_jj_mask, ii])

            ranks = np.empty_like(sort_order)
            ranks[sort_order] = np.arange(sort_order.size) + 1

            class_ii_size = targets[class_ii_mask, :].shape[0]
            class_jj_size = targets[class_jj_mask, :].shape[0]


            """
            Obtaining the sum of class ii ranks can be done by indexing with either targets[:, jj] == 0 (i.e. NOT class jj) or targets[:, ii] == 1 (class ii).
            The off-diagonal elements (ii != jj) will be the same in both cases, but the diagonal elements (ii = jj) will be different, complementary values
            depending on the case used (i.e. p vs 1-p). AUROC is obtained from the upper-triangular elements ii < jj, thus final computed AUROC is unaffected
            by this choice.
            
            Considering the interpretation of the AUROC, the case ii = jj corresponds to computing the probability that a randomly selected member of class
            jj = ii will be ranked lower than another randomly selected member of the same class (ii = jj). Theory: The two possible complementary results stems from
            the arbitrary choice of which order to consider the two results?
            """

            pairwise_aurocs[ii, jj] = (np.sum(ranks[targets_ii_jj[:, ii] == 1]) - class_ii_size * (class_ii_size+1)/2)/class_ii_size/class_jj_size

            #pairwise_aurocs[ii, jj] = (np.sum(ranks[targets[:, jj] == 0]) - class_ii_size * (class_ii_size+1)/2)/class_ii_size/class_jj_size

            #pairwise_aurocs[ii, jj] = (np.sum(ranks[targets[:, ii] == 1]) - class_ii_size * (class_ii_size+1)/2)/class_ii_size/class_jj_size

            #tmp = targets[sort_order, jj]
            #pairwise_aurocs[ii, jj] = (np.sum(np.arange(tmp.size)[tmp == 0] + 1) - class_ii_size * (class_ii_size+1)/2)/class_ii_size/class_jj_size
           
    pairwise_aurocs = (pairwise_aurocs + pairwise_aurocs.transpose())/2
    
    return 2/class_count/(class_count - 1) * np.sum(np.triu(pairwise_aurocs, 1))



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


