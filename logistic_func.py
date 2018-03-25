"""
Project: Neural Network

Logistic regression using a neural network

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import numpy as np
import sys
import random

def read_cmdln_arg():
    """command line operation"""
    if len(sys.argv) != 4:
        sys.exit("Incorrect arguments...")
    else:
        l = float(sys.argv[1])  # learning rate
        e = int(sys.argv[2])  # number of epochs for training
        filename_trn = str(sys.argv[3])  # training file
        filename_test = str(sys.argv[4])  # testing file

    return l, e, filename_trn, filename_test


def init_weights(num_input):
    """
    randomize the initial weights between input layer and the output
    :param num_input: number of nodes in layer_i
    :return:
    """

    # note that the bias weight should be considered
    # the 1st weight is for bias
    w = [np.random.uniform(-0.01, 0.01) for j in range(1+num_input)]

    return w


def neuron_output(input, weights_to_neuron):
    """
    Compute the neuron output
    :param input: input from last layer, maybe input layer or last hidden layer
    :param weights_to_neuron: weights coming to this neuron
    :return:
    """

    net = weights_to_neuron[0] + np.dot(weights_to_neuron[1:], input)

    # activation using sigmoid function
    nout = 1.0 / (1.0 + np.exp(-net))

    return nout


def cross_entropy_error(out_label, out_pred):
    """
    use cross entropy to compute the error between true label and predicted output
    :param out_label:
    :param out_pred:
    :return:
    """
    return -out_label * np.log2(out_pred) - (1.0-out_label) * np.log2(1.0-out_pred)


def delta_output(output, label):
    """

    :param out_j: prediction of the output node
    :param y_j: label of the output node
    :return:
    """

    return label - output


def delta_w(eta, delta_out, inputs):
    """
    Compute delta w between input nodes and output nodes
    :param eta:
    :param delta_out:
    :param inputs:
    :return:
    """

    # loop over current layer
    dw = [eta * delta_out * inp for inp in inputs]

    return dw


def data_preproc(instance_set, meta_data):
    """
    Change the label into numerical data
    :param instance_set:
    :param meta_data:
    :return:
    """

    var_ranges = [meta_data[name][1] for name in meta_data.names()]

    label_range = var_ranges[-1]

    instance_set_new = []
    for ins in instance_set:
        ins_new = list(ins)
        lb = ins_new.pop()
        if lb == label_range[0]:
            ins_new.append(0)
        else:
            ins_new.append(1)
        instance_set_new.append(ins_new)

    return instance_set_new


def instance_normalization(instance):

    instance = np.array(instance)

    ins_mean = np.mean(instance)
    ins_std = np.std(instance) + 0.0000001

    instance = (instance - ins_mean) / ins_std

    return instance


def one_epoch_training(instance_set, weights, eta):

    ################## Training to update weights #######################
    num_ins = len(instance_set)  # number of instances
    for i in range(num_ins):
        instance = instance_set[i][:-1]
        label = instance_set[i][-1]

        # instance normalization
        instance = instance_normalization(instance)

        # get the output for current instance
        out_pred = neuron_output(instance, weights)

        # compute delta out
        delta_out = delta_output(out_pred, label)

        # compute delta w to update weights
        # note that Do NOT forget to add bias node [1] to instance
        dw = delta_w(eta, delta_out, [1] + list(instance))

        # update weights
        for k in range(len(weights)):
            weights[k] = weights[k] + dw[k]


    ###################### Prediction and Cross Entropy for this epoch ###################
    pred = []  # prediction for each instance
    num_correct_pred = 0
    cn_err = 0  # cross entropy error
    # use converged weights for prediction and cross entropy computation
    for i in range(num_ins):
        instance = instance_set[i][:-1]
        label = instance_set[i][-1]

        # instance normalization
        instance = instance_normalization(instance)

        # get the output for current instance
        out_pred = neuron_output(instance, weights)

        # compute the cross entropy error
        cn_err = cn_err + cross_entropy_error(label, out_pred)

        # store the prediction class for each instance
        if out_pred >= 0.5:
            pred.append(1)
        else:
            pred.append(0)

        if pred[i] == label:
            num_correct_pred += 1

    num_mis_pred = num_ins - num_correct_pred

    return cn_err, pred, num_correct_pred, num_mis_pred, weights


def multi_epochs_training(num_epochs, instance_set, eta, num_vars):

    # shuffle the data
    random.shuffle(instance_set)

    # initialize weights
    weights = init_weights(num_vars)

    for i in range(num_epochs):
        cn_err, pred, num_correct_pred, num_mis_pred, weights = one_epoch_training(instance_set, weights, eta)
        print(i+1, cn_err, num_correct_pred, num_mis_pred)



