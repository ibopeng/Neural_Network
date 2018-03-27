"""
Project: Neural Network

Logistic regression using a neural network

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import numpy as np
import sys
import random
import data_proc as dp


def read_cmdln_arg():
    """command line operation"""
    if len(sys.argv) != 5:
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
#    w = [np.random.uniform(-0.01, 0.01) for j in range(1+num_input)]
    w = np.random.uniform(-0.01, 0.01, 1+num_input)

    return np.array(w)


def neuron_output(input, weights_to_neuron):
    """
    Compute the neuron output
    :param input: input from last layer, maybe input layer or last hidden layer
    :param weights_to_neuron: weights coming to this neuron
    :return:
    """

#    net = weights_to_neuron[0] + np.dot(weights_to_neuron[1:], input)
    net = np.dot([1] + list(input), weights_to_neuron)

    # activation using sigmoid function
    nout = 1.0 / (1.0 + np.exp(-net))

    return nout


def cross_entropy_error(label, output):
    """
    use cross entropy to compute the error between true label and predicted output
    :param out_label:
    :param out_pred:
    :return:
    """
    return -label * np.log2(output) - (1.0-label) * np.log2(1.0-output)


def delta_output(label, output):
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
    # note that Do NOT forget to add bias node [1] to instance
    dw = [eta * delta_out * inp for inp in [1] + list(inputs)]

    return np.array(dw)


def logistic_output(instance, mu, sigma, weights, meta_data):

    # instance standardization
    instance = dp.instance_standardization(instance, mu, sigma)

    # instance encoding
    instance = dp.instance_encoding(instance, meta_data)

    # get the output for current instance
    output = neuron_output(instance, weights)

    return output


def one_epoch_training(instance_set, weights, eta, mu, sigma, meta_data):

    label_name = meta_data.names()[-1]
    label_range = meta_data[label_name][1]

    # *********** Training to update weights *********** #
    num_ins = len(instance_set)  # number of instances
    for i in range(num_ins):
        instance = instance_set[i][:-1]
        label = instance_set[i][-1]

        # ********** Feedforward ********** #
        output = logistic_output(instance, mu, sigma, weights, meta_data)

        # compute delta out
        delta_out = delta_output(label, output)

        # compute delta w to update weights
        dw = delta_w(eta, delta_out, instance)

        # update weights
        weights = weights + dw
#        for k in range(len(weights)):
#            weights[k] = weights[k] + dw[k]


    # *********** Prediction and Cross Entropy for this epoch *********** #
    ins_set_pred = []  # prediction for instance set
    num_correct_pred = 0
    crs_ent_err = 0  # cross entropy error
    # use converged weights for prediction and cross entropy computation
    for i in range(num_ins):
        instance = instance_set[i][:-1]
        label = instance_set[i][-1]

        # ********** Feedforward ********** #
        output = logistic_output(instance, mu, sigma, weights, meta_data)

        # compute the cross entropy error
        _err_ = cross_entropy_error(label, output)
        crs_ent_err = crs_ent_err + _err_

        # store the prediction class for each instance
        if output >= 0.5:
            ins_set_pred.append(1)
        else:
            ins_set_pred.append(0)

        if ins_set_pred[i] == label:
            num_correct_pred += 1

    num_mis_pred = num_ins - num_correct_pred

    return crs_ent_err, ins_set_pred, num_correct_pred, num_mis_pred, weights


def multi_epochs_training(num_epochs, instance_set, eta, mu, sigma, meta_data):

    # number of variables
    num_input_nodes = dp.input_dimension(meta_data)

    # initialize weights
    weights = init_weights(num_input_nodes)

    for i in range(num_epochs):
        # shuffle the data
        random.shuffle(instance_set)
        crs_ent_err, _, num_correct_pred, num_mis_pred, weights = one_epoch_training(instance_set, weights, eta, mu, sigma, meta_data)
        print('{0}\t{1}\t{2}\t{3}'.format(i+1, crs_ent_err, num_correct_pred, num_mis_pred))

    return weights





def testset_prediction(instance_set_test, weights, mu, sigma, meta_data):

    num_ins = len(instance_set_test)
    num_correct_pred = 0
    ins_set_pred = []
    labels = []

    TP = 0  # number of true postive instances that are also predicted postive

    # use converged weights for prediction and cross entropy computation
    for i in range(num_ins):
        instance = instance_set_test[i][:-1]
        label = instance_set_test[i][-1]

        # ********** Feedforward ********** #
        output = logistic_output(instance, mu, sigma, weights, meta_data)

        # store the label for this instance
        labels.append(label)

        # store the prediction class for each instance
        if output >= 0.5:
            ins_set_pred.append(1)
        else:
            ins_set_pred.append(0)

        if ins_set_pred[i] == labels[i]:
            num_correct_pred += 1

        # number of true postive instances that are also predicted postive
        if ins_set_pred[i] == 1 and labels[i] == 1:
            TP = TP + 1

        print('{0:.9f}\t{1}\t{2}'.format(output, ins_set_pred[i], labels[i]))

    # number of mis_prediction
    num_mis_pred = num_ins - num_correct_pred
    print('{0}\t{1}'.format(num_correct_pred, num_mis_pred))

    # compute recall and precision
    F1, recall, precision = dp.F1_score(ins_set_pred, labels, TP)
    print(F1)

    return ins_set_pred, recall, precision, F1




