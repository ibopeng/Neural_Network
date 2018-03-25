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
#    w = [np.random.uniform(-0.01, 0.01) for j in range(1+num_input)]
    w = np.random.uniform(-0.01, 0.01, 1+num_input)

    return w


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


def instance_set_stat(instance_set):

    # extract instance data, not including label
    instance_data = []
    for ins in instance_set:
        instance_data.append(ins[:-1])

    # compute mean of this dataset
    mu = np.mean(instance_data, axis=0)
    sigma = np.std(instance_data, axis=0)

    return mu, sigma


def instance_normalization(instance):

    instance = np.array(instance)

    ins_mean = np.mean(instance)
    ins_std = np.std(instance) + 0.0000001

    instance = (instance - ins_mean) / ins_std

    return instance


def one_epoch_training(instance_set, weights, eta, mu, sigma):

    ################## Training to update weights #######################
    num_ins = len(instance_set)  # number of instances
    for i in range(num_ins):
        instance = instance_set[i][:-1]
        label = instance_set[i][-1]

        # instance standardization
        # 0.00000001 is used for avoiding zero standard deviation
        instance = np.divide(np.subtract(instance, mu), np.add(sigma, 0.00000001))

        # instance normalization
#        instance = instance_normalization(instance)

        # get the output for current instance
        output = neuron_output(instance, weights)

        # compute delta out
        delta_out = delta_output(label, output)

        # compute delta w to update weights
        dw = delta_w(eta, delta_out, instance)

        # update weights
        for k in range(len(weights)):
            weights[k] = weights[k] + dw[k]


    ###################### Prediction and Cross Entropy for this epoch ###################
    ins_set_pred = []  # prediction for instance set
    num_correct_pred = 0
    cross_entropy_err = 0  # cross entropy error
    # use converged weights for prediction and cross entropy computation
    for i in range(num_ins):
        instance = instance_set[i][:-1]
        label = instance_set[i][-1]

        # instance standardization
        # 0.00000001 is used for avoiding zero standard deviation
        instance = np.divide(np.subtract(instance, mu), np.add(sigma, 0.00000001))

        # instance normalization
#        instance = instance_normalization(instance)

        # get the output for current instance
        output = neuron_output(instance, weights)

        # compute the cross entropy error
        _err_ = cross_entropy_error(label, output)
        cross_entropy_err = cross_entropy_err + _err_

        # store the prediction class for each instance
        if output >= 0.5:
            ins_set_pred.append(1)
        else:
            ins_set_pred.append(0)

        if ins_set_pred[i] == label:
            num_correct_pred += 1

    num_mis_pred = num_ins - num_correct_pred

    return cross_entropy_err, ins_set_pred, num_correct_pred, num_mis_pred, weights


def multi_epochs_training(num_epochs, instance_set, eta, num_vars, mu, sigma):

    # initialize weights
    weights = init_weights(num_vars)

    for i in range(num_epochs):
        # shuffle the data
        random.shuffle(instance_set)
        cn_err, pred, num_correct_pred, num_mis_pred, weights = one_epoch_training(instance_set, weights, eta, mu, sigma)
        print('{0}\t{1}\t{2}\t{3}'.format(i+1, cn_err, num_correct_pred, num_mis_pred))

    return weights


def testset_prediction(instance_set_test, weights, mu, sigma):

    num_ins = len(instance_set_test)
    num_correct_pred = 0
    prediction = []
    label = []

    TP = 0  # number of true postive instances that are also predicted postive

    # use converged weights for prediction and cross entropy computation
    for i in range(num_ins):
        instance = instance_set_test[i][:-1]
        label.append(instance_set_test[i][-1])

        # instance standardization
        # 0.00000001 is used for avoiding zero standard deviation
        instance = np.divide(np.subtract(instance, mu), np.add(sigma, 0.00000001))

        # instance normalization
#        instance = instance_normalization(instance)

        # get the output for current instance
        output = neuron_output(instance, weights)

        # store the prediction class for each instance
        if output >= 0.5:
            prediction.append(1)
        else:
            prediction.append(0)

        if prediction[i] == label[i]:
            num_correct_pred += 1

        # number of true postive instances that are also predicted postive
        if prediction[i] == 1 and label[i] == 1:
            TP = TP + 1

        print('{0:.9f}\t{1}\t{2}'.format(output, prediction[i], label[i]))

    # number of mis_prediction
    num_mis_pred = num_ins - num_correct_pred
    print('{0}\t{1}'.format(num_correct_pred, num_mis_pred))

    # compute recall and precision
    num_pos_pred = np.sum(prediction)  # number of predicted postives
    num_pos_true = np.sum(label)  # number of true postives
    recall = 1.0 * TP / num_pos_true
    precision = 1.0 * TP / num_pos_pred
    # compute F1 score
    F1 = 2.0 / (1.0/recall + 1.0/precision)
    print(F1)




