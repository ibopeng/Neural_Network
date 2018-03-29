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
    if len(sys.argv) != 6:
        sys.exit("Incorrect arguments...")
    else:
        l = float(sys.argv[1])  # learning rate
        h = int(sys.argv[2])    # number of hidden nodes
        e = int(sys.argv[3])    # number of epochs for training
        filename_trn = str(sys.argv[4])  # training file
        filename_test = str(sys.argv[5])  # testing file

    return l, h, e, filename_trn, filename_test


def init_node_weights(num_input):

    # note that the bias weight should be considered
    # the 1st weight is for bias
    w = np.random.uniform(-0.01, 0.01, 1+num_input)

    return w


def neuron_output(inputs, weights_to_neuron):
    """
    Compute the neuron output
    :param input: input from last layer, maybe input layer or last hidden layer
    :param weights_to_neuron: a vector of weights coming to this neuron
    :return:
    """

#    net = weights_to_neuron[0] + np.dot(weights_to_neuron[1:], input)

    # note do not forget the bias node [1]
    net = np.dot([1] + list(inputs), weights_to_neuron)

    # activation using sigmoid function
    nout = 1.0 / (1.0 + np.exp(-net))

    return nout


def layer_output(inputs, weights_to_layer):
    """
    Compute the outputs of each neuron of one layer
    :param inputs:
    :param weights_to_layer: [m, n], m = number of outputs, n = number of inputs
    :return:
    """

    (m, _) = np.shape(weights_to_layer)

    lout = [neuron_output(inputs, weights_to_layer[i, :]) for i in range(m)]

    return lout


def delta_output_layer(labels, outputs):

    return [y_j - o_j for y_j in labels for o_j in outputs]


def deltas_hidden_neuron(out_hidden_neuron, deltas_output_layer, weights_to_output):
    """
    delta_j for hidden neurons
    :param out_hidden_neuron: prediction of current jth neuron
    :param deltas_output_layer: deltas for neurons in the next layer, deltas_k
    :param weights_to_output: weights vector between this current hidden neuron and each output nodes in next layer
    :return:
    """

    return out_hidden_neuron * (1-out_hidden_neuron) * np.dot(deltas_output_layer, weights_to_output)


def deltas_hidden_layer(out_hidden_layer, deltas_output_layer, weights_to_output):
    """

    :param out_hidden_layer:
    :param deltas_output_layer:
    :param weights_to_output: weights matrix between this hidden layer and next output layer
    :return:
    """

    num_neuron = len(out_hidden_layer) # number of neurons in this hidden layer

    _deltas = []  # deltas_j for hidden units
    for j in range(num_neuron):
        # weights between current single neuron j in this layer to EACH neuron k in next layer:
        # [weights_1j, weights_2j, ..., weights_kj]
        weights_j = weights_to_output[:, j]
        _deltas.append(deltas_hidden_neuron(out_hidden_layer[j], deltas_output_layer, weights_j))

    return _deltas


def cross_entropy_error(label, output):
    """
    use cross entropy to compute the error between true label and predicted output
    :param out_label:
    :param out_pred:
    :return:
    """
    return -label * np.log2(output) - (1.0-label) * np.log2(1.0-output)



def delta_wji(eta, deltas_this_layer, outputs_last_layer):
    """
    delta wji for updating the weights, j: index of this current layer, i: index of last layer
    :param eta:
    :param deltas_this_layer:
    :param outputs_last_layer:
    :return:

    """
    # ***********************************************************************
    # the shape of delta weights matrix generated here is [num_hid_nodes * num_inputs], the same as the weight matrix to be updated
    # every row of the delta weight matrix is the weights between all inputs nodes to one hidden/ouput node
    # ***********************************************************************

    # note that the bias node should be considered
    outputs_last_layer = [1] + list(outputs_last_layer)

    # loop over current layer
    d_wji = []
    for dj in deltas_this_layer:
        d_wji.append([eta * dj * oi for oi in outputs_last_layer])

    return np.array(d_wji)


def create_nnet(num_inputs, num_hid_nodes, num_outputs=1):
    """
    create a 3-layer neural network, 1: input layer, 2: hidden layer, 3: output layer (only 1 output)
    this step is actually initializing the weights
    :param num_inputs:
    :param num_hid_nodes:
    :param num_outputs:
    :return:
    """

    # ***********************************************************************
    # the shape of weights generated here is [num_hid_nodes * num_inputs]
    # every row of the weight matrix is the weights between all inputs nodes to one hidden/ouput node
    # ***********************************************************************

    # generate weights coming from input layer to each hidden node one by one
    weights_hidden = []
    for h in range(num_hid_nodes):
        weights_hidden.append(init_node_weights(num_inputs))

    # generate weights coming from hidden layer to the output layer
    weights_output = []
    for o in range(num_outputs):
        weights_output.append(init_node_weights(num_hid_nodes))

    return np.array(weights_hidden), np.array(weights_output)


def nnet_feedforward(instance, weights_hidden, weights_output):

    # ********** Feed forward ********** #
    # outputs at hidden layer
    out_hidden_layer = layer_output(instance, weights_hidden)

    # outputs at output layer
    out_output_layer = layer_output(out_hidden_layer, weights_output)

    return out_hidden_layer, out_output_layer


def nnet_backpropagate(labels, out_output_layer, out_hidden_layer, weights_output, weights_hidden, learning_rate, instance):

    # ********** Backpropagate ********** #
    # compute delta at output layer
    # since the operation is on list, so change label to be a list by [lable]
    delta_o = delta_output_layer(labels, out_output_layer)

    # compute delta at hidden layer
    delta_h = deltas_hidden_layer(out_hidden_layer, delta_o, weights_output)

    # compute delta w between hidden layer and output layer
    dw_o_h = delta_wji(learning_rate, delta_o, out_hidden_layer)
    # update weights
    weights_output = np.add(weights_output, dw_o_h)

    # compute delta w between input layer and hidden layer
    dw_h_i = delta_wji(learning_rate, delta_h, instance)
    weights_hidden = np.add(weights_hidden, dw_h_i)

    return weights_hidden, weights_output


def one_epoch_training(eta, instance_set, mu, sigma, weights_hidden, weights_output, meta_data):

    # *********** Training to update weights *********** #
    num_ins = len(instance_set)  # number of instances

    for i in range(num_ins):
        # preprocessing for current instance
        instance, label = dp.instance_preproc(instance_set[i], mu, sigma, meta_data)

        # Feedforward
        out_hidden_layer, out_output_layer = nnet_feedforward(instance, weights_hidden, weights_output)

        # Backpropagate
        weights_hidden, weights_output = nnet_backpropagate([label], out_output_layer, out_hidden_layer,
                                                            weights_output, weights_hidden,
                                                            eta, instance)

    # *********** Prediction for training set *********** #
    ins_set_pred = []  # prediction for instance set
    num_correct_pred = 0
    crs_ent_err = 0  # cross entropy error

    # use converged weights for prediction and cross entropy computation
    for i in range(num_ins):
        # preprocessing for current instance
        instance, label = dp.instance_preproc(instance_set[i], mu, sigma, meta_data)

        # ********** Feedforward ********** #
        out_hidden_layer, out_output_layer = nnet_feedforward(instance, weights_hidden, weights_output)

        # note that there is only one output node in this neural network
        # output_layer_out is a list, not a single numerical number
        output = out_output_layer[0]

        # compute the cross entropy error
        _err_ = cross_entropy_error(label, output)
        crs_ent_err = crs_ent_err + _err_

        # store the prediction class for each instance
        # the threshold = 0.5 is set for defining positive or negative class
        if output >= 0.5:
            ins_set_pred.append(1)
        else:
            ins_set_pred.append(0)

        if ins_set_pred[i] == label:
            num_correct_pred += 1

    num_mis_pred = num_ins - num_correct_pred

    return crs_ent_err, ins_set_pred, num_correct_pred, num_mis_pred, weights_hidden, weights_output


def multi_epochs_training(eta, num_hid_nodes, num_epochs, instance_set, mu, sigma, meta_data):

    # number of variables
    num_input_nodes = dp.input_dimension(meta_data)

    # create neural network
    weights_hidden, weights_output = create_nnet(num_input_nodes, num_hid_nodes, 1)

    
    for i in range(num_epochs):
    	# shuffle the data
    	random.shuffle(instance_set)
        crs_ent_err, _, num_correct_pred, num_mis_pred, weights_hidden, weights_output = one_epoch_training(eta, instance_set, mu, sigma, weights_hidden, weights_output, meta_data)
        print('{0}\t{1}\t{2}\t{3}'.format(i+1, crs_ent_err, num_correct_pred, num_mis_pred))

    return weights_hidden, weights_output


def testset_prediction(instance_set, mu, sigma, weights_hidden, weights_output, meta_data):

    num_ins = len(instance_set)  # number of instances in this dataset
    num_correct_pred = 0  # number of correctly predicted instances
    ins_set_pred = []  # prediction for instance set
    labels = []  # labels for the instance set

    TP = 0  # number of true postives that are also predicted postive

    # use converged weights for prediction and cross entropy computation
    for i in range(num_ins):
        # preprocessing for current instance
        instance, label = dp.instance_preproc(instance_set[i], mu, sigma, meta_data)

        # ********** Feedforward ********** #
        out_hidden_layer, out_output_layer = nnet_feedforward(instance, weights_hidden, weights_output)
        # store the label for this instance
        labels.append(label)

        # note that there is only one output node in this neural network
        # output_layer_out is a list, not a single numerical number
        output = out_output_layer[0]

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




