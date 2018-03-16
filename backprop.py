"""
Project: Neural Network

Back propagation and feed forward

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import numpy as np

def neuron(input, weights_neuron):
    """
    Compute the neuron output
    :param input:
    :param weights_neuron: weights for current neuron
    :return:
    """

    net = weights_neuron[0] + np.dot(weights_neuron[1:], input)

    # activation using sigmoid function
    neuron_output = 1.0 / (1.0 + np.exp(-net))

    return neuron_output


def layer(input, weights_layer):
    """
    Compute the outputs of each neuron of one layer
    :param input:
    :param weights_layer: [m, n], m = number of outputs, n = number of inputs
    :return:
    """

    (m, _) = np.shape(weights_layer)

    layer_output = [neuron(input, weights_layer[i, :]) for i in range(m)]

    return np.array(layer_output)


def error_loss(out_label, out_pred):

    return -out_label * np.log2(out_pred) - (1-out_label) * np.log2(1-out_pred)


def delta_j_out_unit(out_j, y_j):
    """

    :param out_j: prediction of the jth output neuron
    :param y_j: label of the jth output neuron
    :return:
    """

    return y_j - out_j


def delta_j_hidden_unit(out_j, deltas_k, weights_kj):
    """
    delta_j for hidden neurons
    :param out_j: prediction of current jth neuron
    :param deltas_k: delta for neurons in the next layer
    :param weights_kj: weights between neuron j in this layer to neuron k in next layer
    :return:
    """

    return out_j * (1-out_j) * np.dot(deltas_k, weights_kj)


def deltas_hidden(output_hidden, deltas_next_layer, weights_layer):

    num_neuron = len(output_hidden) # number of neurons in this hidden layer

    _deltas = []  # deltas_j for hidden units
    for j in range(num_neuron):
        weights_kj = weights_layer[:, j]
        _deltas.append(delta_j_hidden_unit(output_hidden[j], deltas_next_layer, weights_kj))

    return np.array(_deltas)


"""Define weights"""
weights_hidden = []
weights_hidden.append([ 1, 2,  3, -2, 1])
weights_hidden.append([ 2, 3,  1,  4, 1])
weights_hidden.append([-1, 1, -2,  0, 3])
weights_hidden = np.array(weights_hidden)

weights_output = np.array([[1, 3, 2, 1]])

"""Define input instance and output"""
input_instance = np.array([1, 3, 2, 1])
output = 1

"""Question 1"""
#Compute the hidden Layer
output_hidden = layer(input_instance, weights_hidden)

#Compute the predicted output
out_pred = layer(output_hidden, weights_output)

print("Outputs for h1, h2, h3: ", output_hidden[:])
print("Outputs for o: ", out_pred[0])


"""Question 2"""
#Compute delta_j for output unit
delta_out = delta_j_out_unit(out_pred[0], output)
deltas_hidden_neurons = deltas_hidden(output_hidden, delta_out, weights_output)
#deltas_input_neurons = deltas_hidden(input_instance, deltas_hidden_neurons, weights_hidden)

#print(deltas_input_neurons)
