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

    return layer_output


def deltas_out_layer(outputs, labels):
    """

    :param out_j: prediction of the output layer
    :param y_j: label of the output layer
    :return:
    """

    return [y_j - out_j for y_j in labels for out_j in outputs]


def deltas_hidden_neuron(output, deltas, weights):
    """
    delta_j for hidden neurons
    :param output: prediction of current jth neuron
    :param deltas: delta for neurons in the next layer, deltas_k
    :param weights: weights between current single neuron j in this layer to each neuron k in next layer: weights_kj
    :return:
    """

    return output * (1-output) * np.dot(deltas, weights)


def deltas_hidden_layer(output_hidden, deltas_next_layer, weights_layer):

    num_neuron = len(output_hidden) # number of neurons in this hidden layer

    _deltas = []  # deltas_j for hidden units
    for j in range(num_neuron):
        # weights between current single neuron j in this layer to each neuron k in next layer:
        # [weights_1j, weights_2j, ..., weights_kj]
        weights_j = weights_layer[:, j]
        _deltas.append(deltas_hidden_neuron(output_hidden[j], deltas_next_layer, weights_j))

    return _deltas


def delta_wji(eta, deltas, outputs):

    # loop over current layer
    d_wji = []
    for dj in deltas:
        d_wji.append([eta * dj * oi for oi in outputs])

    return d_wji




"""Define weights"""
weights_hidden = [[ 1, 2,  3, -2, 1],
                  [ 2, 3,  1,  4, 1],
                  [-1, 1, -2,  0, 3]]
weights_hidden = np.array(weights_hidden)

weights_output = np.array([[1, 3, 2, 1]])

"""Define input instance and output"""
input_instance = [1, 3, 2, 1]
output = [1]

"""Question 1"""
#Compute the hidden Layer
output_hidden = layer(input_instance, weights_hidden)

#Compute the predicted output
out_pred = layer(output_hidden, weights_output)

print('Question 1:')
print('Outputs for h1, h2, h3: ', output_hidden)
print('Outputs for o: ', out_pred)


"""Question 2"""
#Compute delta_j for output unit
delta_out = deltas_out_layer(out_pred, output)
deltas_hidden_neurons = deltas_hidden_layer(output_hidden, delta_out, weights_output)
deltas_input_neurons = deltas_hidden_layer(input_instance, deltas_hidden_neurons, weights_hidden)

print('\nQuestion 2:')
print('delta of output unit', delta_out)
print('delta of hidden unit', deltas_hidden_neurons)
print('delta of input unit', deltas_input_neurons)

"""Question 3"""
# learning rate
eta = 1
# weights going to output neurons
output_dw = delta_wji(eta, delta_out, [1] + output_hidden)  # [1] is the bias neuron
hidden_dw = delta_wji(eta, deltas_hidden_neurons, [1] + input_instance)

print('\nQuestion 3:')
print("delta wji of output layer: ", output_dw)
print("delta wji of hidden layer: ", hidden_dw)