"""
Project: Neural Network

Data preprocessing

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import numpy as np


def data_preproc(instance_set, meta_data):
    """
    Change the label into numerical data
    :param instance_set:
    :param meta_data:
    :return:
    """

    label_name = meta_data.names()[-1]
    label_range = meta_data[label_name][1]

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


def input_dimension(meta_data):

    num_vars = len(meta_data.names()) - 1
    var_ranges = [meta_data[name][1] for name in meta_data.names()]
    num_input_nodes = 0
    for i in range(num_vars):
        vr = var_ranges[i]
        if vr is None:
            num_input_nodes = num_input_nodes + 1
        else:
            num_input_nodes = num_input_nodes + len(vr)

    return num_input_nodes


def instance_standardization(instance, mu, sigma):
    """
    standardize instance for numeric variables
    :param instance: single instance without class label
    :param mu:
    :param sigma:
    :return:
    """

    for k in range(len(instance)):
        if mu[k] is not None:  # numeric variable
            instance[k] = (instance[k] - mu[k]) / sigma[k]

    return instance


def instance_set_stat(instance_set, meta_data):

    var_types = [meta_data[name][0] for name in meta_data.names()]  # type of variables, numerical or nominal/discrete
    num_vars = len(var_types) - 1  # number of variables, excluding the class variable

    # compute mean of this dataset, only process numeric variable
    mu = []
    sigma = []
    for i in range(num_vars):
        if var_types[i] == 'numeric':  # numeric variable
            # get all instance values on the ith variable
            val_i = [ins[i] for ins in instance_set]
            # compute mean and std
            mu.append(np.mean(val_i))
            sigma.append(np.std(val_i) + 0.000000001)
        else:  # nominal variable
            mu.append(None)
            sigma.append(None)

#    mu = np.mean(instance_data, axis=0)
#    sigma = np.std(instance_data, axis=0)

    return mu, sigma


def instance_encoding(instance, meta_data):
    """
    Change the label into numerical data
    :param instance_set:
    :param meta_data:
    :return:
    """

    var_ranges = [meta_data[name][1] for name in meta_data.names()]  # range of variables
    var_types = [meta_data[name][0] for name in meta_data.names()]  # type of variables, numerical or nominal/discrete
    num_vars = len(var_types) - 1  # number of variables, excluding the class variable

    # one of k encoding for nominal/discrete variables
    ins_new = []
    for k in range(num_vars):
        if var_types[k] == 'numeric':  # numeric variable
            ins_new.append(instance[k])
        else:  # nominal variable
            var_range_length = len(var_ranges[k])
            var_val = [0] * var_range_length
            idx = var_ranges[k].index(instance[k])
            var_val[idx] = 1
            ins_new += var_val  # var_val is a list, append it to the new ins

    return ins_new


def F1_score(ins_set_pred, labels, TP):

    num_pos_pred = np.sum(ins_set_pred)  # number of predicted postives
    num_pos_true = np.sum(labels)  # number of true postives
    recall = 1.0 * TP / num_pos_true
    precision = 1.0 * TP / num_pos_pred

    # compute F1 score
#    F1 = 2.0 / (1.0/recall + 1.0/precision)
    F1 = 2.0 * (recall * precision) / (recall + precision)

    return F1, recall, precision