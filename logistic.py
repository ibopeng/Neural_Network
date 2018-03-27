"""
Project: Neural Network

main file to excute the logistic regression

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import scipy.io.arff as af
import logistic_func as lf

""" Input parameters"""
l = 0.01
e = 3
filename_trn = './hw4/magic_train.arff'
filename_test = './hw4/magic_test.arff'


""" Load training and testing data"""
ins_set_trn, meta_data = af.loadarff(filename_trn)
ins_set_test, meta_data = af.loadarff(filename_test)
test_labels = [ins[-1] for ins in ins_set_test]

# relative parameters
var_ranges = [meta_data[name][1] for name in meta_data.names()]
num_vars = len(var_ranges[:-1])  # number of variables, excluding class variable
label_range = var_ranges[-1]


"""Training"""
# Data preprocessing, including discrete feature encoding, and dataset standardization
ins_set_trn = lf.data_preproc(ins_set_trn, meta_data)
ins_set_test = lf.data_preproc(ins_set_test, meta_data)

# compute the mean and sigma of training set for dataset standardization
mu, sigma = lf.instance_set_stat(ins_set_trn)

# training with multiple epochs
weights = lf.multi_epochs_training(e, ins_set_trn, l, num_vars, mu, sigma)


"""Testset prediction"""
_, _, _, F1 = lf.testset_prediction(ins_set_test, weights, mu, sigma)
