"""
Project: Neural Network

main file to excute the logistic regression

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import scipy.io.arff as af
import logistic_func as lf
import data_proc as dp


""" Input parameters"""
l = 0.01
e = 3
filename_trn = './hw4/magic_train.arff'
filename_test = './hw4/magic_test.arff'

#l, e, filename_trn, filename_test = lf.read_cmdln_arg()


""" Load training and testing data"""
ins_set_trn, meta_data = af.loadarff(filename_trn)
ins_set_test, meta_data = af.loadarff(filename_test)
test_labels = [ins[-1] for ins in ins_set_test]


"""Training"""
# Data preprocessing, change the label to be 0 or 1
ins_set_trn = dp.data_preproc(ins_set_trn, meta_data)
ins_set_test = dp.data_preproc(ins_set_test, meta_data)

# compute the mean and sigma of training set for dataset standardization
mu, sigma = dp.instance_set_stat(ins_set_trn, meta_data)

# training with multiple epochs
weights = lf.multi_epochs_training(e, ins_set_trn, l, mu, sigma, meta_data)

# testing
_, _, _, F1 = lf.testset_prediction(ins_set_test, weights, mu, sigma, meta_data)
