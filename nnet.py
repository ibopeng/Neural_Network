"""
Project: Neural Network

main file to excute the logistic regression

@ Author: Bo Peng
@ University of Wisconsin - Madison
"""

import scipy.io.arff as af
import nnet_func as nf
import data_proc as dp

""" Input parameters"""
#l = 0.001
#h = 9
#e = 10
#filename_trn = 'magic_train.arff'
#filename_test= 'magic_test.arff'

l, h, e, filename_trn, filename_test = nf.read_cmdln_arg()


# Load training and testing data
ins_set_trn, meta_data = af.loadarff(filename_trn)
ins_set_test, meta_data = af.loadarff(filename_test)

# Data preprocessing, change the label to be 0 or 1
ins_set_trn = dp.data_preproc(ins_set_trn, meta_data)
ins_set_test = dp.data_preproc(ins_set_test, meta_data)

# compute the mean and sigma of training set for dataset standardization
mu, sigma = dp.instance_set_stat(ins_set_trn, meta_data)

# training with multiple epochs
weights_hidden, weights_output = nf.multi_epochs_training(l, h, e, ins_set_trn, mu, sigma, meta_data)

# testing
_, _, _, F1 = nf.testset_prediction(ins_set_test, mu, sigma, weights_hidden, weights_output, meta_data)