from sklearn.preprocessing import scale
import scipy.io.arff as af
import data_proc as dp
import numpy as np

# ******************** Input parameters ******************** #
l = 0.1
e = 10
filename_trn = 'diabetes_train.arff'
filename_test = 'diabetes_test.arff'

#l, e, filename_trn, filename_test = lf.read_cmdln_arg()


# ******************** Load data ******************** #
ins_set_trn, meta_data = af.loadarff(filename_trn)
ins_set_test, meta_data = af.loadarff(filename_test)


# Data preprocessing, change the label to be 0 or 1
ins_set_trn = dp.data_preproc(ins_set_trn, meta_data)
ins_set_test = dp.data_preproc(ins_set_test, meta_data)

# compute the mean and sigma of training set for dataset standardization
mu, sigma = dp.instance_set_stat(ins_set_trn, meta_data)

ins_set_trn_sc = np.array(ins_set_trn)
ins_set_trn_sc = scale(ins_set_trn_sc)

for ins in ins_set_trn:
    for k in range(len(ins)-1):
        if mu[k] is not None:  # numeric variable
            ins[k] = (ins[k] - mu[k]) / sigma[k]

print('Done')