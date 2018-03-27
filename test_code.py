import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('F1_list.txt')

plt.hist(data,bins=30)
plt.show()