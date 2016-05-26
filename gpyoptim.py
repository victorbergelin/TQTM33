import sys, csv
import GPyOpt
from numpy.random import seed
from numpy import array
import numpy as np

from numpy import vstacka
seed(12345)

import learning

# Variables:
# sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3

bounds = [(15.1,300),(0,1,0.9),(0.1,0.9),(1,15)]
max_iter=20

def myf(x):
	return learning.run_crf(x)


print 'starting script'
BOnao = GPyOpt.methods.BayesianOptimization(myf,bounds)
BOnao.run_optimization(max_iter)
BOnao.save_report()
BOnao.plot_convergence()



