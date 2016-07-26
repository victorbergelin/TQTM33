# Features
# - Automatiserad testning

print 'importing'
import sys, csv
import GPyOpt
from numpy.random import seed
from numpy import array
import numpy as np

from numpy import vstack
seed(12345)
# np.set_printoptions(precision=3)
bounds = [(0.001,0.080),(-0.122,0.122),(-0.122,0.122),(0.005,0.040),(0,1),(0.001,0.524),(0.101,0.160),(0,1)]
max_iter=1

# :param *f* the function to optimize. Should get a nxp numpy array as imput and return a nx1 numpy array.
def myf(x):
	header = 'MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency'
	print header
	print np.array_repr(x[0]).replace('\n', '').replace('\t', '')
	speed = input('Input 1/average_speed = ')
	output = np.array(float(speed))

	with open("data_new.py",'a') as f:
		np.savetxt(f, x, delimiter=",")
	# 	for item in x:
	# 		f.write("%s\n" % str(np.array_repr(item).replace('\n', '').replace('\t', '')))
	with open("readings_new.py",'a') as f:
		f.write("%s\n" % str(output))
	return output

print 'load data'
from custom import *
X = np.array([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = X
readings = Y
from default import *
X = ([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = vstack((data,X))
readings = vstack((readings,Y))
from lowstiffness import *
X = ([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = vstack((data,X))
readings = vstack((readings,Y))
from msh import *
X = ([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = vstack((data,X))
readings = vstack((readings,Y))
from msl import *
X = ([MaxStepX, TorsoWy, TorsoWx, StepHeight, Stiffness, MaxStepTheta, MaxStepY, MaxStepFrequency])
Y = 1/average_speed
data = vstack((data,X))
readings = vstack((readings,Y))


# READ COLLECTED DATA
# import string
# all=string.maketrans('','')
# nodigs=all.translate(all, string.digits)
# row.translate(all, nodigs)

with open("data_new.py", "r") as f:
	rows = np.loadtxt(f,delimiter=',')
	print 'length of data: ' + str(len(rows))
	for X in rows:
		data = vstack((data,X))

with open('readings_new.py', "r") as f:
	rows = np.loadtxt(f,delimiter='\n')
	print rows
	for Y in rows:
		readings = vstack((readings,Y))

print 'starting script'
BOnao = GPyOpt.methods.BayesianOptimization(myf,bounds,X=data,Y=readings)
N_iter = 50
for i in range(N_iter):
	if BOnao.run_optimization(max_iter) == 0: break
	BOnao.save_report()
BOnao.plot_convergence()

