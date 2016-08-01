#!/usr/local/bin/python
"""

Working with Deep learning for prediction of addictive behaviors 

Victor Bergelin

"""

import pandas as pd  
import matplotlib.pyplot as plt
from random import random

flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000  
pdata = pd.DataFrame({"a":flow}) # , "b":flow})  
#pdata.b = pdata.b.shift(9)  
data = pdata.iloc[10:] * random()  # some noise  

import numpy as np
np.random.seed(42)

def dummy_data(timesteps,data_dim, nb_classes):
	# generate dummy training data
	X_train = np.random.random((1000, timesteps, data_dim))
	y_train = np.random.random((1000, nb_classes))

	# generate dummy validation data
	X_test = np.random.random((100, timesteps, data_dim))
	y_test = np.random.random((100, nb_classes))
	return (X_train, y_train), (X_test, y_test)

def _load_data(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(np.array(int(round(data.iloc[i+n_prev].as_matrix()))))
		# docY.append(int(round(data.iloc[i+n_prev]).as_matrix()))
        # docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = int(len(df) * (1 - test_size))
    X_train, y_train = _load_data(df.iloc[:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])
    return (X_train, y_train), (X_test, y_test)

from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

in_neurons = 1
out_neurons = 1 # 2 
hidden_neurons = 50

batch_size = 450
nb_epoch = 5

def test_baseline(X_train=[], y_train=[], X_test=[], y_test=[]):
	# Get data:
	#(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
	
	timesteps = 100
	in_neurons = 20
	nb_classes = 1
	if(X_train == []):
		(X_train, y_train), (X_test, y_test) = dummy_data(timesteps,in_neurons, nb_classes)
	else:
		in_neurons = len(X_train[0][0])
		timesteps = len(X_train[0])
	model = Sequential()  
	model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=False)) 
	model.add(Dense(output_dim=out_neurons, input_dim=hidden_neurons))
	model.add(Activation("sigmoid"))  
	model.compile(class_mode='binary', loss='binary_crossentropy', optimizer='rmsprop')
	
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), validation_split=0.05, show_accuracy=True)
	#score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
	#print score
	#print acc

	predicted = model.predict(X_test)  

	# and maybe plot it
	pd.DataFrame(predicted[:100]).to_csv("predicted.csv")  
	pd.DataFrame(y_test[:100]).to_csv("test_data.csv") 
	pd.DataFrame(y_test[:100]).plot()  
	pd.DataFrame(predicted[:100]).plot()
	plt.show()
	

def test_morelayers(X_train=[], y_train=[], X_test=[], y_test=[]):
	model = Sequential()  
	# model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=False)) 
	model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=True)) 
	model.add(LSTM(output_dim=hidden_neurons, input_dim=hidden_neurons, return_sequences=True))
	model.add(LSTM(output_dim=hidden_neurons, input_dim=hidden_neurons, return_sequences=False))
	model.add(Dense(output_dim=out_neurons, input_dim=hidden_neurons))
	
	# 3 model.add(Activation("linear"))  
	model.add(Activation("sigmoid"))  
	# 2 model.compile(loss="mean_squared_error", optimizer="rmsprop")  
	# 4 model.compile(class_mode='binary', loss="mean_squared_error", optimizer="rmsprop")  
	model.compile(class_mode='binary', loss='binary_crossentropy', optimizer='rmsprop')
	# 7
	(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
	"""
	timesteps = 100
	nb_classes = 2
	(X_train, y_train), (X_test, y_test) = dummy_data(timesteps,in_neurons, nb_classes)
	"""
	# 3 model.fit(X_train, y_train, batch_size=450, nb_epoch=10, validation_split=0.05)  
	# 6 model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_split=0.05, show_accuracy=True)  
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), validation_split=0.05, show_accuracy=True)

	score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
	print score
	print acc

	predicted = model.predict(X_test)  

	# and maybe plot it
	#pd.DataFrame(predicted[:100]).to_csv("predicted.csv")  
	#pd.DataFrame(y_test[:100]).to_csv("test_data.csv") 
	pd.DataFrame(y_test[:100]).plot()  
	pd.DataFrame(predicted[:100]).plot()
	plt.show()



import learning as lr
import numpy as np
label_time_shift = 0
inputvect = np.array([30, 0.7, 0.8, 2])
data_frequency = 4
label_prior={1:[30,600],0:[600,7200]}
train_path = '107/ph2/'
base_path = '/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/'
full_train_path = base_path + train_path
training_vs_testing = inputvect[2]
no_lable_vs_lable = inputvect[1]
 
train_data = lr.load_raw_data(full_train_path,label_time_shift)
X,y,time_seq = lr.format_raw_data(train_data,inputvect,label_prior, export_to_list_or_dict=False)
 
X_train,X_test,y_train,y_test = lr.shuffle_and_cut(X,y,training_vs_testing)

test_baseline(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
# test_morelayers()

"""

HINT: Use the Theano flag 'exception_verbosity=high' for a debugprint

"""


