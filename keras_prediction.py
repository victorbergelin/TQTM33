#!/usr/local/bin/python
"""

Working with Deep learning for prediction of addictive behaviors 

Victor Bergelin

"""
import sys
import pandas as pd  
import matplotlib.pyplot as plt
from random import random
import numpy as np
np.random.seed(42)

# Create data: {{{
# ------------------------------------------

def semi_dummy_data():

	flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000  
	pdata = pd.DataFrame({"a":flow}) # , "b":flow})  
	#pdata.b = pdata.b.shift(9)  
	data = pdata.iloc[10:] * random()  # some noise  
	return data

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

# ------------------------------------------ }}}

from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

# non changable parameters:
out_neurons = 1
data_frequency = 4
base_path = '/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/'

# Full model functions: {{{ 
# ------------------------------------------

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
	
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), validation_split=0.05, show_accuracy=True,shuffle=False)
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

def test_morelayers(X_train=[], y_train=[], X_test=[], y_test=[],plot=0):
	timesteps = 100
	in_neurons = 20
	nb_classes = 1
	if(X_train == []):
		(X_train, y_train), (X_test, y_test) = dummy_data(timesteps,in_neurons, nb_classes)
	else:
		in_neurons = len(X_train[0][0])
		timesteps = len(X_train[0])

	model = Sequential()  
	model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=True)) 
	model.add(LSTM(output_dim=hidden_neurons, input_dim=hidden_neurons, return_sequences=True))
	model.add(LSTM(output_dim=hidden_neurons, input_dim=hidden_neurons, return_sequences=False))
	model.add(Dense(output_dim=out_neurons, input_dim=hidden_neurons))
	
	model.add(Activation("sigmoid"))  
	model.compile(class_mode='binary', loss='binary_crossentropy', optimizer='rmsprop')
	"""
	(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
	timesteps = 100
	nb_classes = 2
	(X_train, y_train), (X_test, y_test) = dummy_data(timesteps,in_neurons, nb_classes)
	"""
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), validation_split=0.05, show_accuracy=True,shuffle=False)
	"""
	score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
	print score
	print acc
	"""
	predicted = model.predict(X_test)  
	# and maybe plot it
	#pd.DataFrame(predicted[:100]).to_csv("predicted.csv")  
	#pd.DataFrame(y_test[:100]).to_csv("test_data.csv")
	if plot:
		pd.DataFrame(y_test[:500]).plot()  
		pd.DataFrame(predicted[:500]).plot()
		plt.show()

# ------------------------------------------ }}}

# Model functions: {{{ 
# ------------------------------------------
def simple_model(in_neurons, hidden_neurons):
	model = Sequential()  
	model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=False)) 
	model.add(Dense(output_dim=out_neurons, input_dim=hidden_neurons))
	model.add(Activation("sigmoid"))  
	model.compile(class_mode='binary', loss='binary_crossentropy', optimizer='rmsprop')
	return model

def simple_model(in_neurons, hidden_neurons):
	model = Sequential()  
	model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=True)) 
	model.add(LSTM(output_dim=hidden_neurons, input_dim=hidden_neurons, return_sequences=True))
	model.add(LSTM(output_dim=hidden_neurons, input_dim=hidden_neurons, return_sequences=False))
	model.add(Dense(output_dim=out_neurons, input_dim=hidden_neurons))
	model.add(Activation("sigmoid"))  
	model.compile(class_mode='binary', loss='binary_crossentropy', optimizer='rmsprop')
	return model

# ------------------------------------------ }}}

# Main and sys: {{{ 
# ------------------------------------------
def main(inputargs):
	import learning as lr
	import numpy as np

	# Tuning parameters
	label_time_shift = 1 # compensate for button press interference
	seqlength = 30
	seqseqlength = 2
	training_vs_testing = 0.8
	no_lable_vs_lable = 0.7 # not working now
	label_prior = {1:[30,600],0:[600,7200]}
	class_weights = {0: 1, 1: 20}
	hidden_neurons = 50
	# -----------------

	# Test parameters
	train_path = '**/ph2/' # train_path = '100/ph2/'
	plot = 1
	# -----------------

	# Test setup
	batch_size = 450
	nb_epoch = 50
	# -----------------
	
	# Test setup auto
	in_neurons = 1
	inputvect = np.array([seqlength, no_lable_vs_lable, training_vs_testing, seqseqlength])

	full_train_path = base_path + train_path
	# -----------------

	# Get data 
	train_data = lr.load_raw_data(full_train_path,label_time_shift)
	X,y,time_seq = lr.format_raw_data(train_data,inputvect,label_prior, export_to_list_or_dict=False)
	# Class_weights?
	#X.y = lr.equalize_data(X,y,no_lable_vs_lable)
	X_train,X_test,y_train,y_test = lr.cut_data(X,y,training_vs_testing)
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	y_train = np.array([int(max(y_t)) for y_t in y_train])
	y_test = np.array([int(max(y_t)) for y_t in y_test])
	in_neurons = len(X_train[0][0])
	timesteps = len(X_train[0])
	# -----------------

	# Setup and train model
	# model = simple_model(in_neurons, hidden_neurons)
	model = complex_model(in_neurons, hidden_neurons)
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), validation_split=0.05, show_accuracy=True,shuffle=False, class_weight=class_weights)
	# -----------------
	predicted = model.predict(X_test)
	
	# Plot results:
	if plot:
		pd.DataFrame(y_test[:500]).plot()
		pd.DataFrame(predicted[:500]).plot()
		plt.show()

# test_baseline(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
# test_morelayers(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

if __name__ == '__main__':
	sys.exit(main(sys.argv))

# ------------------------------------------ }}}

# Command line: {{{ 
# ------------------------------------------
"""
import learning as lr
import numpy as np
label_time_shift = -600 # compensate for button press interference

seqlength = 30
seqseqlength = 2
training_vs_testing = 0.8
no_lable_vs_lable = 0.7 # not working now


inputvect = np.array([seqlength, 0.7, 0.8, seqseqlength])
data_frequency = 4
label_prior={1:[30,600],0:[600,7200]}
train_path = '100/ph2/'
train_path = '**/ph2/'
base_path = '/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/'
full_train_path = base_path + train_path
training_vs_testing = inputvect[2]
no_lable_vs_lable = inputvect[1]
 
train_data = lr.load_raw_data(full_train_path,label_time_shift)
X,y,time_seq = lr.format_raw_data(train_data,inputvect,label_prior, export_to_list_or_dict=False)

X_train,X_test,y_train,y_test = lr.cut_data(X,y,training_vs_testing)
# X_train,X_test,y_train,y_test = lr.shuffle_and_cut(X,y,training_vs_testing)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array([int(max(y_t)) for y_t in y_train])
y_test = np.array([int(max(y_t)) for y_t in y_test])
import keras_prediction as kp


kp.test_baseline(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
kp.test_morelayers(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

import panda as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

nb_classes = 1
in_neurons = len(X_train[0][0])
timesteps = len(X_train[0])
in_neurons = 1            
out_neurons = 1 # 2       
hidden_neurons = 50       
                          
batch_size = 45           
nb_epoch = 5  

model = Sequential()  
model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=True)) 
model.add(LSTM(output_dim=hidden_neurons, input_dim=hidden_neurons, return_sequences=True))
model.add(LSTM(output_dim=hidden_neurons, input_dim=hidden_neurons, return_sequences=False))
model.add(Dense(output_dim=out_neurons, input_dim=hidden_neurons))

model.add(Activation("sigmoid"))  
model.compile(class_mode='binary', loss='binary_crossentropy', optimizer='rmsprop')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), validation_split=0.05, show_accuracy=True,shuffle=False)

predicted = model.predict(X_test)  
predictedclass = model.predict_classes(X_test, batch_size)

pd.DataFrame(y_test[:500]).plot()  
pd.DataFrame(predicted[:500]).plot()
plt.show()

loss_and_metrics = model.evaluate(X_test, y_test)




model = Sequential()
model.add(LSTM(output_dim=hidden_neurons, input_dim=in_neurons, return_sequences=False))
model.add(Dense(output_dim=out_neurons, input_dim=hidden_neurons))
model.add(Activation("sigmoid"))



"""

"""
import keras_prediction as kp
kp.test_baseline(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

HINT: Use the Theano flag 'exception_verbosity=high' for a debugprint

"""

# ------------------------------------------ }}}
