""" 

Script to implement CRF for HAR on smoking pattern data

Dartmouth College

Victor Bergelin

"""
import sys
import csv
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy.fft import fft
np.seterr(divide='ignore', invalid='ignore')
import random
#

import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

#

import sklearn_crfsuite
# import pycrfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics



# Print options
np.set_printoptions(suppress=True)

# Parameters
sequence_length_sec = 30 
sub_seq_length_sec = 3
data_frequency = 4

# HELPER FUNCITONS
def getfilelist(directory):
	list_of_files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and f[0]!='.' ]
	return list_of_files

def loaddata(file_name):
	sequence_list = []
	with open(file_name,'rb') as f:
		reader = csv.reader(f)
		header = ''
		for i, row in enumerate(reader):
			if i == 0:
				header = list(row)
				continue
			sequence_list.append(list(row))
	return sequence_list

# DATA HANDLER FUNCTIONS
# New data seq
def data2seq(training_data,window_length):
	y_label = []
	x_list = []
	for sequence in training_data:
        # extract windows
		nrsequences = int(len(sequence) / window_length)
		for i in range(nrsequences):
			x_list.append(np.vstack(sequence[i*window_length:(i+1)*window_length-1]).astype(np.float)[:,[5,6,7,9,10]])
			y_label.append(int(sequence[0][0]))
	return x_list,y_label


# normalize to zero mean, unit variance
def normalize_train(sequences):
	mean = []
	variance = []
	col_range = range(len(sequences[0][0]));
	for col in col_range:
		mean.append(np.mean([seq.T[col] for seq in sequences]))
		variance.append(np.var([seq.T[col] for seq in sequences]))
	normalization_constants = (np.array(mean),np.array(variance))
	normalized_sequences = []
	for seq in sequences:
		temp_seq = []
		for col in col_range:
			temp_seq.extend([(seq.T[col]-mean[col])/variance[col]])
		normalized_sequences.append(np.array(temp_seq,dtype=float))
	return [normalized_seq.T for normalized_seq in normalized_sequences],normalization_constants

feature_length = sub_seq_length_sec*data_frequency
def seq2seqfeatures(sequences,labels,feature_length):
	x_train = []
	y_train = []
	for label,seq in zip(labels,sequences):
		# looking at 30 seconds of data, 120 points.
		data_points = int(len(seq) / feature_length)
		features = []
		label_set = []
		for i in range(data_points):
			features.append(extractQfeatures(seq[i*feature_length:(i+1)*feature_length-1]))
			label_set.append(str(label))
		x_train.append(features)
		y_train.append(label_set)
	return x_train,y_train

def extractQfeatures(feature_data):
	magnitude = np.sum(np.square(feature_data[:,range(3)]),1)
	# normalize magnitude: 
	magnitude = (magnitude - np.mean(magnitude))/np.var(magnitude)
	feature_data = np.c_[feature_data,magnitude] 
	#features
	feature_seq={}
	mean = np.mean(feature_data,0)
	variance = np.std(feature_data,0)
	freq_space = abs(fft(feature_data))
	freq_mean= np.mean(freq_space,0)
	freq_var = np.std(feature_data,0)
	for i in range(len(feature_data.T)):
		feature_seq["mean"+str(i)] = mean[i]
		feature_seq["var"+str(i)] = variance[i]
		feature_seq["fmean"+str(i)] = freq_mean[i]
		feature_seq["fvar"+str(i)] = freq_var[i]
	return feature_seq

	# 25 % energy
	# 75 %
	# 10 %
	# fluctuations, diff
	## Frequency: fft
	# sum of frequency in bands: 0-1 hz, 1-3, 3-10 hz, 10+ hz
	## Nr peaks: 
	# avg peak width
	# apply features

def training(X_train, y_train):
	# pycrfsuite.ItemSequence
	# %%time
	crf = sklearn_crfsuite.CRF(
			algorithm='lbfgs',
			c1=0.1,
			c2=0.1,
			max_iterations=100,
			all_possible_transitions=True
			)
	crf.fit(X_train, y_train)
	return crf



# labels = list(crf.classes_)
# labels.remove('O')


def testing(crf,X_test,y_test):
	labels = list(crf.classes_)
	y_pred = crf.predict(X_test)

	sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
	print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
	return

def load_q_data(no_lable_vs_lable):
	non_label_directory = '/Users/victorbergelin/Repo/Exjobb/Code/Data/TrainingData/NonSmoking'
	label_directory = '/Users/victorbergelin/Repo/Exjobb/Code/Data/TrainingData/Smoking'
	testing_directory = '/Users/victorbergelin/Repo/Exjobb/Code/Data/TestingData/Smoking'
	# load lables
	list_of_files = getfilelist(label_directory)
	lable_data = [loaddata(file_path) for file_path in list_of_files]
	random.shuffle(lable_data)
	# load non lables
	list_of_files = getfilelist(non_label_directory)
	non_lable_data = [loaddata(file_path) for file_path in list_of_files]
	random.shuffle(non_lable_data)
	# count content
	non_lable_fraction = float(sum(len(x) for x in lable_data))/float(sum(len(x) for x in non_lable_data))
	conversion_fraction = non_lable_fraction/no_lable_vs_lable
	print "Data filter:" 
	if conversion_fraction < 1:
		non_lable_cut_id = int(len(non_lable_data)*conversion_fraction)
		training_data = lable_data + non_lable_data[:non_lable_cut_id]
		print "Slicing to " + str(conversion_fraction*100) + "% of non lable data"
	else:
		lable_cut_id = int(len(lable_data)/conversion_fraction)
		training_data = lable_data[:lable_cut_id] + non_lable_data
		print "Slicing to " + str(100/conversion_fraction) + "% of lable data"
	return training_data

def main():
	"""Main entry point for the script."""
	
	no_lable_vs_lable = 0.7
	training_vs_testing = 0.8

	# Sort and feature extract:
	training_data = load_q_data(no_lable_vs_lable)
	sequences,labels = data2seq(training_data,sequence_length_sec*data_frequency)
	norm_sequences,normalization_constants = normalize_train(sequences)
	X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency)

	# Randomize and split:
	combined = zip(X, y)
	random.shuffle(combined)
	X[:], y[:] = zip(*combined)
	cut_id = int(len(X)*training_vs_testing)
	X_train = X[:cut_id]
	X_test = X[cut_id:]
	y_train = y[:cut_id]
	y_test = y[cut_id:]

	# Train algorithm:
	crf = training(X_train, y_train)

	# Test algorithm:
	testing(crf,X_test,y_test)
	
	return

if __name__ == '__main__':
	sys.exit(main())

