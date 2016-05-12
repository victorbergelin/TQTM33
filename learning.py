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

import random
from time import time
#

import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.utils import shuffle

# SEQUENCE
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# FEATURE IMPLEMENTAITON: 
from collections import Counter

# CLUSTERING 
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
# from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
# np.seterr(divide='ignore', invalid='ignore')

# Parameters
sequence_length_sec = 30 
sub_seq_length_sec = 3
data_frequency = 4
feature_length = sub_seq_length_sec*data_frequency

np.random.seed(1)

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

# DATA HANDLER FUNCTIONS
# New data seq
def data2seq(training_data,window_length):
	y_label = []
	x_list = []
	for sequence in training_data:
        # extract windows
		nrsequences = int(len(sequence) / window_length)
		if nrsequences == 0:
			nrsequences = 1
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


def seq2seqfeatures(sequences,labels,feature_length,export_to_list_or_dict):
	x_train = []
	y_train = []
	for label,seq in zip(labels,sequences):
		data_points = int(len(seq) / feature_length)
		if data_points == 0:
			data_points = 1
		features = []
		label_set = []
		for i in range(data_points):
			temp = extractQfeatures(seq[i*feature_length:(i+1)*feature_length-1],export_to_list_or_dict)
			if len(temp)>0:
				features.append(temp)
				label_set.append(str(label))
		x_train.append(features)
		y_train.append(label_set)
	return x_train,y_train

"""

feature_data = 
array([[ -0.93,   0.48,   0.13,  33.7 ,   1.29],
	   [ -0.98,   0.34,   0.46,  33.7 ,   1.3 ],
	   [ -0.91,   0.21,   0.43,  33.7 ,   1.32],
	   [ -0.94,   0.1 ,   0.42,  33.7 ,   1.3 ],
	   [ -0.91,   0.03,   0.39,  33.7 ,   1.31],
	   [ -0.88,  -0.1 ,   0.28,  33.7 ,   1.31],
	   [ -0.95,  -0.04,   0.32,  33.7 ,   1.31],
	   [ -0.97,  -0.05,   0.36,  33.7 ,   1.33],
	   [ -0.96,  -0.08,   0.65,  33.7 ,   1.31],
	   [ -1.02,  -0.05,   0.43,  33.7 ,   1.31],
	   [ -0.98,  -0.09,   0.26,  33.7 ,   1.32]])

"""


def extractQfeatures(feature_data,list_or_dict,feature_selection=[]):
	#magnitude = np.sum(np.square(feature_data[:,range(3)]),1)
	#magnitude = (magnitude - np.mean(magnitude))/np.max(np.abs(magnitude))
	#feature_data = np.c_[feature_data,magnitude] 
	#features
	mean = np.mean(feature_data,0)
	variance = np.std(feature_data,0)
	freq_space = abs(fft(feature_data))
	freq_mean= np.mean(freq_space,0)
	freq_var = np.std(feature_data,0)
	# Square sum of any over 75% or under 25%
	p25 = np.percentile(np.abs(feature_data),25,0)
	p75 = np.percentile(np.abs(feature_data),75,0)
	p25_feat = ((feature_data*(np.abs(feature_data)<p25))**2).sum(axis=0)
	p75_feat = ((feature_data*(np.abs(feature_data)>p75))**2).sum(axis=0)
	p25 = np.percentile(np.abs(freq_space),25,0)
	p75 = np.percentile(np.abs(freq_space),75,0)
	p25_freq = ((freq_space*(np.abs(freq_space)<p25))**2).sum(axis=0)
	p75_freq = ((freq_space*(np.abs(freq_space)>p75))**2).sum(axis=0)
	# Sum difference, volatility 
	diff_feat = sum(np.abs(np.diff(feature_data.transpose()).transpose()))
	diff_freq = sum(np.abs(np.diff(freq_space.transpose()).transpose()))
	# Check for nan values:
	if list_or_dict:
		feature_seq={}
		if np.any(np.isnan([mean,variance,freq_mean,freq_var])):
			print "Nan feature"
			print [mean,variance,freq_mean,freq_var,diff_feat ,diff_freq ,p25_feat ,p75_feat, p25_freq, p75_freq]
		else:
			#print "-----"
			#print len(feature_data.T)
			#print range(len(feature_data.T))
			feature_seq={}
			for i in range(len(feature_data.T)):
				feature_seq["mean"+str(i)] = mean[i]
				feature_seq["var"+str(i)] = variance[i]
				feature_seq["fmean"+str(i)] = freq_mean[i]
				feature_seq["freq_var"+str(i)] = freq_var[i]
				#feature_seq["diff_feat"+str(i)] = diff_feat[i]
				#feature_seq["diff_freq"+str(i)] = diff_freq[i]
				#feature_seq["p25_feat"+str(i)] = p25_feat[i]
				#feature_seq["p75_feat"+str(i)] = p75_feat[i]
				#feature_seq["p25_freq"+str(i)] = p25_freq[i]
				#feature_seq["p75_freq"+str(i)] = p75_freq[i]
	else:
		feature_seq=[]
		if np.any(np.isnan([mean,variance,freq_mean,freq_var])):
			print "Nan feature"
			print [mean,variance,freq_mean,freq_var]
		else:
			feature_seq = list(np.concatenate((mean, variance, freq_mean, freq_var)))
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
	print("Results:")
	labels = list(crf.classes_)
	y_pred = crf.predict(X_test)
	sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
	print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))

def shuffle_and_cut(X,y,training_vs_testing):
	X, y = shuffle(X, y, random_state=0)
	cut_id = int(len(X)*training_vs_testing)
	X_train = X[:cut_id]
	X_test = X[cut_id:]
	y_train = y[:cut_id]
	y_test = y[cut_id:]
	return X_train,X_test,y_train,y_test
	
# FEATURE DATA: 
def print_state_features(state_features):
	for (attr, label), weight in state_features:
		print("%0.6f %-8s %s" % (weight, label, attr))
		print("Top positive:")
		print_state_features(Counter(crf.state_features_).most_common(30))
		print("\nTop negative:")
		print_state_features(Counter(crf.state_features_).most_common()[-30:])


crf.state_features_

# CLUSTERING: 
def clustering():
	sequence_length_sec = 15
	sub_seq_length_sec = 15
	data_frequency = 4
	n_neighbors = 15
	no_lable_vs_lable = 0.7
	training_vs_testing = 0.8
	# Sort and feature extract:
	training_data = load_q_data(no_lable_vs_lable)
	sequences,labels = data2seq(training_data,sequence_length_sec*data_frequency)
	norm_sequences,normalization_constants = normalize_train(sequences)
	X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,False)
	X = [item for sublist in X for item in sublist]
	y = [item for sublist in y for item in sublist]
	X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
	X, y = shuffle(X, y, random_state=0)
	n_samples = len(X)
	n_features = len(X[0])
	n_clusters = len(np.unique(y))
	print("n_clusters: %d, \t n_samples %d, \t n_features %d"
		  % (n_clusters, n_samples, n_features))
	print(79 * '_')
	print('% 9s' % 'init    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
	score11, score12 = bench_k_means(KMeans(init='k-means++', n_clusters = n_clusters, n_init = 10),"k-means++",X,y,n_samples)
	score21, score22 = bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),"random",X,y,n_samples)
	print(79 * '_')
	return 

def bench_k_means(estimator, name, X, y, sample_size):
	t0 = time()
	estimator.fit(X)
	rand_score = metrics.adjusted_rand_score(y, estimator.labels_)
	mutual_info = metrics.adjusted_mutual_info_score(y,  estimator.labels_)

	print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f' #    %.3f'
			% (name, (time() - t0), estimator.inertia_,
				metrics.homogeneity_score(y, estimator.labels_),
				metrics.completeness_score(y, estimator.labels_),
				metrics.v_measure_score(y, estimator.labels_),
				rand_score,
				mutual_info))
	return rand_score,mutual_info

"""
Data filter:
	Slicing to 18.0511563087% of non lable data
				 precision    recall  f1-score   support

			0      0.851     0.914     0.881     18441
			1      0.870     0.783     0.824     13590
  avg / total      0.859     0.858     0.857     32031

"""
def run_crf():

# Parameters
sequence_length_sec = 30
no_lable_vs_lable = 0.7
training_vs_testing = 0.8
sub_seq_length_sec = 3
data_frequency = 4
feature_length = sub_seq_length_sec*data_frequency

training_data = load_q_data(no_lable_vs_lable)
sequences,labels = data2seq(training_data,sequence_length_sec*data_frequency)
norm_sequences,normalization_constants = normalize_train(sequences)
X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,True)
# Randomize and split:
X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
# Train algorithm:
crf = training(X_train, y_train)
# Test algorithm:
testing(crf,X_test,y_test)

def main():
	"""Main entry point for the script."""
	# clustering()	
	run_crf()

	return

if __name__ == '__main__':
	sys.exit(main())

