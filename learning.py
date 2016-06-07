#!/usr/local/bin/python
""" 

Script to implement CRF for HAR on smoking pattern data
Dartmouth College
Victor Bergelin

"""
from __future__ import division
import sys
import csv
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy.fft import fft
import glob
import random
import time
import datetime
from collections import defaultdict 

# Scoring module
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.utils import shuffle

# CRF module
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# Feature module
from collections import Counter

# Ploting
import matplotlib.pyplot as plt

# Print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
# np.seterr(divide='ignore', invalid='ignore')

# Config
data_frequency = 4
np.random.seed(4)
random.seed(4)

# HELPER FUNCITONS
# ------------------------------------------
def getfilelist(directory):
	list_of_files = [join(directory, f) for f in glob.glob(directory) if isfile(join(directory, f)) and f[0]!='.']
	if not list_of_files:
		print "No files found at: " + directory
	return list_of_files

def loaddata(file_name,headerrows=1):
	sequence_list = []
	with open(file_name,'rb') as f:
		reader = csv.reader(f)
		header = ''
		for i, row in enumerate(reader):
			if i <= headerrows-1:
				header = list(row)
				continue
			sequence_list.append(list(row))
	return sequence_list

def loadrawdata(file_name,headerrows=1):
	sequence_list = []
	with open(file_name,'rb') as f:
		reader = csv.reader(f)
		header = list()
		for i, row in enumerate(reader):
			if i <= headerrows-1:
				header.append(list(row))
			else:
				sequence_list.append(list(row))
		# print header
	return sequence_list,header
# ------------------------------------------


# DATA HANDLER FUNCTIONS
# ------------------------------------------
def load_q_data(no_lable_vs_lable):
	# standard directory: 
	non_label_directory = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TrainingData/NonSmoking/*'
	label_directory = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TrainingData/Smoking/*'
	testing_directory = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TestingData/Smoking/*'
	# load lables
	list_of_files = getfilelist(label_directory)
	# load lables
	list_of_files = getfilelist(label_directory)
	# LOAD DATA for file
	lable_data = [loaddata(file_path) for file_path in list_of_files]
	# lable_data_train
	# lable_data_test
	random.shuffle(lable_data)
	# load non lables
	list_of_files = getfilelist(non_label_directory)
	non_lable_data = [loaddata(file_path) for file_path in list_of_files]
	random.shuffle(non_lable_data)
	# count content
	if len(lable_data)==0 or len(non_lable_data)==0:
		print("no data found, quiting")
		quit()
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

def load_raw_data(filepath = ''):
	# 1. Check for markers:
	list_of_markers = getfilelist(filepath+'markers*')
	marker_data_headers = [loadrawdata(file_path,2) for file_path in list_of_markers]
	if not marker_data_headers:
		print "No files loaded, quiting"
		quit()
	data = []
	marker_data = []
	marker_set = []
	for files in marker_data_headers:
		file_marks = []
		timestamps = []
		marker_set.append(files[0][0][0])
		for mark in [mark for mark in files[0] if mark]:
			if mark[0]=='Dataset':
				continue
			file_marks.append(mark[3])
			s = mark[0][11:]+"-"+mark[3]
			timestamps.append(time.mktime(datetime.datetime.strptime(s, "%Y_%m_%d-%H:%M:%S").timetuple()))
		marker_data.append([mark[0]] + [mark[0][:10]] + [mark[0][11:]] + [file_marks] + [timestamps])
	# 2. Load data from markers:
	all_data = []
	for i, set_name in enumerate(marker_set):
		print str(i) + " " + set_name
		log = ""
		try:
			log_matrix = []
			log_data = loadrawdata(getfilelist(filepath+set_name+'*')[0],8)
			for log in log_data[0]:
				timestr = time.mktime(datetime.datetime.strptime(set_name[11:]+"-"+log[0], "%Y_%m_%d-%H:%M:%S.%f").timetuple())
				log_row = [i] + [set_name[:10]] + [0]  + [timestr] + [set_name[11:]] + log
				log_matrix.append(log_row)
			marker_timestamps = marker_data[i][4]
			log_timestaps = [log[3] for log in log_matrix]
			for marker in marker_timestamps:
				index = np.argmin(np.abs(np.subtract(marker,log_timestaps)))
				log_matrix[index][2] = 1 # *** MAKE THIS LABEL MORE DYNAMIC AND GENERALL
			all_data.append(log_matrix)
		except:
			print "error at:"
			print set_name+"-"+log + " " + str(i)
			print str(marker_set) + " " + filepath
	return all_data


def load_raw_test_data(filepath = ''):
	list_of_test_data = getfilelist(filepath+'LOG*.csv')
	test_data = []
	for i, test_file in enumerate(list_of_test_data):
		test_file_data = []
		try:
			temp_data = loadrawdata(test_file,8)
			# print str(len(temp_data[0]))
			name = ""
			for ii, log in enumerate(temp_data[0]):
				# Get name from test_file:
				name = test_file[-25:-15]
				date = test_file[-14:-4]
				timestr = log[0]
				s = date + "-" + timestr
				timestr = time.mktime(datetime.datetime.strptime(s, "%Y_%m_%d-%H:%M:%S.%f").timetuple())
				log_row = [i] + [name] + [0] + [timestr] + [date] + log
				test_file_data.append(log_row)
		except:
			print "error at:"
			print name + " " + str(ii)
		test_data.append(test_file_data)
	return test_data

def data2seq_raw(data,window_length,label_prior):
	x_list = []
	y_list = []
	info_list = []
	label_length = int(data_frequency * np.mean(label_prior[1])) # Average over min and max time of label
	label_margin = int(np.diff(label_prior[1]))
	# data to sequences: use all as one serquence:
	for files_data in data:
		labler_iterator = label_length
		# If sequence [window * i..] have label: 
		#	Extend label with label_prior
		window_iterator = 0
		x_window_list = np.array([])
		info_window_list = np.array([])
		y_window_list = np.array([])
		label = 0
		label_pass = False
		for i, row in enumerate(files_data):
			data_row = []
			# Labeling: should detect time shift between frames? (non continous) ***
			if row[2] != 0:
				label = row[2]
				labler_iterator = 0
				label_pass = True # Skipp window untill label length is achived
				#print "start of label: " + str(i)
			# MARGIN LABEL HERE ***
			elif labler_iterator < label_length:
				labler_iterator += 1
			elif labler_iterator == label_length and label:
				#print "End of label pass:" + str(i)
				label = 0
				label_pass = True
			else:
				label = 0
			# Windows: detect time shifts ***
			if window_iterator < window_length:
				if len(x_window_list)==0:
					x_window_list = np.hstack(row)[[6,7,8,10,11]].astype(np.float)
					info_window_list = np.hstack(row)[[0,1,2,3,4,5]]
					y_window_list = np.array(label)
				else:
					x_window_list = np.vstack((x_window_list,np.hstack(row)[[6,7,8,10,11]].astype(np.float)))
					info_window_list = np.vstack((info_window_list,np.hstack(row)[[0,1,2,3,4,5]]))
					y_window_list = np.append(y_window_list,label)
				window_iterator += 1
			# End of window catcher
			else:
				if label_pass:
					#print "pass"
					label_pass = False
				else:
					x_list.append(x_window_list)
					y_list.append(y_window_list)
					info_list.append(info_window_list)
				x_window_list = np.array([])
				info_window_list = np.array([])
				y_window_list = np.array([])
				window_iterator = 0
	return x_list,y_list,info_list


# New data seq
def data2seq(data,window_length):
	y_label = []
	x_list = []
	info_list = []
	for sequence in data:
		# extract windows
		nrsequences = int(len(sequence) / window_length)
		if nrsequences == 0:
			nrsequences = 1
		for i in range(nrsequences):
			x_list.append(np.vstack(sequence[int(i*window_length):int((i+1)*window_length-1)]).astype(np.float)[:,[5,6,7,9,10]])
			info_list.append(np.vstack(sequence[int(i*window_length):int((i+1)*window_length-1)]).astype(np.float)[:,[0,1,2,3,4]])
			y_label.append(int(sequence[0][0]))
	return x_list,y_label,info_list

# normalize to zero mean, unit variance
# ------------------------------------------
def normalize_train(sequences):
	mean = []
	variance = []
	try:
		col_range = range(len(sequences[0][0]));
		for col in col_range:
			mean.append(np.mean([seq.T[col] for seq in sequences]))
			variance.append(np.max([seq.T[col] for seq in sequences]))
		normalization_constants = (np.array(mean),np.array(variance))
		normalized_sequences = []
		for seq in sequences:
			temp_seq = []
			for col in col_range:
				if variance[col]==0:
					temp_seq.extend([(seq.T[col]-mean[col])])
				else:
					temp_seq.extend([(seq.T[col]-mean[col])/variance[col]])
			normalized_sequences.append(np.array(temp_seq,dtype=float))
		return [normalized_seq.T for normalized_seq in normalized_sequences],normalization_constants
	except:
		print sequences
		print "-----------"
		print sequences[0]
		print "-----------"

def normalize_test(sequences,normalization_constants):
	mean = normalization_constants[0]
	maxval = normalization_constants[1]
	col_range = range(len(sequences[0][0]))
	normalized_sequences = []
	for seq in sequences:
		temp_seq = []
		for col in col_range:
			temp_seq.extend([(seq.T[col]-mean[col])/(maxval[col]-mean[col])])
		normalized_sequences.append(np.array(temp_seq,dtype=float))
	return [normalized_seq.T for normalized_seq in normalized_sequences]
# ------------------------------------------


def seq2seqfeatures(sequences,labels,feature_length,export_to_list_or_dict):
	x_train = []
	y_train = []
	for label_features,seq in zip(labels,sequences):
		data_points = int(len(seq) / feature_length)
		if data_points == 0:
			data_points = 1
		features = []
		label_set = []
		for i in range(data_points):
			temp = extractQfeatures(seq[int(i*feature_length):int((i+1)*feature_length-1)],export_to_list_or_dict)
			if len(temp)>0:
				features.append(temp)
				if isinstance(label_features,int):
					label=label_features
				else:
					label = str(max(label_features))
				label_set.append(label)
		x_train.append(features)
		y_train.append(label_set)
	return x_train,y_train

def extractQfeatures(feature_data,list_or_dict,feature_selection=[]):
	magnitude = np.sum(np.square(feature_data[:,range(3)]),1)
	magnitude = (magnitude - np.mean(magnitude))/np.max(np.abs(magnitude))
	feature_data = np.c_[feature_data,magnitude] 
	#features
	mean = np.mean(feature_data,0)
	variance = np.std(feature_data,0)
	freq_space = abs(fft(feature_data))
	freq_mean= np.mean(freq_space,0)
	freq_var = np.std(feature_data,0)
	# Square sum of any over 75% or under 25%
	# Sum difference, volatility 
	diff_feat = sum(np.abs(np.diff(feature_data.transpose()).transpose()))
	diff_freq = sum(np.abs(np.diff(freq_space.transpose()).transpose()))
	# Check for nan values:
	if list_or_dict:
		feature_seq={}
		if np.any(np.isnan([mean,variance,freq_mean,freq_var])):
			print "Nan feature"
			print [mean,variance,freq_mean,freq_var,diff_feat ,diff_freq] # ,p25_feat ,p75_feat, p25_freq, p75_freq]
		else:
			feature_seq={}
			for i in range(len(feature_data.T)):
				feature_seq["mean"+str(i)] = mean[i]
				feature_seq["var"+str(i)] = variance[i]
				feature_seq["fmean"+str(i)] = freq_mean[i]
				feature_seq["freq_var"+str(i)] = freq_var[i]
				feature_seq["diff_feat"+str(i)] = diff_feat[i]
				feature_seq["diff_freq"+str(i)] = diff_freq[i]
				# percentile decrease performance:
	else:
		feature_seq=[]
		if np.any(np.isnan([mean,variance,freq_mean,freq_var])):
			print "Nan feature"
			print [mean,variance,freq_mean,freq_var]
		else:
			feature_seq = list(np.concatenate((mean, variance, freq_mean, freq_var)))
	return feature_seq

# CRF MODEL:
# ------------------------------------------
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

def trainingRandomized(X_train, y_train):
	crf = sklearn_crfsuite.CRF(
			algorithm='lbfgs',
			max_iterations=100,
			all_possible_transitions=True
			)
	params_space = {
			'c1': scipy.stats.expon(scale=0.5),
			'c2': scipy.stats.expon(scale=0.05),
			}
	# use the same metric for evaluation
	f1_scorer = make_scorer(metrics.flat_f1_score,average='weighted', labels=labels)
	rs = RandomizedSearchCV(crf, params_space,
			cv=3,
			verbose=1,
			n_jobs=-1,
			n_iter=50,
			scoring=f1_scorer)
	rs.fit(X_train, y_train)
	# crf = rs.best_estimator_
	print('best params:', rs.best_params_)
	print('best CV score:', rs.best_score_)
	print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
	return rs
# labels = list(crf.classes_)
# labels.remove('O')

def testing(crf,X_test,y_test=[]):
	if y_test:
		print("Results:")
		labels = list(crf.classes_)
		y_pred = crf.predict(X_test)
		sorted_labels = [str(x) for x in sorted(labels,key=lambda name: (name[1:], name[0]))]
		print(metrics.flat_classification_report(y_test, y_pred, digits=3, labels=sorted_labels))
		return metrics.flat_accuracy_score(y_test, y_pred) # *** , labels=sorted_labels)
	else:
		y_pred = crf.predict(X_test)
		plt.plot(y_pred)
		plt.ylabel('predicted smokes')
		plt.show()
# ------------------------------------------

# DATA HELPERS:
# ------------------------------------------
def shuffle_data(X,y,no_lable_vs_lable):
	X, y = shuffle(X, y, random_state=0)
	# balance labels by subsampling:
	y_dict = defaultdict(list)
	for i, y_i in enumerate(y):
		y_dict[y_i[0]].append(i)
	# subsample
	y_set = set(y_dict)
	y_dict_len = [len(y_dict[y_set_i]) for y_set_i in sorted(list(y_set))]
	quotent = y_dict_len[0] / sum(y_dict_len)
	# generalize over multiple classes: 
	if(quotent > no_lable_vs_lable):
		# decrease 0 class labels:
		newLen = int(sum(y_dict_len[1:])/(1-no_lable_vs_lable))
		id_new = y_dict['0'][:newLen] + [y_dict[id] for id in y_set if not id in ['0']][0]
		X_sub = [X[id] for id in id_new]
		y_sub = [y[id] for id in id_new]
	else:
		newLen = int(y_dict_len[0]*(1-no_lable_vs_lable))
		id_new = y_dict['1'][:newLen] + [y_dict[id] for id in y_set if not id in ['0']][0]
		X_sub = [X[id] for id in id_new]
        y_sub = [y[id] for id in id_new]
	# X, y = shuffle(X_sub, y_sub, random_state=0)
	return X,y

def cut_data(X_sub,y_sub,training_vs_testing):
	X, y = shuffle(X_sub, y_sub, random_state=0)
	cut_id = int(len(X)*training_vs_testing)
	X_train = X[:cut_id]
	X_test = X[cut_id:]
	y_train = y[:cut_id]
	y_test = y[cut_id:]
	return X_train,X_test,y_train,y_test

def shuffle_and_cut(X,y,training_vs_testing=0.8,no_lable_vs_lable=0.7):
	X,y = shuffle_data(X,y,no_lable_vs_lable)
	X_train,X_test,y_train,y_test = cut_data(X,y,training_vs_testing)
	return X_train,X_test,y_train,y_test

def format_raw_data(data, inputvect,label_prior,normalization_constants=0):
	sequence_length_sec = inputvect[0]
	no_lable_vs_lable = inputvect[1]
	training_vs_testing = inputvect[2]
	sub_seq_length_sec = inputvect[3]
	window_length = int(sequence_length_sec*data_frequency)
	feature_length = sub_seq_length_sec*data_frequency
	sequences,labels,info_list = data2seq_raw(data,window_length,label_prior)
	print "len sequences = " + str(len(sequences))
	if normalization_constants == 1:
		norm_sequences,normalization_constants = normalize_train(sequences)
		X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,True)
		return X,y,normalization_constants
	elif isinstance(normalization_constants,int):
		norm_sequences,normalization_constants = normalize_train(sequences)
		X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,True)
		return X,y
	else:
		# USE normalization_constants in normalize_test:
		# norm_sequences,normalization_constants = normalize_train(sequences)
		norm_sequences = normalize_test(sequences, normalization_constants)
		X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,True)
		return X,y
# ------------------------------------------

# PRESENTER MODUL:
# ------------------------------------------
def present_run(inputvect = "", label_prior="",train_path="", test_path=""):
	print "---- INPUT VECTOR ----"
	print "sequence length (sec): " + str(inputvect[0])
	print "no lable vs lable: " + str(inputvect[1])
	print "training vs testing data: " + str(inputvect[2])
	print "sub seq length (sec): " + str(inputvect[3])
	print "---- LABEL PRIOR ----"
	print label_prior
	print "---- TRAIN TEST -----"
	print "Train: " + train_path
	print "Test: " + test_path
	print "---- MAIN RUN ----"
# ------------------------------------------


# FEATURE DATA: 
# ------------------------------------------
def print_state_features(state_features):
	for (attr, label), weight in state_features:
		print("%0.6f %-8s %s" % (weight, label, attr))
		print("Top positive:")
		print_state_features(Counter(crf.state_features_).most_common(30))
		print("\nTop negative:")
		print_state_features(Counter(crf.state_features_).most_common()[-30:])
# ------------------------------------------


# FULL RUNNERS:
# ------------------------------------------

def run_crf(inputvect = np.array([30, 0.7, 0.8, 3])):
	# Parameters
	sequence_length_sec = inputvect[0]
	no_lable_vs_lable = inputvect[1]
	training_vs_testing = inputvect[2]
	sub_seq_length_sec = inputvect[3]
	feature_length = sub_seq_length_sec*data_frequency
	training_data = load_q_data(no_lable_vs_lable)
	sequences,labels,info_list = data2seq(training_data,int(sequence_length_sec*data_frequency))
	norm_sequences,normalization_constants = normalize_train(sequences)
	X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,True)
	# Randomize and split:
	X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
	# Train algorithm:
	crf = training(X_train, y_train)
	crf = trainingRandomized(X_train, y_train)
	# Test algorithm:
	return testing(crf,X_test,y_test)


# Run on raw data:
def run_crf_raw(inputvect = np.array([30, 0.7, 0.8, 5]),subj_train=[],subj_test=[],label_prior={1:[30,600],0:[600,7200]},train_path="",test_path=""):
	starttime = time.time()
	present_run(inputvect, label_prior,train_path, test_path)
	data_frequency = 4
	no_lable_vs_lable = inputvect[1]
	training_vs_testing = inputvect[2]

	X_train = []
	X_test = []
	y_train = []
	y_test = []

	# Test data or not:
	if test_path=="": 
		data = load_raw_data(train_path)
		print "len(data) = " + str(len(data))
		X,y = format_raw_data(data,inputvect,label_prior)
		print "len(X) = " + str(len(x))
		X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
	else:
		train_data = load_raw_data(train_path)
		print "len(train_data) = " + str(len(train_data))
		X_train,y_train,normalization_constants = format_raw_data(train_data,inputvect,label_prior,normalization_constants=1)
		print "len(X) = " + str(len(X_train))
		X_train,y_train = shuffle_data(X_train,y_train,no_lable_vs_lable)
		print "Shuffle data"
		test_data = load_raw_test_data(test_path)
		print "len(test_data) = " + str(len(test_data))
		X_test,y_test= format_raw_data(test_data,inputvect,label_prior,normalization_constants)

	crf = training(X_train, y_train)
	#res = testing(crf,X_test,y_test)
	print "Run time: " + str(time.time()-starttime)
	res = testing(crf,X_test)
# ------------------------------------------


# MAIN AND SYS:
# ------------------------------------------
def main():
	"""Main entry point for the script."""
	# run_crf()

	train_path='/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/**/ph2/'
	test_path='/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/**/ph3/'
	

	train_path='/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/100/ph2/'
	test_path='/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/100/ph3/'
	run_crf_raw(train_path=train_path,test_path=test_path) 


if __name__ == '__main__':
	sys.exit(main())


""" 

inputvect = np.array([30, 0.7, 0.8, 5])
subj_train=[]
subj_test=[]
label_prior={1:[30,600],0:[600,7200]}
train_path=""
test_path=""

train_path='/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/100/ph2/'
test_path='/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/100/ph3/'


present_run(inputvect, label_prior,train_path, test_path)
data_frequency = 4
no_lable_vs_lable = inputvect[1]
training_vs_testing = inputvect[2]

X_train = []
X_test = []
y_train = []
y_test = []

# Test data or not:
if test_path=="": 
    data = load_raw_data(train_path)
    print "len(data) = " + str(len(data))
    X,y = format_raw_data(data,inputvect,label_prior)
    print "len(X) = " + str(len(x))
    X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
else:
    train_data = load_raw_data(train_path)
    print "len(train_data) = " + str(len(train_data))
    X_train,y_train,normalization_constants = format_raw_data(train_data,inputvect,label_prior,normalization_constants=1)
    print "len(X) = " + str(len(X_train))
    X_train,y_train = shuffle_data(X_train,y_train,no_lable_vs_lable)
    print "Shuffle data"
    test_data = load_raw_test_data(test_path)
    print "len(test_data) = " + str(len(test_data))
    X_test,y_test= format_raw_data(test_data,inputvect,label_prior,normalization_constants)

crf = training(X_train, y_train)
#res = testing(crf,X_test,y_test)
res = testing(crf,X_test)



"""
