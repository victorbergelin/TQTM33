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
import glob
import random
from time import time
import datetime

import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.utils import shuffle

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# FEATURE IMPLEMENTAITON: 
from collections import Counter

# Print options
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
# np.seterr(divide='ignore', invalid='ignore')

# Parameters



# Config
data_frequency = 4
np.random.seed(1)
random.seed(1)

# HELPER FUNCITONS
def getfilelist(directory):
    list_of_files = [join(directory, f) for f in glob.glob(directory) if isfile(join(directory, f)) and f[0]!='.' ]
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


def load_q_data_subj(no_lable_vs_lable, train_subjects, test_subjects):
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
	#lable_data_train
	#lable_data_test
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

def loadsubjectdata(train_dir_or_file,test_dir_or_file, window_length,sequence_length):
	root_dir = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TrainingData/Subjects/'
	file_list = [root_dir + file_search for file_search in train_dir_or_file]
	
	list_of_files = [getfilelist(file_path) for file_path in file_list]
	data = [loaddata(file_path) for sublist in list_of_files for file_path in sublist]
	random.shuffle(data)
	print str(len(data)) + " records of labeled data imported for training"
	print len(data[0])
	sequences,labels = data2seq(data,int(sequence_length_sec*data_frequency))

	
	norm_sequences = normalize_test(sequences,normalization_constants)
	X_Train,Y_Train = seq2seqfeatures(norm_sequences, labels, sequence_length, True)
	X_Test = []
	list_of_files = getfilelist(test_dir_or_file)
	data = [loaddata(file_path) for file_path in list_of_files]
	random.shuffle(data)
	x_list = []
	for sequence in data:
		nrsequences = int(len(sequence) / window_length)
		if nrsequences == 0:
			nrsequences = 1
		for i in range(nrsequences):
			x_list.append(np.vstack(sequence[int(i*window_length):int((i+1)*window_length-1)]).astype(np.float)[:,[5,6,7,9,10]])
	norm_sequences = normalize_test(sequences,normalization_constants)
	X_Test,Y_Test = seq2seqfeatures(norm_sequences, labels, sequence_length, True)
	return X_Train,Y_Train,X_Test,Y_Test

"""
PRIMARY:
File Exported by Q - (c) 2009 Affectiva Inc.
File Version: 1.01
Firmware Version: 1.81
UUID: AQL441200KG
Sampling Rate: 4
Start Time: 2014-09-11 08:06:19 Offset:-04
Time,Z-axis,Y-axis,X-axis,Battery,Celsius,EDA(uS),Event
---------------------------------------------------------
08:06:19.000,0.450,0.430,-0.660,-1,42.000,1.100,0
"""

filepath = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/Rawdataimport/'
def load_raw_data(filepath):
	# *** Hur formateras filerna fran csv export?
	list_of_markers = getfilelist(filepath+'markers*')
	marker_data_headers = [loadrawdata(file_path,2) for file_path in list_of_markers]
	data = []
	marker_data = []
	marker_set = []
	for files in marker_data_headers:
		file_marks = []
		timestamps = []
		marker_set.append(files[0][0][0])
		for mark in [mark for mark in files[0] if mark]:
			file_marks.append(mark[3])
			s = mark[0][11:]+"-"+mark[3]
			timestamps.append(time.mktime(datetime.datetime.strptime(s, "%Y_%m_%d-%H:%M:%S").timetuple()))
		marker_data.append([mark[0]] + [mark[0][:10]] + [mark[0][11:]] + [file_marks] + [timestamps])
	all_data = []
	for i, set_name in enumerate(marker_set):
		log = ""
		ii = 0
		try:
			log_matrix = []
			log_data = loadrawdata(getfilelist(filepath+set_name+'*')[0],8)
			for log in log_data[0]:
				timestr = time.mktime(datetime.datetime.strptime(set_name[11:]+"-"+log[0], "%Y_%m_%d-%H:%M:%S.%f").timetuple())
				log_row = [set_name[:10]] + [0]  + [timestr] + [set_name[11:]] + log
				log_matrix.append(log_row)
			marker_timestamps = marker_data[i][4]
			log_timestaps = [log[2] for log in log_matrix]
			for marker in marker_timestamps:
				index = np.argmin(np.abs(np.subtract(marker,log_timestaps)))
				log_matrix[index][1] = 1 # *** MAKE THIS LABEL MORE DYNAMIC AND GENERALL
			all_data.append(log_matrix)
			ii = ii + 1
		except:
			print "error at:"
			print set_name[11:]+"-"+log[0] + " " + str(ii)
	return all_data



# DATA HANDLER FUNCTIONS
# New data seq
def data2seq(training_data,window_length):
	y_label = []
	x_list = []
	info_list = []
	for sequence in training_data:
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
			temp = extractQfeatures(seq[int(i*feature_length):int((i+1)*feature_length-1)],export_to_list_or_dict)
			if len(temp)>0:
				features.append(temp)
				label_set.append(str(label))
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
	#p25 = np.percentile(np.abs(feature_data),25,0)
	#p75 = np.percentile(np.abs(feature_data),75,0)
	#p25_feat = ((feature_data*(np.abs(feature_data)<p25))**2).sum(axis=0)
	#p75_feat = ((feature_data*(np.abs(feature_data)>p75))**2).sum(axis=0)
	#p25 = np.percentile(np.abs(freq_space),25,0)
	#p75 = np.percentile(np.abs(freq_space),75,0)
	#p25_freq = ((freq_space*(np.abs(freq_space)<p25))**2).sum(axis=0)
	#p75_freq = ((freq_space*(np.abs(freq_space)>p75))**2).sum(axis=0)
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
			feature_seq={}
			for i in range(len(feature_data.T)):
				feature_seq["mean"+str(i)] = mean[i]
				feature_seq["var"+str(i)] = variance[i]
				feature_seq["fmean"+str(i)] = freq_mean[i]
				feature_seq["freq_var"+str(i)] = freq_var[i]
				feature_seq["diff_feat"+str(i)] = diff_feat[i]
				feature_seq["diff_freq"+str(i)] = diff_freq[i]
				# percentile decrease performance:
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

def testing(crf,X_test,y_test):
	print("Results:")
	labels = list(crf.classes_)
	y_pred = crf.predict(X_test)
	sorted_labels = [str(x) for x in sorted(labels,key=lambda name: (name[1:], name[0]))]
	
	print(metrics.flat_classification_report(y_test, y_pred, digits=3, labels=sorted_labels))
	return metrics.flat_accuracy_score(y_test, y_pred) # *** , labels=sorted_labels)

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

"""
Data filter:
Slicing to 18.0511563087% of non lable data
             precision    recall  f1-score   support

        0      0.851     0.914     0.881     18441
        1      0.870     0.783     0.824     13590
avg / total      0.859     0.858     0.857     32031

"""

def run_crf(inputvect = np.array([30, 0.7, 0.8, 3])):
	# Parameters
	sequence_length_sec = inputvect[0]
	no_lable_vs_lable = inputvect[1]
	training_vs_testing = inputvect[2]
	sub_seq_length_sec = inputvect[3]
	feature_length = sub_seq_length_sec*data_frequency
	training_data = load_q_data(no_lable_vs_lable)
	sequences,labels = data2seq(training_data,int(sequence_length_sec*data_frequency))
	norm_sequences,normalization_constants = normalize_train(sequences)
	X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,True)
	# Randomize and split:
	X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
	# Train algorithm:
	crf = training(X_train, y_train)
	#crf = trainingRandomized(X_train, y_train)
    # Test algorithm:
	return testing(crf,X_test,y_test)

def shuffle_and_cut_subj(X,y,training_vs_testing,train_subj,test_subj,info_list):
	X, y = shuffle(X, y, random_state=0)

	X_train = []
	Y_train = []
	X_test = []
	Y_test = []

	# X is list of list of dirs

	# user_ids =  [info_row[1] for sublist in info_list for info_row in sublist]
	user_ids =  [info_row[0][1] for info_row in info_list]
	try:
		for i, user_id in enumerate(user_ids):

			if i == len(user_ids): 
				print 'Index warnig'
				continue
			if str(int(user_id)) in train_subj:
				X_train.append(X[i])
				Y_train.append(y[i])
			elif str(int(user_id)) in test_subj:
				X_test.append(X[i])
				Y_test.append(y[i])
			else:
				print str(int(user_id))

	except:
		print("Unexpected error:", sys.exc_info()[0])

	return X_train,X_test,Y_train,Y_test

def run_crf_subjects(inputvect = np.array([30, 0.7, 0.8, 3]),subj_train=[],subj_test=[]):
    # Parameter

	sequence_length_sec = inputvect[0]
	no_lable_vs_lable = inputvect[1]
	training_vs_testing = inputvect[2]
	sub_seq_length_sec = inputvect[3]
	feature_length = sub_seq_length_sec*data_frequency
	training_data = load_q_data(no_lable_vs_lable)
	sequences,labels,info_list = data2seq(training_data,int(sequence_length_sec*data_frequency))
	norm_sequences,normalization_constants = normalize_train(sequences)
	X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,True)
	# print "DATA " + str(len(X))
	subjects = ['100','101','102','103','104','106','107','108','109','110']
	for subject in subjects:
		subj_train = [x for x in subjects if not x in subject]
		subj_test = subject
		print "-----------------"
		print "train: " + str(subj_train)
		print "test: " + str(subj_test)
		X_train,X_test,y_train,y_test = shuffle_and_cut_subj(X,y,training_vs_testing,subj_train,subj_test,info_list)
		crf = training(X_train, y_train)
		testing(crf,X_test,y_test)

def run_crf_test(test_data_files = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TestingData/Smoking/107.csv'):
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
    crf = trainingRandomized(X_train, y_train)
    # Test algorithm:
    testing(crf,X_test,y_test)
    X_test_real = loadtestdata(test_data_files,sequence_length_sec*data_frequency,sub_seq_length_sec*data_frequency)


def main():
	"""Main entry point for the script."""
	subjects = ['100','101','102','103','104','106','107','108','109','110']
	run_crf_subjects()

	# run_crf()
	# run_crf_subjects(inputvect = np.array([30, 0.7, 0.8, 3]),subj_train=[str(x) for x in range(100,110)],subj_test=['110'])
	
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)

if __name__ == '__main__':
    sys.exit(main())

