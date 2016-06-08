#!/usr/local/bin/python
"""

Victor Bergelin

"""

import numpy as np
import sys
import learning as lr
import random
from sklearn.utils import shuffle
import time

def shuffle_and_cut_subj(X,y,training_vs_testing,train_subj,test_subj,info_list):
	X, y = shuffle(X, y, random_state=0)
	X_train = []
	Y_train = []
	X_test = []
	Y_test = [] # X is list of list of dirs
	# user_ids = u[info_row[1] for sublist in info_list for info_row in sublist]
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

def run_crf_subjects(inputvect = np.array([30, 0.7, 0.8, 3]),subj_train=[],subj_test=[],label_prior={1:[30,600],0:[600,7200]}):
	# Parameter
	filepath = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/Rawdataimport/'
	data = lr.load_raw_data(filepath)
	X,y = lr.format_raw_data(data,inputvect,label_prior)
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

	#X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
	#crf = training(X_train, y_train)
	#return testing(crf,X_test,y_test)

def run_crf_raw(inputvect = np.array([30, 0.7, 0.8, 5]),subj_train=[],subj_test=[],label_prior={1:[30,600],0:[600,7200]}):
	# Parameter
	filepath = ''
	data = load_raw_data()
	X,y = format_raw_data(data,inputvect,label_prior)
	data_frequency = 4
	training_vs_testing = inputvect[2]

	data = load_raw_data()
	X,y = format_raw_data(data,inputvect,label_prior)
	X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
	crf = training(X_train, y_train)
	res = testing(crf,X_test,y_test)   
	#X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
	#crf = training(X_train, y_train)
	#return testing(crf,X_test,y_test)


# Run on raw data:
def run_crf_raw_subjects(inputvect = np.array([30, 0.7, 0.8, 5]),subj_train=[],subj_test=[],label_prior={1:[30,600],0:[600,7200]},train_path="",test_path=""):
	starttime = time.time()
	lr.present_run(inputvect, label_prior,train_path, test_path)
	data_frequency = 4
	no_lable_vs_lable = inputvect[1]
	training_vs_testing = inputvect[2]

	X_train = []
	X_test = []
	y_train = []
	y_test = []
	time_seq = []

	# Test data or not:
	#if test_path=="": 
	#   data = load_raw_data(train_path)
	#   print "len(data) = " + str(len(data))
	#   X,y,time_seq = format_raw_data(data,inputvect,label_prior)
	#   print "len(X) = " + str(len(X))
	#   X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
	#else:
	train_data = lr.load_raw_data(train_path)
	print "len(train_data) = " + str(len(train_data))
	X_train,y_train,normalization_constants = lr.format_raw_data(train_data,inputvect,label_prior,normalization_constants=1)
	print "len(X) = " + str(len(X_train))
	X_train,y_train = lr.shuffle_data(X_train,y_train,no_lable_vs_lable)
	print "Shuffle data"
	crf = lr.training(X_train, y_train)
	print "Train time: " + str(time.time()-starttime)

	subjects = ['100','101','102','103','104','106','107','108','109','110']
	for s in subjects:
		test_path = '/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/' + s + '/ph3/'
		print "-------------" 
		print s
		test_data = lr.load_raw_test_data(test_path)
		print "len(test_data) = " + str(len(test_data))
		X_test,y_test,time_seq = lr.format_raw_data(test_data,inputvect,label_prior,normalization_constants)
		#res = testing(crf,X_test,y_test)
		print "Run time: " + str(time.time()-starttime)
		res = lr.testing(crf,X_test,time_seq=time_seq,save=str(starttime)+"_"+str(s))


def main():
	train_path='/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/**/ph2/'
	test_path='/Users/victorbergelin/LocalRepo/Data/Rawdataimport/subjects/**/ph3/'
	run_crf_raw_subjects(train_path=train_path,test_path=test_path)
	#subjects = ['100','101','102','103','104','106','107','108','109','110']
	#run_crf_subjects()
	#run_crf_subjects(inputvect = np.array([30, 0.7, 0.8, 3]),subj_train=[str(x) for x in range(100,110)],subj_test=['110'])

if __name__ == '__main__':
	sys.exit(main())



