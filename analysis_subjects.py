"""

Victor Bergelin

"""

import numpy as np
import sys
import learning as lr
import random
from sklearn.utils import shuffle



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

def main():
	run_crf_subjects()

if __name__ == '__main__':
	sys.exit(main())


"""

import numpy as np
import sys
import learning as lr

inputvect = np.array([10, 0.7, 0.8, 2])
label_prior={1:[30,600],0:[600,7200]}
data_frequency = 4

data = lr.load_raw_data()

sequence_length_sec = inputvect[0]
no_lable_vs_lable = inputvect[1]
training_vs_testing = inputvect[2]
sub_seq_length_sec = inputvect[3]
window_length = int(sequence_length_sec*data_frequency)
feature_length = sub_seq_length_sec*data_frequency

sequences,labels,info_list = lr.data2seq_raw(data,window_length,label_prior)

norm_sequences,normalization_constants = lr.normalize_train(sequences)
X,y = lr.seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,True)

X_train,X_test,y_train,y_test = lr.shuffle_and_cut(X,y,training_vs_testing)
crf = lr.training(X_train, y_train)
res = lr.testing(crf,X_test,y_test)

"""
