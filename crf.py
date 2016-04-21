""" 

Script to implement CRF for HAR on smoking pattern data

Dartmouth College

Victor Bergelin

"""
import sys
import pycrfsuite
import csv
from os import listdir
from os.path import isfile, join
import numpy as np

training_directory = '/Users/victorbergelin/Repo/Exjobb/Code/Data/TrainingData/Smoking'
testing_directory = '/Users/victorbergelin/Repo/Exjobb/Code/Data/TestingData/Smoking'


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

def data2seq(full_sequence_list,window_length):
	# format: [[label,list_of_seq:[seq:[..., data, ...]]],[label2[....]]]
	# get all labels:
	labels = list(set([int(sequence[0][0]) for sequence in full_sequence_list]))
	list_of_sequences = [list([]) for _ in xrange(len(labels))]
	# loop over all sequences
	seq_id = 0
	for sequence in full_sequence_list:
		# extract windows
		label = labels.index(int(sequence[0][0]))
		# quick fix:
		label = 0
		nrsequences = int(len(sequence) / window_length)
		for i in range(nrsequences):
			list_of_sequences[label].append(sequence[i*window_length:(i+1)*window_length-1])
	# return list_of_sequences
	return list_of_sequences

def seq2labels(data_file,seq_length_sec):
	pass

def seq2seqfeatures(sequences,feature_length):
	x_train = []
	for labeled_sequences in sequences:
		for sequence in labeled_sequences:
			# looking at 30 seconds of data, 120 points.
			data_points = int(len(sequence) / feature_length)
			for i in range(data_points):
				features = extractseqfeatures(sequence[i*feature_length:(i+1)*feature_length-1])
				x_train.append(features)
	return 

def extractseqfeatures(data_list):
	data = np.array(data_list)
	z = data[:,5]
	y = data[:,6]
	x = data[:,7]
	# mag = (x2
	temp = data[:,9]
	cond = data[:,10]
	
	# apply features
	feature_seq = []

	return feature_seq

def showresult():

	pass

def training(data):

	# pycrfsuite.ItemSequence
	pass


def main():
	"""Main entry point for the script."""
	list_of_files = getfilelist(training_directory)
	
	training_data = [loaddata(file_path) for file_path in list_of_files]
	sequences = data2seq(training_data,sequence_length_sec*data_frequency)
	
	# Split to sequences
	x_train = seq2seqfeatures(sequences,sub_seq_length_sec*data_frequency)
	y_train = seq2labels(sequences,sub_seq_length_sec*data_frequency)

	# Test algorithm
	# Train algorithm


	return

if __name__ == '__main__':
	sys.exit(main())

