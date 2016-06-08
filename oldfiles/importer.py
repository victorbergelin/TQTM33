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
#

import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from sklearn.utils import shuffle

def load_q_data(input_dir_data,input_dir_label,output_dir):

    # 1. Get directory to rawdata



	# standard directory: 
    non_label_directory = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TrainingData/NonSmoking/*'
    label_directory = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TrainingData/Smoking/*'
    testing_directory = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TestingData/Smoking/*'
    # load lables
    list_of_files = getfilelist(label_directory)

	

    # 
    non_label_directory = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TrainingData/NonSmoking/*'
    label_directory = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TrainingData/Smoking/*'
    testing_directory = '/Users/victorbergelin/Dropbox/Liu/TQTM33/Code/Data/TestingData/Smoking/*'

    # load lables
    list_of_files = getfilelist(label_directory)

    # LOAD DATA for file
    lable_data = [loaddata(file_path) for file_path in list_of_files]

    lable_data_train
    lable_data_test


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
V


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
    run_crf()
    return
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)
#    run_crf(sequence_length_sec = 30, no_lable_vs_lable = 0.7, training_vs_testing = 0.8, sub_seq_length_sec = 3)

if __name__ == '__main__':
    sys.exit(main())




