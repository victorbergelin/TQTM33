
import learning

    sequence_length_sec = inputvect[0]
    no_lable_vs_lable = inputvect[1]
    training_vs_testing = inputvect[2]
    sub_seq_length_sec = inputvect[3]
    feature_length = sub_seq_length_sec*data_frequency
    training_data = load_q_data(no_lable_vs_lable)
    sequences,labels = data2seq(training_data,int(sequence_length_sec*data_frequency))
    norm_sequences,normalization_constants = normalize_train(sequences)
    X,y = seq2seqfeatures(norm_sequences, labels, sub_seq_length_sec*data_frequency,True)
	
    X_train,X_test,y_train,y_test = shuffle_and_cut(X,y,training_vs_testing)
    crf = training(X_train, y_train)
    return testing(crf,X_test,y_test)


def main():
	run_crf_subjects()

if __name__ == '__main__':
    sys.exit(main())


