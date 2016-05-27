

# CLUSTERING 
#import matplotlib  
#matplotlib.use('TkAgg')   
#import matplotlib.pyplot as plt  
#from matplotlib.colors import ListedColormap
#from sklearn import metrics
#from sklearn.cluster import KMeans
#from sklearn.datasets import load_digits
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import scale





# crf.state_features_

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
