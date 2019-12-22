# Music Genre Classification



import os
import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from scipy import stats
import scipy
from itertools import permutations


# A. k-NN classifier: [30 points]
# [25 points] Write a function [est_class] = knearestneighbor(test_data, train_data, train_label, k) that implements a
# k-NN classifier based on the Euclidean distance (L-2 norm). test_data and train_data are np arrays with the
# dimensions from the attached zip.
# [5 points] In your report, clearly explain how you solve the equal distance problem in your implementation and why
# you think your approach makes sense.



def knearestneighbor(test_data: np.array, train_data: np.array, train_label, k) -> np.array:
    numOfTrain = train_data.shape[0]
    
    numOfTest = test_data.shape[0]
    
    est_class = np.zeros(numOfTest)
    take_labels = np.zeros(k)

    for i in range(numOfTest):
        distanceArr = np.zeros(numOfTrain)
        for j in range(numOfTrain):
            #distance between this test point and this train point
            distance = euclidean([test_data[i]],train_data[j])
            distanceArr[j] = distance 
        #sort distance for this test data - small to large - in index
        sorted_index = np.argsort(distanceArr)
        k_index = sorted_index[:k]
        for idx in range(k):
            take_labels[idx] = int(train_label[k_index[idx]])
        mode = scipy.stats.mode(take_labels)
        est_label = mode[0][0]
        est_class[i] = est_label
    
    return np.array(est_class)


# B. Cross Validation: [25 points]
# [20 points] Write a function [avg_accuracy, fold_accuracies, conf_mat] = cross_validate(data, gt_labels, k,
# num_folds) that implements a N-fold cross validation where num_folds defines the N (the number of folds).

#fold_accuracies - accuracy for each fold, length - num_folds
#confusion matrix - 5*5
#avg_accuracy for all folds_accuracies
def cross_validate(data, gt_labels, k, num_folds) -> np.array:
    numberOfData = data.shape[0]

    
    fold_size = np.floor(numberOfData/num_folds)
    
    #num of label types = 5
    fold_accuracies = np.zeros(num_folds)
    conf_mat = np.zeros((5,5))
    
    #for each fold
    for i in range(num_folds):
        #split data and lable into num_folds, 1 fold for testing, numFolds - 1 for training(which is the rest of data)
        testData = data[int(fold_size*i):int(fold_size*(i+1))]
        testLabel = gt_labels[int(fold_size*i):int(fold_size*(i+1))]
        
        trainData1 = data[0:int(fold_size*i)]
        trainData2 = data[int(fold_size*(i+1)):]
        trainData = np.concatenate((trainData1, trainData2),axis = 0)

        trainLabel1 = gt_labels[0:int(fold_size*i)]
        trainLabel2 = gt_labels[int(fold_size*(i+1)):]
        trainLabel = np.concatenate((trainLabel1, trainLabel2))

        
        est_class = knearestneighbor(testData, trainData, trainLabel, k)
        
        #confusion matrix for this fold, y-axis(row index) is actual, x_axis(col index) is est
        this_conf_mat = np.zeros((5,5))
        error = 0
        for j in range(testLabel.shape[0]):
            this_conf_mat[int(testLabel[j])-1][int(est_class[j])-1] += 1
            if (int(testLabel[j])!=int(est_class[j])):
                error += 1   
        #this fold accracy
        fold_accuracies[i] = 1 - error/testLabel.shape[0]
        
        conf_mat = conf_mat + this_conf_mat
    
    avg_accuracy = np.mean(fold_accuracies)
    
    return np.array(avg_accuracy), np.array(fold_accuracies), np.array(conf_mat)


# [5 points] Rank the single best feature using a 3-fold cross validation using the above method from a function
# called [feature_index] = find_best_features(data, labels, k, num_folds). In this you need to call your
# cross_validate() with num_folds = 3, K = 3 for all the features individually and find the feature which performs the
# best. Report your result.


def find_best_features(data, labels, k, num_folds, sel_features=[]):
    feature_to_select = np.arange(10)
    thisBestAcc = 0
    if len(sel_features) == 0:
        for fea in feature_to_select:
            ave_acc, fold_accuracies, conf_mat = cross_validate(data[:, fea.astype(int)], labels, k,
                                                                num_folds)
            if thisBestAcc < ave_acc:
                feature_index = fea.astype(int)
                thisBestAcc = ave_acc
    else:
        for j in range(len(sel_features)):
            found_index = np.in1d(feature_to_select, sel_features[j].astype(int)).nonzero()[0]
            feature_to_select = np.delete(feature_to_select, found_index)
        for fea in feature_to_select:
            this_selected = np.append(sel_features, fea)
            ave_acc, fold_accuracies, conf_mat = cross_validate(data[:, this_selected.astype(int)], labels, k,
                                                                num_folds)
            if thisBestAcc < ave_acc:
                feature_index = fea.astype(int)
                thisBestAcc = ave_acc

    return feature_index


# C. Feature Selection: [25 points]
# [20 points] Write a function [sel_feature_ind, accuracy] = select_features(data, labels, k, num_folds) which computes
# the best feature set based on a sequential forward selection process. accuracy is a vector containing the accuracy
# given the number of selected features (accuracy with the best feature, the best feature pair, etc.).
# [5 points] Run your feature selection  with k = 3 and num_folds = 3. Report the best order of features. Plot how the
# accuracy changes with the feature selection iterations.


def select_features(data, labels, k, num_folds):

    numberOfFeatures = data.shape[1]
    
    accuracy = np.zeros(numberOfFeatures)
    selected_fea = []

    i = 0
    while i < 10:
        feature_index = find_best_features(data, labels, k, num_folds, selected_fea)
        selected_fea = np.append(selected_fea, int(feature_index))
        ave_acc, fold_accuracies, conf_mat = cross_validate(data[:, selected_fea.astype(int)], labels, k, num_folds)
        accuracy[i] = ave_acc
        print('In this round, the best accuracy is', feature_index)
        print('Feature taken in this round:', feature_index)
        print('Accuracy for this round:', accuracy[i])

        i += 1
    
    sel_feature_ind = selected_fea.astype(int)
    print('acclist', accuracy)
    print('sel_fea', sel_feature_ind)
    
    return np.array([sel_feature_ind, accuracy])


# D. Evaluation: [20 points]
# [10 points] Using the best set of features obtained in Part C.2 run your cross_validate() for different values of
# k = 1,3 and 7. Use num_folds = 10 for this. Implement this in a function [accuracies, conf_matrices] = evaluate(data,
# labels). Report the results of your cross validation (accuracy and confusion matrices).
# [10 points] Discuss the confusion matrices obtained in terms of what genres are confused with each other. How does k
# affect the classification performance?


def evaluate(data, labels):
    i_k = [1, 3, 7]
    accuracies = np.zeros(3)
    conf_matrices = np.zeros(3, dtype=object)

    for i in range(0, 3):
        sel_feature_ind, accuracy = select_features(data, labels, i_k[i], 10)
        i_cols=sel_feature_ind[:np.argmax(accuracy)]
        avg_accuracy,fold_accuracies, conf_mat = cross_validate(data[:,i_cols.astype(int)], labels, i_k[i], 10)
        accuracies[i] = avg_accuracy
        conf_matrices[i] = conf_mat

    return accuracies, conf_matrices


# Bonus: [20 points]
# [10 points] Implement a function [c_labels, centroids] = kmeans_clustering(data, k) that implements a k-Means
# Clustering algorithm.
# [10 points] Perform your k-means clustering using k = 5 for the genre data provided to you. Contrast the cluster
# labels with the ground truth genre labels. Discuss your results in terms of how closely genre relates to music
# similarity (based on this particular set of the features).


def euclidean(a,b):
    return np.linalg.norm(a-b, ord=2)


def kmeans_clustering(data, k):

    num_datapoints = data.shape[0]

    # stop conditions
    threshold = 0.00001
    max_iterations = 500

    # randomly initialized centroids
    centroid_idx = np.random.choice(data.shape[0],k, replace=False)
    centroids = data[centroid_idx]

    # variables that get computed below
    c_labels = np.arange(num_datapoints)
    distances = np.zeros([num_datapoints, k])

    for iter in np.arange(max_iterations):
        for n, data_point in enumerate(data):
            for m in np.arange(k):
                distances[n][m] = euclidean(data_point, centroids[m])

        for n, distance in enumerate(distances):
            c_labels[n] = np.argmin(distance)

        # updating the centroid values
        for p in range(k):
            data_idx = np.where(c_labels == p)
            data_at_idx = data[data_idx]

            old_centroid = np.copy(centroids)
            centroids[p] = np.mean(data_at_idx, axis=0)

        error = euclidean(centroids, old_centroid)

        if error < threshold:
            print("The final error is: " + str(error))
            break

    return np.array((c_labels, centroids))


def kmeans_gt_compare(gt_labels,est_labels, k):
    perm = list(permutations(range(k),k))

    gt_labels = gt_labels - 1
    gt_labels = gt_labels.astype(int)

    best_accuracy = 0.
    for label_order in perm:
        # label_order1 = [4,3,2,1,0]
        for j in range(len(est_labels)):
            for i,n in enumerate(label_order):
                if est_labels[j] == i:
                    est_labels[j] = n
                    break

        matches = np.where(gt_labels == est_labels)
        count = len(matches[0])

        accuracy = count / len(gt_labels)

        if best_accuracy < accuracy:
            best_accuracy = accuracy

    print("The accuracy of k-means is: " + str(best_accuracy))

    return


if __name__ == "__main__":
    data = np.loadtxt('data/data.txt')
    labels = np.loadtxt('data/labels.txt')

    # Transpose data first, was 10x500, now 500x10
    data = data.T

    #shuffle data
    numData = data.shape[0]
    indeces = np.random.permutation(numData)
    data = data[indeces]
    labels = labels[indeces]

    accuracies, conf_matrices = evaluate(data, labels)

    print(accuracies, conf_matrices)
