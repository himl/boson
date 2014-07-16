# -*- coding: UTF-8 -*-
""" PCA + Linear SVM """

import numpy as np
# from time import time

from SVM.DataHandler import DataHandler
from sklearn.cross_validation import KFold
from sklearn.decomposition import RandomizedPCA
# from sklearn import svm
from sklearn import neighbors


DEFAULT_FOLDS_NUMBER = 5
DEFAULT_COMPONENTS_NUMBER = 10


def pca_estimator(data, targets, estimator, components_number=DEFAULT_COMPONENTS_NUMBER,
                  folds_number=DEFAULT_FOLDS_NUMBER):

    kf = KFold(len(targets), n_folds=folds_number)

    # 'scores' is numpy array. An index is a number of a fold. A value is a percent of right
    # predicted samples from a test.
    scores = np.zeros(folds_number)

    # start = time()

    index = 0
    for train, test in kf:
        x_train, x_test, y_train, y_test = data[train], data[test], targets[train], targets[test]

        pca = RandomizedPCA(n_components=components_number, whiten=True).fit(x_train)
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)

        clf = estimator.fit(x_train_pca, y_train)
        scores[index] = clf.score(x_test_pca, y_test)
        index += 1
        # print("Iteration %d from %d has done! Score: %f" % (index, folds_number,
        #                                                     scores[index - 1]))
    # finish = time()

    # return scores.mean(), scores.std() * 2, (finish - start)
    return scores.mean(), scores.std() * 2


if __name__ == "__main__":
    data_handler = DataHandler()
    all_data, all_targets = data_handler.read_training_data()

    samples_size = 5000
    training_data = all_data[-samples_size:]
    training_targets = all_targets[-samples_size:]

    best_components_number = 0
    best_mean = 0
    best_standart_deviation = 0
    # best_time = 0

    # estimator = svm.SVC(kernel='linear', C=1)
    # estimator = svm.SVC(kernel='rbf', C=1, gamma=0.0001)
    estimator = neighbors.KNeighborsClassifier(n_neighbors=12)

    for n_components in xrange(1, 100):
        mean, standart_deviation = pca_estimator(training_data, training_targets, estimator,
                                                 n_components)
        if mean > best_mean:
            best_components_number = n_components
            best_mean = mean
            best_standart_deviation = standart_deviation
            # best_time = time
        print(n_components)

    # n_components = 10
    # mean, standart_deviation, time = pca_estimator(training_data, training_targets, estimator,
    #                                          n_components)
    # best_components_number = n_components
    # best_mean = mean
    # best_standart_deviation = standart_deviation
    # best_time = time

    print("N_components: %d" % best_components_number)
    print("Accuracy: %0.2f (+/- %0.2f)" % (best_mean, best_standart_deviation))
    # print("Time: %0.2f" % best_time)
