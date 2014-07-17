# -*- coding: UTF-8 -*-
""" SVM
    (to see http://scikit-learn.org/stable/modules/svm.html#classification). """

from SVM.DataHandler import DataHandler
from SVM.EvaluatingEstimator import cross_validation_for_grid
from SVM.EvaluatingEstimator import cross_validation
from sklearn import svm
from sklearn.grid_search import GridSearchCV


def find_best_linear_param(data, targets):
    # param_grid = [{'C': [1, 10, 100, 1000]}]
    param_grid = [{'C': [1, 2, 3, 4, 5, 6, 7, 8, 9]}]
    estimator = GridSearchCV(svm.SVC(kernel='linear'), param_grid)
    return cross_validation_for_grid(estimator, data, targets)


def find_best_rbf_param(data, targets):
    param_grid = {'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    estimator = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
    return cross_validation_for_grid(estimator, data, targets)


def learn_by_one_feature(data, targets, estimator):
    for columnNumber in range(19, data.shape[1]):
        mean, standart_deviation, time = cross_validation(estimator,
            data[:, columnNumber:columnNumber + 1], targets)
        print("Column number: %d" % columnNumber)
        print("Accuracy: %0.2f (+/- %0.2f)" % (mean, standart_deviation))
        print("Time: %0.2f" % time)


if __name__ == "__main__":
    data_handler = DataHandler()
    all_data, all_targets = data_handler.get_training_data()

    samples_size = 5000
    data = all_data[-samples_size:]
    targets = all_targets[-samples_size:]

    # estimator = svm.SVC(kernel='linear', C=1)
    # estimator = svm.SVC(kernel='rbf', C=1, gamma=0.0001)
    # mean, standart_deviation, time = cross_validation(estimator, data, targets)

    # mean, standart_deviation, time = find_best_linear_param(data, targets)
    # mean, standart_deviation, time = find_best_rbf_param(data, targets)

    # print("Accuracy: %0.2f (+/- %0.2f)" % (mean, standart_deviation))
    # print("Time: %0.2f" % time)

    estimator = svm.SVC(kernel='linear', C=1)
    learn_by_one_feature(data, targets, estimator)
