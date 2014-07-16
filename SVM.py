# -*- coding: UTF-8 -*-
""" SVM
    (to see http://scikit-learn.org/stable/modules/svm.html#classification). """

from SVM.DataHandler import DataHandler
from SVM.EvaluatingEstimator import cross_validation_for_grid
from sklearn import svm
from sklearn.grid_search import GridSearchCV


def find_best_linear_param(data, targets):
    # param_grid = [{'C': [1, 10, 100, 1000]}]
    param_grid = [{'C': [1, 2, 3, 4, 5, 6, 7, 8, 9]}]
    estimator = GridSearchCV(svm.SVC(kernel='linear'), param_grid)
    return cross_validation_for_grid(estimator, data, targets)


if __name__ == "__main__":
    data_handler = DataHandler()
    all_data, all_targets = data_handler.read_training_data()

    samples_size = 5000
    data = all_data[-samples_size:]
    targets = all_targets[-samples_size:]

    mean, standart_deviation, time = find_best_linear_param(data, targets)

    # estimator = svm.SVC(kernel='linear', C=1)
    # mean, standart_deviation, time = cross_validation(estimator, data, targets)

    print("Accuracy: %0.2f (+/- %0.2f)" % (mean, standart_deviation))
    print("Time: %0.2f" % time)

