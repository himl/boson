# -*- coding: UTF-8 -*-
""" k - Nearest Neighbors
    see http://scikit-learn.org/stable/modules/neighbors.html """

from SVM.DataHandler import DataHandler
from SVM.EvaluatingEstimator import cross_validation
from sklearn import neighbors


def learn_by_one_feature(data, targets, estimator):
    for columnNumber in range(data.shape[1]):
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

    # best_mean = 0
    # best_standart_deviation = 0
    # best_time = 0
    # best_neighbors_number = 0
    # for neighbors_number in range(1, 500):
    #     estimator = neighbors.KNeighborsClassifier(n_neighbors=neighbors_number)
    #     mean, standart_deviation, time = cross_validation(estimator, data, targets)
    #     if mean > best_mean:
    #         best_neighbors_number = neighbors_number
    #         best_mean = mean
    #         best_standart_deviation = standart_deviation
    #         best_time = time
    #     print(neighbors_number)
    #
    # print("Neighbors number: %d" % best_neighbors_number)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (best_mean, best_standart_deviation))
    # print("Time: %0.2f" % best_time)

    estimator = neighbors.KNeighborsClassifier(n_neighbors=12)
    learn_by_one_feature(data, targets, estimator)