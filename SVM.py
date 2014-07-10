# -*- coding: UTF-8 -*-
""" Linear SVM
    (to see http://scikit-learn.org/stable/modules/svm.html#classification). """

from SVM.DataHandler import DataHandler
from SVM.EvaluatingEstimator import cross_validation
from sklearn import svm

if __name__ == "__main__":
    data_handler = DataHandler()
    all_data, all_targets = data_handler.read_training_data()

    samples_size = 15000
    data = all_data[-samples_size:]
    targets = all_targets[-samples_size:]

    estimator = svm.SVC(kernel='linear', C=1)
    mean, standart_deviation, time = cross_validation(estimator, data, targets)

    print("Accuracy: %0.2f (+/- %0.2f)" % (mean, standart_deviation))
    print("Time: %0.2f" % time)
