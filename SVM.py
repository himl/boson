# -*- coding: UTF-8 -*-
""" Linear SVM
    (to see http://scikit-learn.org/stable/modules/svm.html#classification). """

from SVM.DataHandler import DataHandler
from SVM.EvaluatingEstimator import CrossValidation
from sklearn import svm

if __name__ == "__main__":
    dataHandler = DataHandler()
    samplesSize = 20
    data, target = dataHandler.ReadTrainingData(samplesSize=samplesSize)


    estimator = svm.SVC(kernel='linear', C=1)
    mean, standartDeviation, time = CrossValidation(estimator, data, target)

    print("Accuracy: %0.2f (+/- %0.2f)" % (mean, standartDeviation))
    print("Time: %0.2f" % time)
