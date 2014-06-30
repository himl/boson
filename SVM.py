# -*- coding: UTF-8 -*-
""" Linear SVM
    (to see http://scikit-learn.org/stable/modules/svm.html#classification). """

import DataHandler as dh
dataHandler = dh.DataHandler()
samplesSize = 400
data, target = dataHandler.ReadTrainingData(samplesSize=samplesSize)

from sklearn import svm
estimator = svm.SVC(kernel='linear', C=1)
import EvaluatingEstimator as ee
mean, standartDeviation, time = ee.CrossValidation(estimator, data, target)

print "Accuracy: %0.2f (+/- %0.2f)" % (mean, standartDeviation)
print "Time: %0.2f" % time
