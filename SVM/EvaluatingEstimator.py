# -*- coding: UTF-8 -*-

DEFAULT_FOLDS_NUMBER = 5

def CrossValidation(estimator, data, target, foldsNumber = DEFAULT_FOLDS_NUMBER):
    """ This function used "K-fold Cross Validation"
    
    "KFold divides all the samples: k groups of samples, called folds 
    (if k = n, this is equivalent to the Leave One Out strategy), of equal sizes (if possible). 
    The prediction function is learned using k - 1 folds, and the fold left out is used for test."
    
    to see http://scikit-learn.org/stable/modules/cross_validation.html#k-fold 
    
    This function return three objects:
    'meanScore' is the mean of percent of right predicted samples from a test;
    'standardDeviation' is a standard deviation from the mean value;
    'time' is a number of work this function()    """
    
    
    from sklearn.cross_validation import KFold
    kf = KFold(len(target), n_folds=foldsNumber)

    # 'scores' is numpy array. An index is a number of a fold. A value is a percent of right
    # predicted samples from a test.
    import numpy as np
    scores = np.empty(foldsNumber)

    import time
    start = time.time()

    index = 0
    for train, test in kf:
        x_train, x_test, y_train, y_test = data[train], data[test], target[train], target[test]

        clf = estimator.fit(x_train, y_train)
        scores[index] = clf.score(x_test, y_test)
        index += 1
        print("Iteration %d from %d has done!" % (index, foldsNumber))
    finish = time.time()
    
    return scores.mean(), scores.std() * 2, (finish - start)
