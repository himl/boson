import pandas as pd


EMPTY_VALUE = -999.0


def signals_backgrounds_correlation(signals, backgrounds):
    """ This function makes KS-test
        http://ru.wikipedia.org/wiki/%D0%9A%D1%80%D0%B8%D1%82%D0%B5%D1%80%D0%B8%D0%B9_%D1%81%D0%BE%D0%B3%D0%BB%D0%B0%D1%81%D0%B8%D1%8F_%D0%9A%D0%BE%D0%BB%D0%BC%D0%BE%D0%B3%D0%BE%D1%80%D0%BE%D0%B2%D0%B0
        for two samples (signals and backgrouds) for every column.
        Every column of a training data splits of two samples(signals and backgrouds) and
        for every column checks homogeneity of the samples (signals and backgrouds). """
    from scipy.stats import ks_2samp
    for column in signals:
        first_sample = signals[column][signals[column] != EMPTY_VALUE]
        second_sample = backgrounds[column][backgrounds[column] != EMPTY_VALUE]
        print column, ks_2samp(first_sample, second_sample)
    print "KS-test have finished!"


def correlation_matrix(data):
    """ Pearson correlation for any two columns from data """
    result_matrix = dict.fromkeys(data.keys())
    from scipy.stats import pearsonr
    for first_column in data:
        # result_matrix[first_column] = dict.fromkeys(data.keys())
        first_sample = data[first_column][data[first_column] != EMPTY_VALUE]
        for second_column in data:
            second_sample = data[second_column][data[second_column] != EMPTY_VALUE]
            min_rows = min(first_sample.shape[0], second_sample.shape[0])
            # result_matrix[first_column][second_column] = pearsonr(
            #     first_sample[:min_rows], second_sample[:min_rows])
            result = pearsonr(first_sample[:min_rows], second_sample[:min_rows])
#            if abs(result[0]) > 0.5 and first_column != second_column:
            print first_column, second_column, result
    print "Pearson correlation coefficients have calculated!"

import time
start = time.time()

trainingData = pd.read_csv("../resources/training.csv")

signals = trainingData[trainingData.Label == 's'].drop(['EventId', 'Weight', 'Label'], 1)
backgrounds = trainingData[trainingData.Label == 'b'].drop(['EventId', 'Weight', 'Label'], 1)

signals_backgrounds_correlation(signals, backgrounds)
# correlation_matrix(signals)
# correlation_matrix(backgrounds)

finish = time.time()
print "Total time:", finish - start