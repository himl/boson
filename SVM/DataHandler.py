# -*- coding: UTF-8 -*-


class DataHandler:
    """ This class gets data from file. """

    __TRAINING_SAMPLES_SIZE = 250000
    __TEST_SAMPLES_SIZE = 550000
    __EMPTY_VALUE = -999.0
    __DEFAULT_PATH_TO_TRAINING_FILE = 'resources/training.csv'
    __DEFAULT_PATH_TO_TEST_FILE = 'resources/test.csv'
    
    def __init__(self):
        pass

    def remove_empty_values_marks(self, data):
        """ This method gets data and replaces __EMPTY_VALUE value to object None. """
        
        import numpy as np
        data_new = np.ndarray(shape=data.shape, dtype=data.dtype)
 
        __ACCURACY = 0.00001

        sample_number = 0
        for features in data:
            feature_index = 0
            for value in features:
                if abs(self.__EMPTY_VALUE - value) > __ACCURACY:
                    data_new[sample_number][feature_index] = value
                else:
                    data_new[sample_number][feature_index] = None
                feature_index += 1
            sample_number += 1
        print("Remove marks about empty values has done!")
        return data_new
    
    def read_training_data(self, training_file=__DEFAULT_PATH_TO_TRAINING_FILE,
                           samples_size=__TRAINING_SAMPLES_SIZE):
        """" This method returns two object:
        'data' is two dimensions numpy array. The first index is a sample number,
        the second index is feature. The both index is numbers.
        'target' is one dimension numpy array. The index is a sample number. """

        import pandas as pd
        training_data = pd.read_csv(training_file)

        # The second index is a number of column
        # The first column is 'EventId', the last but one 'Weight', the last 'Label'.
        # These third columns must be skip
        data = training_data.ix[0:samples_size - 1, 1:-2].values
        target = training_data["Label"][:samples_size].values
        print("Training samples has read!")
        # return self.remove_empty_values_marks(data), target
        return data, target

    def read_test_data(self, test_file=__DEFAULT_PATH_TO_TEST_FILE,
                       samples_size=__TEST_SAMPLES_SIZE):
        """" This method returns one object:
        'data' is two dimensions numpy array. The first index is a sample number,
        the second index is feature. The both index is numbers. """

        import pandas as pd
        test_data = pd.read_csv(test_file)

        # The second index is a number of column
        # The first column is 'EventId' must be skip
        data = test_data.ix[:samples_size - 1, 1:].values
        print("Test samples has read!")
        return data
