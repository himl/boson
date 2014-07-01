# -*- coding: UTF-8 -*-


class DataHandler:
    """ This class gets data from file. """

    __TRAINING_SAMPLES_SIZE = 250000
    __TEST_SAMPLES_SIZE = 550000
    __EMPTY_VALUE = -999.0
    __DEFAULT_PATH_TO_TRAINING_FILE = 'training/training.csv'
    __DEFAULT_PATH_TO_TEST_FILE = 'test/test.csv'
    
    def __init__(self):
        pass

    def RemoveMarksAboutEmptyValues(self, data):
        """ This method gets data and replaces __EMPTY_VALUE value to object None. """
        
        import numpy as np
        dataNew = np.ndarray(shape=data.shape, dtype=data.dtype)
 
        ACCURACY = 0.00001

        sampleNumber = 0
        for features in data:
            featureIndex = 0
            for value in features:
                if abs(self.__EMPTY_VALUE - value) > ACCURACY:
                    dataNew[sampleNumber][featureIndex] = value
                else:
                    dataNew[sampleNumber][featureIndex] = None
                featureIndex += 1
            sampleNumber += 1
        print "Remove marks about empty values has done!"
        return dataNew
    
    def ReadTrainingData(self, traingFile = __DEFAULT_PATH_TO_TRAINING_FILE,
        samplesSize = __TRAINING_SAMPLES_SIZE):
        """" This method returns two object:
        'data' is two dimentions numpy array. The first index is a sample number,
        the second index is feature. The both index is numbers.
        'target' is one dimention numpy array. The index is a sample number. """

        import pandas as pd
        trainingData = pd.read_csv(traingFile)

        # The second index is a number of column
        # The first column is 'EventId', the last but one 'Weight', the last 'Label'.
        # These third colmuns must be skip
        data = trainingData.ix[0:samplesSize - 1,1:-2].values
        target = trainingData["Label"][:samplesSize]
        print "Training samples has read!"
#        return self.RemoveMarksAboutEmptyValues(data), target
        return data, target

    def ReadTestData(self, testFile = __DEFAULT_PATH_TO_TEST_FILE,
        samplesSize = __TEST_SAMPLES_SIZE):
        """" This method returns one object:
        'data' is two dimentions numpy array. The first index is a sample number,
        the second index is feature. The both index is numbers. """

        import pandas as pd
        testData = pd.read_csv(testFile)

        # The second index is a number of column
        # The first column is 'EventId' must be skip
        data = testData.ix[0:samplesSize - 1,1:].values
        print "Test samples has read!"
        return data
