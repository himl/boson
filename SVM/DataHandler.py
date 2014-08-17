# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np


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

    def get_training_data(self, training_file=__DEFAULT_PATH_TO_TRAINING_FILE,
                          samples_size=__TRAINING_SAMPLES_SIZE):
        """" This method returns two object:
        'data' is two dimensions numpy array. The first index is a sample number,
        the second index is feature. The both index is numbers.
        'target' is one dimension numpy array. The index is a sample number. """

        training_data = pd.read_csv(training_file)

        # The second index is a number of column
        # The first column is 'EventId', the last but one 'Weight', the last 'Label'.
        # These third columns must be skip
        data = training_data.ix[0:samples_size - 1, 1:-2].values
        targets = training_data["Label"][:samples_size].values
        print("Training samples has read!")
        # return self.remove_empty_values_marks(data), target
        return data, targets

    def split(self, data, targets):
        signals_indices = [index for index, value in enumerate(targets) if value == 's']
        backgrounds_indices = [index for index, value in enumerate(targets) if value == 'b']
        return data[signals_indices], data[backgrounds_indices]

    def get_separate_training_data(self, training_file=__DEFAULT_PATH_TO_TRAINING_FILE,
                                   samples_size=__TRAINING_SAMPLES_SIZE):
        data, targets = self.get_training_data(training_file, samples_size)
        return self.split(data, targets)

    def get_test_data(self, test_file=__DEFAULT_PATH_TO_TEST_FILE,
                      samples_size=__TEST_SAMPLES_SIZE):
        """" This method returns one object:
        'data' is two dimensions numpy array. The first index is a sample number,
        the second index is feature. The both index is numbers. """

        test_data = pd.read_csv(test_file)

        # The second index is a number of column
        # The first column is 'EventId' must be skip
        data = test_data.ix[:samples_size - 1, 1:].values
        print("Test samples has read!")
        return data

    def get_headers(self):
        training_data = pd.read_csv(self.__DEFAULT_PATH_TO_TRAINING_FILE)
        return training_data.columns.values[1:-2]

    def get_info(self):
        training_data = pd.read_csv(self.__DEFAULT_PATH_TO_TRAINING_FILE)
        print("Training data:")
        print(training_data.describe())

        test_data = pd.read_csv(self.__DEFAULT_PATH_TO_TEST_FILE)
        print("Test data:")
        print(test_data.describe())

    def dummy_encode_categorical_columns(self, df, categorical_columns):
        """ Deployment of categorical attributes:
        http://nbviewer.ipython.org/urls/raw2.github.com/Newozz/shad_ml/master/2014_Spring/baseline_competition_1.ipynb """
        from copy import deepcopy
        result_df = deepcopy(df)
        for column in categorical_columns:
            result_df = pd.concat([result_df, pd.get_dummies(
                result_df[column], prefix=column, prefix_sep=': ')], axis=1)
            del result_df[column]
        return result_df

    def get_pretreated_data(self, training_file=__DEFAULT_PATH_TO_TRAINING_FILE,
                            training_samples_size=__TRAINING_SAMPLES_SIZE,
                            test_file=__DEFAULT_PATH_TO_TEST_FILE,
                            test_samples_size=__TEST_SAMPLES_SIZE,
                            remove_columns_names=("EventId",)):
        """ Data without columns with big correlation (>90%) +
            dummy dummy_encode_categorical_columns """
        training_data = pd.read_csv(training_file)
        training_targets = training_data["Label"][:training_samples_size].values
        training_data = training_data.drop('Weight', axis=1)
        training_data = training_data.drop('Label', axis=1)

        test_data = pd.read_csv(test_file)
        print("Data has read!")

        full_data = pd.concat([training_data, test_data])

        categorical_columns = ("PRI_jet_num",)

        dummy_full_data = self.dummy_encode_categorical_columns(full_data,
                                                                categorical_columns)

        dummy_training_data = dummy_full_data[dummy_full_data["EventId"] < 350000]
        dummy_test_data = dummy_full_data[dummy_full_data["EventId"] >= 350000]

        for column_name in remove_columns_names:
            dummy_training_data = dummy_training_data.drop(column_name, axis=1)
            dummy_test_data = dummy_test_data.drop(column_name, axis=1)

        training_samples = dummy_training_data.ix[:training_samples_size - 1, :].values
        test_samples = dummy_test_data.ix[:test_samples_size - 1, :].values

        return training_samples, training_targets, test_samples
