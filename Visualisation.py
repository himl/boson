from SVM.DataHandler import DataHandler
import matplotlib.pyplot as plt


EMPTY_VALUE = -999.0


def skip_empty_values(column):
    return [value for value in column if value != EMPTY_VALUE]


def by_one_features(training_data, training_targets, test_data, headers):
    signals, backgrounds = split_training_data(training_data, training_targets)

    import numpy as np
    subplots_number = 5
    for index in range(training_data.shape[1]):
        corrected_signals = skip_empty_values(signals[:, index])
        corrected_backgrounds = skip_empty_values(backgrounds[:, index])
        corrected_test_data = skip_empty_values(test_data[:, index])

        min_value = min(0, min(corrected_signals), min(corrected_backgrounds),
                        min(corrected_test_data))
        max_value = max(max(corrected_signals), max(corrected_backgrounds),
                        max(corrected_test_data))

        fig = plt.figure(figsize=(15, 12))

        # signals
        ax = fig.add_subplot(subplots_number, 1, 1)
        ax.set_title("signals", size='14')
        plt.xlim(min_value, max_value)
        plt.yticks([])
        plt.scatter(corrected_signals, np.random.randn(len(corrected_signals)), alpha=0.5,
                    color='green')

        ax = fig.add_subplot(subplots_number, 1, 2)
        ax.set_title("signals", size='14')
        plt.xlim(min_value, max_value)
        plt.hist(corrected_signals, color='green')

        # backgrounds
        ax = fig.add_subplot(subplots_number, 1, 3)
        ax.set_title("backgrounds", size='14')
        plt.xlim(min_value, max_value)
        plt.yticks([])
        plt.scatter(corrected_backgrounds, np.random.randn(len(corrected_backgrounds)),
                    color='blue')

        ax = fig.add_subplot(subplots_number, 1, 4)
        ax.set_title("backgrounds", size='14')
        plt.xlim(min_value, max_value)
        plt.hist(corrected_backgrounds, 10, color='blue')

        # test
        ax = fig.add_subplot(subplots_number, 1, 5)
        ax.set_title("test", size='14')
        plt.xlim(min_value, max_value)
        plt.yticks([])
        plt.scatter(corrected_test_data, np.random.randn(len(corrected_test_data)),
                    color='yellow')

        fig.tight_layout()
        fig.savefig('reports/by_one_feature/' + str(index) + "_" + headers[index] + '.png',
                    dpi=120)
        print("Column # " + str(index) + " " + headers[index] + ": ok!")


def skip_empty_values_from_pair(first_column, second_column):
    corrected_first_column = [first_value for index, first_value in enumerate(first_column)
                              if first_value != EMPTY_VALUE and
                              second_column[index] != EMPTY_VALUE]
    corrected_second_column = [second_value for index, second_value in enumerate(second_column)
                               if second_value != EMPTY_VALUE and
                               first_column[index] != EMPTY_VALUE]
    return corrected_first_column, corrected_second_column


def by_pair_features(training_data, training_targets, test_data, headers):
    signals, backgrounds = split_training_data(training_data, training_targets)

    subplots_number = 3
    for first_feature in range(training_data.shape[1]):
        for second_feature in range(first_feature + 1, training_data.shape[1]):
            signals_corrected_first_feature, signals_corrected_second_feature = \
                skip_empty_values_from_pair(signals[:, first_feature], signals[:, second_feature])

            backgrounds_corrected_first_feature, backgrounds_corrected_second_feature = \
                skip_empty_values_from_pair(backgrounds[:, first_feature], backgrounds[:,
                                                                           second_feature])

            test_corrected_first_feature, test_corrected_second_feature = \
                skip_empty_values_from_pair(test_data[:, first_feature], test_data[:,
                                                                         second_feature])

            min_first_value = min(min(signals_corrected_first_feature),
                                  min(backgrounds_corrected_first_feature),
                                  min(test_corrected_first_feature), 0)
            max_first_value = max(max(signals_corrected_first_feature),
                                  max(backgrounds_corrected_first_feature),
                                  max(test_corrected_first_feature))

            min_second_value = min(min(signals_corrected_second_feature),
                                  min(backgrounds_corrected_second_feature),
                                  min(test_corrected_second_feature), 0)
            max_second_value = max(max(signals_corrected_second_feature),
                                  max(backgrounds_corrected_second_feature),
                                  max(test_corrected_second_feature))

            fig = plt.figure(figsize=(15, 12))

            # signals
            ax = fig.add_subplot(subplots_number, 1, 1)
            ax.set_title("signals", size='14')

            plt.xlim(min_first_value, max_first_value)
            ax.set_ylabel(headers[first_feature], size='12')

            plt.ylim(min_second_value, max_second_value)
            ax.set_xlabel(headers[second_feature], size='12')

            ax.scatter(signals_corrected_first_feature, signals_corrected_second_feature,
                       alpha=0.5, color='green')

            # backgrounds
            ax = fig.add_subplot(subplots_number, 1, 2)
            ax.set_title("backgrounds", size='14')

            plt.xlim(min_first_value, max_first_value)
            ax.set_ylabel(headers[first_feature], size='12')

            plt.ylim(min_second_value, max_second_value)
            ax.set_xlabel(headers[second_feature], size='12')

            ax.scatter(backgrounds_corrected_first_feature, backgrounds_corrected_second_feature,
                       alpha=0.5, color='blue')

            # test
            ax = fig.add_subplot(subplots_number, 1, 3)
            ax.set_title("test", size='14')

            plt.xlim(min_first_value, max_first_value)
            ax.set_ylabel(headers[first_feature], size='12')

            plt.ylim(min_second_value, max_second_value)
            ax.set_xlabel(headers[second_feature], size='12')

            ax.scatter(test_corrected_first_feature, test_corrected_second_feature, alpha=0.5,
                       color='yellow')

            fig.tight_layout()
            fig.savefig('reports/by_pair_features/' +
                        str(first_feature) + " " + headers[first_feature] + "-" +
                        str(second_feature) + " " + headers[second_feature] + '.png', dpi=120)
            print("Columns # " + str(first_feature) + "_" + headers[first_feature] + "-" +
                  str(second_feature) + "_" + headers[second_feature] + ": ok!")


def split_training_data(data, targets):
    signals_indices = [index for index, value in enumerate(targets) if value == 's']
    backgrounds_indices = [index for index, value in enumerate(targets) if value == 'b']
    return data[signals_indices], data[backgrounds_indices]


if __name__ == "__main__":
    data_handler = DataHandler()
    training_data, training_targets = data_handler.get_training_data()
    test_data = data_handler.get_test_data()
    headers = data_handler.get_headers()

    # by_one_features(training_data, training_targets, test_data, headers)
    by_pair_features(training_data, training_targets, test_data, headers)