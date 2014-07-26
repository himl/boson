from SVM.DataHandler import DataHandler
import matplotlib.pyplot as plt


EMPTY_VALUE = -999.0


def skip_empty_values(column):
    return [value for value in column if value != EMPTY_VALUE]


def scatterplot(data, data_name, c=None):
    """Makes a scatterplot matrix:
     Inputs:
         data - a list of data [dataX, dataY,dataZ,...];
                  all elements must have same length
         data_name - a list of descriptions of the data;
                  len(data) should be equal to len(data_name)
    Output:
         fig - matplotlib.figure.Figure Object
    """
    N = len(data)
    fig = plt.figure(figsize=(15, 12))
    for i in range(N):
        for j in range(N):
            ax = fig.add_subplot(N, N, i*N+j+1)
            if j == 0:
                ax.set_ylabel(data_name[i],size='12')
            if i == 0:
                ax.set_title(data_name[j],size='12')
            if i == j:
                ax.hist(data[i], 10)
            else:
                corrected_first_column = skip_empty_values(data[j])
                corrected_second_column = skip_empty_values(data[i])
                if c is not None:
                    ax.scatter(corrected_first_column, corrected_second_column, c=c, alpha=0.5,
                               cmap='rainbow')
                else:
                    ax.scatter(corrected_first_column, corrected_second_column, alpha=0.5,
                               cmap='rainbow')
    return fig


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


def split_training_data(data, targets):
    signals_indices = [index for index, value in enumerate(targets) if value == 's']
    backgrounds_indices = [index for index, value in enumerate(targets) if value == 'b']
    return data[signals_indices], data[backgrounds_indices]


if __name__ == "__main__":
    data_handler = DataHandler()
    training_data, training_targets = data_handler.get_training_data()
    test_data = data_handler.get_test_data()
    headers = data_handler.get_headers()

    by_one_features(training_data, training_targets, test_data, headers)
