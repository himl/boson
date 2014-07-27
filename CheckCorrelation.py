from SVM.DataHandler import DataHandler


EMPTY_VALUE = -999.0


def skip_empty_values(column):
    return [value for value in column if value != EMPTY_VALUE]


def signals_backgrounds_correlation(headers, signals, backgrounds):
    """ This function makes KS-test
        http://ru.wikipedia.org/wiki/%D0%9A%D1%80%D0%B8%D1%82%D0%B5%D1%80%D0%B8%D0%B9_%D1%81%D0%BE%D0%B3%D0%BB%D0%B0%D1%81%D0%B8%D1%8F_%D0%9A%D0%BE%D0%BB%D0%BC%D0%BE%D0%B3%D0%BE%D1%80%D0%BE%D0%B2%D0%B0
        for two samples (signals and backgrouds) for every column.
        Every column of a training data splits of two samples(signals and backgrouds) and
        for every column checks homogeneity of the samples (signals and backgrouds). """
    from scipy.stats import ks_2samp
    for index, header in enumerate(headers):
        first_sample = skip_empty_values(signals[:, index])
        second_sample = skip_empty_values(backgrounds[:, index])
        print(header, ks_2samp(first_sample, second_sample))
    print("KS-test have finished!")


def pearson_correlation_matrix(headers, data):
    """ Pearson correlation for any two columns from data """
    result_matrix = dict.fromkeys(headers)
    from scipy.stats import pearsonr
    for first_index, first_header in enumerate(headers):
        result_matrix[first_header] = dict.fromkeys(headers)
        first_sample = skip_empty_values(data[:, first_index])
        for second_index, second_header in enumerate(headers):
            second_sample = skip_empty_values(data[:, second_index])

            min_rows_number = min(len(first_sample), len(second_sample))

            result_matrix[first_header][second_header] = pearsonr(
                first_sample[:min_rows_number], second_sample[:min_rows_number])

            print(first_header, second_header, result_matrix[first_header][second_header])
    print("Pearson correlation coefficients have calculated!")
    return result_matrix


def write_to_csv(file_name, headers, data):
    with open(file_name, 'w') as outfile:
        # the first cell is empty
        outfile.write(";" + ";".join(headers) + "\n")
        for first_header in headers:
            outfile.write(first_header)
            for second_header in headers:
                outfile.write(";" + str(data[first_header][second_header][0]) +
                              " " + str(data[first_header][second_header][1]))
            outfile.write("\n")


data_handler = DataHandler()
signals, backgrounds = data_handler.get_separate_training_data()
headers = data_handler.get_headers()

# signals_backgrounds_correlation(headers, signals, backgrounds)

# correlation_matrix_signals = pearson_correlation_matrix(headers, signals)
# file_name = "./reports/correlation/Pearson_correlation_signals.csv"
# write_to_csv(file_name, headers, correlation_matrix_signals)

correlation_matrix_backgrounds = pearson_correlation_matrix(headers, backgrounds)
file_name = "./reports/correlation/Pearson_correlation_backgrounds.csv"
write_to_csv(file_name, headers, correlation_matrix_backgrounds)
