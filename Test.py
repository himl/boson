""" Testing different ideas """
from SVM.DataHandler import DataHandler
from SVM.EvaluatingEstimator import cross_validation
from sklearn import neighbors
from sklearn import svm


data_handler = DataHandler()
# Columns with big correlation (more 90%):
# "DER_sum_pt", "PRI_met_sumet", "PRI_jet_all_pt"

remove_columns_names = ("EventId", "PRI_met_sumet", "PRI_jet_all_pt") # 0.73+0.03
# remove_columns_names = ("EventId", "DER_sum_pt", "PRI_jet_all_pt") # 0.74+0.03
# remove_columns_names = ("EventId", "DER_sum_pt", "PRI_met_sumet") # 0.74 (+/- 0.04) 2369.93
training_data, targets, test_data = data_handler.get_pretreated_data(
    training_samples_size=5000, test_samples_size=5000, remove_columns_names=remove_columns_names)

# estimator = svm.SVC(kernel='linear', C=1)
# estimator = neighbors.KNeighborsClassifier(n_neighbors=12)
estimator = svm.SVC(kernel='rbf', C=1, gamma=0.0001)
mean, standart_deviation, time = cross_validation(estimator, training_data, targets)
print("Accuracy: %0.2f (+/- %0.2f)" % (mean, standart_deviation))
print("Time: %0.2f" % time)
