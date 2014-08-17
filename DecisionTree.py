from sklearn import tree
from SVM.DataHandler import DataHandler
from SVM.EvaluatingEstimator import cross_validation
from sklearn.grid_search import GridSearchCV
from SVM.EvaluatingEstimator import cross_validation_for_grid
from sklearn.decomposition import RandomizedPCA


dt = DataHandler()
training_data, targets = dt.get_training_data(samples_size=5000)

# training_data, targets, test_data = dt.get_pretreated_data(training_samples_size=5000,
#                                                            test_samples_size=5000)

# pca = RandomizedPCA(n_components=5, whiten=False).fit(training_data)
# training_data = pca.transform(training_data)


estimator = tree.DecisionTreeClassifier(max_depth=6, min_samples_leaf=9)
mean, standart_deviation, time = cross_validation(estimator, training_data, targets)


# param_grid = [{'max_depth': list(range(3, 20)), 'min_samples_leaf': list(range(5, 10)),
#                'min_samples_split': list(range(1, 5))}]
#
# estimator = tree.DecisionTreeClassifier()
# estimator = GridSearchCV(estimator, param_grid)
# mean, standart_deviation, time = cross_validation_for_grid(estimator, training_data, targets)

print("Accuracy: %0.2f (+/- %0.2f)" % (mean, standart_deviation))
print("Time: %0.2f" % time)

# from sklearn.externals.six import StringIO
# with open("boson.dot", 'w') as f:
#     f = tree.export_graphviz(estimator, out_file=f)

# from sklearn.externals.six import StringIO
# import pydot
# dot_data = StringIO()
# tree.export_graphviz(estimator, out_file=dot_data)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("boson.pdf")