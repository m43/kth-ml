from sklearn.svm import SVC

from ch.util import grid_nested


def svm(name, x, y):
    clf_name = "svm"
    clf = SVC(decision_function_shape='ovo')
    param_grid = {
        'C': [0.1, 0.5, 1, 2, 5, 10, 100, 1000],
        # 'C':np.logspace(-2, 10, 13),
        # 'gamma':np.logspace(-9, 3, 13),
        'gamma': [3, 1, 0.1, 0.05, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'sigmoid']
    }
    return grid_nested(clf_name, x, y, clf, param_grid)
