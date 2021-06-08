from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import *

from ch.util import grid_nested


def gaussian_na(name, x, y):
    clf_name = "gaussian_na"
    clf = GaussianNB()
    param_grid = {
        # yup, nothing really..
    }
    return grid_nested(clf_name, x, y, clf, param_grid)


def adaboost_gaussian_na(name, x, y):
    clf_name = "adaboost_gaussian_na"
    base_clf = GaussianNB()
    clf = AdaBoostClassifier(base_estimator=base_clf, random_state=1)
    param_grid = {
        # "base_estimator__criterion": ["gini", "entropy"],
        #   "base_estimator__splitter": ["best", "random"],
        # "base_estimator__max_depth": list(range(3, 15)),
        'n_estimators': [5, 10, 15, 25, 50, 75, 500],
    }
    return grid_nested(clf_name, x, y, clf, param_grid)
