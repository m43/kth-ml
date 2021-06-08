from sklearn.neighbors import KNeighborsClassifier

from ch.util import grid_nested


def knn(name, x, y):
    clf_name = "knn"
    clf = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": list(range(1, 200))
    }
    return grid_nested(clf_name, x, y, clf, param_grid)
