import matplotlib.pyplot as plt
import pydotplus
import xgboost as xgb
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from ch.util import grid_nested


def tree(name, x, y, plot_tree=False):
    clf_name = "tree"
    clf = DecisionTreeClassifier()
    param_grid = {
        "max_depth": list(range(3, 18)),
        # "min_samples_leaf": randint(1, 9),
        "min_samples_leaf": list(range(1, 50, 2)),
        "criterion": ["gini", "entropy"]
    }
    result = grid_nested(clf_name, x, y, clf, param_grid)

    if plot_tree:
        dot_data = StringIO()
        export_graphviz(result["clf"], out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=list(x.columns),
                        class_names=["Atsuto", 'Bob', "Jörg"])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(f'{clf_name}.png')
        Image(graph.create_png())

    return result


def random_forest(name, x, y):
    clf_name = "random_forest"
    clf = RandomForestClassifier(random_state=1)
    param_grid = {
        'n_estimators': [10, 25, 50, 75, 500, 1000],
        'max_features': [1, 2, 4, 6, 8, 10, 16]
    }
    return grid_nested(clf_name, x, y, clf, param_grid)


def adaboost_tree(name, x, y):
    clf_name = "adaboost_tree"
    # base_clf = DecisionTreeClassifier(random_state=1, max_features="auto", max_depth=None)
    # clf = AdaBoostClassifier(base_estimator=base_clf, random_state=1)
    clf = AdaBoostClassifier(random_state=1)
    param_grid = {
        # "base_estimator__criterion": ["gini", "entropy"],
        #   "base_estimator__splitter": ["best", "random"],
        # "base_estimator__max_depth": list(range(3, 15)),
        'n_estimators': [5, 10, 15, 25, 50, 75, 500],
    }
    return grid_nested(clf_name, x, y, clf, param_grid)


def gradientboost_tree(name, x, y):
    clf_name = "gradientboost_tree"
    # base_clf = DecisionTreeClassifier(random_state=1, max_features="auto", max_depth=None)
    clf = GradientBoostingClassifier(random_state=1)
    param_grid = {
        'n_estimators': [5, 10, 15, 25, 50, 75, 500],
    }
    return grid_nested(clf_name, x, y, clf, param_grid)


def xgboost_tree(name, x, y, plot_feature_importance=False):
    clf_name = "xgboost_tree"
    # data_dmatrix = xgb.DMatrix(data=X, label=y)
    # base_clf = DecisionTreeClassifier(random_state=1, max_features="auto", max_depth=None)
    clf = xgb.XGBClassifier(random_state=1)
    param_grid = {
        "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        # "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "max_depth": [3, 5, 8, 10, 12, 15],
        "min_child_weight": [1, 3, 5, 7],
        # "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "gamma": [0.0, 0.1, 0.2, 0.3],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
        'n_estimators': [100, 500],
    }
    # param_grid = {
    #     'n_estimators': [5, 10, 15, 25, 50, 75, 500],
    #     'objective': ["reg:linear", "reg:logistic", "binary:logistic", "binary:logitraw", "count:poisson",
    #                   "multi:softmax", "multi:softprob", "rank:pairwise"]
    # }
    result = grid_nested(clf_name, x, y, clf, param_grid)

    if plot_feature_importance:
        clf_plt = xgb.XGBClassifier(random_state=1)
        clf_plt.fit(x, y)
        xgb.plot_importance(clf_plt, height=0.3, )
        plt.show()

    return result


def extra_trees(name, x, y):
    clf_name = "extra_trees"
    clf = ExtraTreesClassifier(random_state=1)
    param_grid = {
        'max_features': [1, 2, 4, 6, 8, 10, 16],
        "min_samples_leaf": list(range(1, 50, 2)),
        'min_samples_split': range(15,36,5),
        'n_estimators': [5, 10, 20, 50, 75, 100, 500],
    }
    return grid_nested(clf_name, x, y, clf, param_grid)
