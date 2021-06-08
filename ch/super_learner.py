import xgboost as xgb
from mlens.ensemble import SuperLearner
from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ch.util import cv_outer, cv_inner


def super_learner_1(name, x, y):
    clf_name = "super_learner_1"

    models = [
        LogisticRegression(random_state=1, solver='liblinear'),
        DecisionTreeClassifier(random_state=1),
        SVC(random_state=1, gamma='scale', probability=True),
        GaussianNB(),
        KNeighborsClassifier(),
        AdaBoostClassifier(random_state=1),
        BaggingClassifier(random_state=1, n_estimators=10),
        RandomForestClassifier(random_state=1, n_estimators=10),
        ExtraTreesClassifier(random_state=1, n_estimators=10),
        xgb.XGBClassifier(random_state=1, colsample_bytree=0.7, eta=0.2, gamma=0.3, max_depth=10, min_child_weight=1),
    ]

    x, y = x.to_numpy(), y.to_numpy()
    ensemble = SuperLearner(scorer=accuracy_score, folds=cv_inner.get_n_splits(), shuffle=True, sample_size=len(x))
    ensemble.add(models)
    meta_learner = LogisticRegression(solver='lbfgs')
    ensemble.add_meta(meta_learner)
    ensemble.fit(x, y)

    scores = cross_val_score(ensemble, x, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)

    print(ensemble.data)
    best = ensemble.fit(x, y)
    cm = metrics.confusion_matrix(y, best.predict(x))
    acc, stddev = mean(scores), std(scores)
    result = {
        "name": clf_name,
        "acc": acc,
        "std": stddev,
        "cm": cm,
        "params": {},
        "clf": best
    }
    return result
