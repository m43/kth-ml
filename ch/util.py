from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train_path = 'train.csv'
test_path = 'eval.csv'
y_mapping = {"Atsuto": 1, "Bob": 2, "Jörg": 3}
y_mapping_reversed = {v: k for k, v in y_mapping.items()}
cv_outer = KFold(n_splits=10, shuffle=True, random_state=72)
cv_inner = KFold(n_splits=5, shuffle=True, random_state=72)

def pick_best(results):
    return sorted(results, key=lambda x: -x["acc"])[0]


def grid_nested(name, x, y, clf, space):
    grid = GridSearchCV(
        estimator=clf,
        param_grid=space,
        scoring='accuracy',
        n_jobs=-1,
        cv=cv_inner,
        refit=True
    )
    scores = cross_val_score(grid, x, y, scoring='accuracy', cv=cv_outer, n_jobs=-1)

    best = grid.fit(x, y)
    cm = metrics.confusion_matrix(y, best.predict(x))
    acc, stddev = mean(scores), std(scores)
    print(f"{'*' * 7} {name} {'*' * 7}")
    print(f"Best params: {best.best_params_}")
    print(f"Best score: {best.best_score_}")
    print(f"Accuracy: {acc:.3f} ({stddev:.3f})")
    print(cm)
    print()

    result = {
        "name": name,
        "acc": acc,
        "std": stddev,
        "cm": cm,
        "params": best.best_params_,
        "clf": best
    }
    return result
