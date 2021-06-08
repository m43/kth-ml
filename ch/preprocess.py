import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ch.util import y_mapping


def preprocess_1(train, test, scale, drop_x3):
    xes = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    if drop_x3:
        xes.remove("x3")
    grade_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'Fx': 2, 'F': 1}
    # scale_indices = xes[:4] + xes[6:]
    scale_indices = xes
    true_false_mapping = {True: 1, False: 0, "True": 1, "False": 0}

    def _preprocess_both(df):
        df.loc[:, 'x6'] = df.loc[:, 'x6'].map(grade_mapping)
        df.loc[:, 'x5'] = df.loc[:, 'x5'].map(true_false_mapping)

    train.dropna(inplace=True)
    train = train[~train[xes].isin(["?"]).any(axis=1)]
    train.loc[:, 'y'] = train.loc[:, 'y'].map(y_mapping)
    train.loc[:, ['x1', 'x2']] = train.loc[:, ['x1', 'x2']].astype(float)

    _preprocess_both(train)
    _preprocess_both(test)

    train = train[(np.abs(stats.zscore(train)) < 3).all(axis=1)]

    if scale:
        scaler = StandardScaler().fit(train[scale_indices])
        scaler.transform(train[scale_indices])
        scaler.transform(test[scale_indices])

    train = train.drop("id", axis=1)
    test = test.drop("Unnamed: 0", axis=1)

    return train, test, xes


def preprocess_2(train, test, scale, drop_x3, drop_first):
    def dummy_encode_x6(df):
        dummies = pd.get_dummies(df["x6"], drop_first=drop_first)
        dummies.rename(columns=lambda x: "x6_" + str(x), inplace=True)
        df = df.drop("x6", axis=1)
        df = df.join(dummies)
        return df

    train, test, _ = preprocess_1(train, test, scale, drop_x3)
    train, test = dummy_encode_x6(train), dummy_encode_x6(test)

    xes = list(train)
    xes.remove("y")
    return train, test, xes


def preprocess_3(train, test, scale):
    return preprocess_1(train, test, scale)
