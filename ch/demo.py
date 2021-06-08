import os
import random
import scipy
from datetime import datetime

import ch
from ch.preprocess import *
from ch.util import *

# import multiprocessing
# multiprocessing.set_start_method('forkserver')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
start_time = datetime.now().strftime('%Y-%m-%d--%H:%M:%S')


# TODO list:
#  3. preprocess_3: PCA
#  4. fix unbalanced data

def get_raw_train():
    return pd.read_csv(train_path, comment="#")


def get_raw_test():
    return pd.read_csv(test_path)


if __name__ == '__main__':
    ####################################
    #### PROGRAMMING CHALLENGE DEMO ####
    ####################################

    ##########################
    ## GLOBAL PARAMETERS :) ##
    ##       me likey       ##
    ##########################
    name = f"L "
    scale = True
    ##########################

    # seed
    random.seed(720)
    pd.np.random.seed(720)
    scipy.random.seed(720)
    print(os.listdir())

    # prepare datasets
    # scale = True
    datasets = [
        # (f"p2{'s' if scale else ''}", preprocess_2(get_raw_train(), get_raw_test(), scale, False, False)),
        # (f"p2{'s' if scale else ''}_dropX3", preprocess_2(get_raw_train(), get_raw_test(), scale, True, False)),
        # (f"p2{'s' if scale else ''}_dropX6_1", preprocess_2(get_raw_train(), get_raw_test(), scale, False, True)),
        (f"p2{'s' if scale else ''}_dropX3_dropX6_1", preprocess_2(get_raw_train(), get_raw_test(), scale, True, True)),
        # (f"p1{'s' if scale else ''}", preprocess_1(get_raw_train(), get_raw_test(), scale, False)),
        # (f"p1{'s' if scale else ''}_dropX3", preprocess_1(get_raw_train(), get_raw_test(), scale, True))
    ]
    # scale = False
    # datasets.extend([
    #     (f"p2{'s' if scale else ''}", preprocess_2(get_raw_train(), get_raw_test(), scale, False, False)),
    #     (f"p2{'s' if scale else ''}_dropX3", preprocess_2(get_raw_train(), get_raw_test(), scale, True, False)),
    #     (f"p2{'s' if scale else ''}_dropX6_1", preprocess_2(get_raw_train(), get_raw_test(), scale, False, True)),
    #     (f"p2{'s' if scale else ''}_dropX3_dropX6_1", preprocess_2(get_raw_train(), get_raw_test(), scale, True, True)),
    #     (f"p1{'s' if scale else ''}", preprocess_1(get_raw_train(), get_raw_test(), scale, False)),
    #     (f"p1{'s' if scale else ''}_dropX3", preprocess_1(get_raw_train(), get_raw_test(), scale, True))
    # ])
    # train.to_csv("train_clean_2.csv")

    # prepare calssifiers to be used
    classifiers = [
        # ch.tree.tree,
        # ch.tree.random_forest,
        # ch.tree.adaboost_tree,
        # ch.tree.gradientboost_tree,
        ch.tree.xgboost_tree,
        # ch.knn.knn,
        # ch.svm.svm,
        # ch.bayes.gaussian_na,
        # ch.bayes.adaboost_gaussian_na,
        # ch.super_learner.super_learner_1,
        # ch.tree.extra_trees
    ]

    results = []
    for dataset_name, (train, test, xes) in datasets:
        x, y = train.loc[:, xes], train.loc[:, "y"]
        test_x = test[xes]

        # get results
        for clf in classifiers:
            results += [clf(name, x, y)]
            results[-1]["dataset"] = dataset_name

    # print and save results
    best = pick_best(results)
    print(results)
    results_summary = []
    for r in results:
        line = f'{r["name"]:<30s} d:{r["dataset"]:<15s} acc:{r["acc"]} std:{r["std"]} best_params:{r["params"]}'
        print(line)
        results_summary.append(line)
    print("BEST RESULT")
    print("Accuracy:", best["acc"])
    print(best["cm"])
    test_predict = [y_mapping_reversed[x] for x in best["clf"].fit(x, y).predict(test_x)]
    # test_predict = [y_mapping_reversed[x] for x in best["clf"].fit(x.values, y.values).predict(test_x.values)]
    with open(f"{name} acc={best['acc']:.5f} std={best['std']} d={best['dataset']} clf_name={best['name']}.txt",
              "w") as f:
        f.write("\n".join(results_summary) + "\n\n")
        f.write(f"{start_time}\n\n")
        f.write(f"{results}\n\n")
        f.write("\n".join(test_predict) + "\n")
    print(name)
    print(start_time)
