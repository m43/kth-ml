import os
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

import dtree as d
import monkdata as m
from drawtree_qt5 import drawTree
from dtree import entropy, averageGain, mostCommon, select, buildTree, check
from monkdata import Attribute

datasets = ("MONK-1", m.monk1, m.monk1test), ("MONK-2", m.monk2, m.monk2test), ("MONK-3", m.monk3, m.monk3test)


def assignment_1():
    print("Entropies:")
    for name, dataset, _ in datasets:
        print(f"{name},{entropy(dataset)}")
    print()


def assignment_3():
    print("Information gains")
    print(f"Dataset,{','.join([str(a) for a in m.attributes])}")
    for name, dataset, _ in datasets:
        print(name, end="")
        for a in m.attributes:
            print(f",{averageGain(dataset, a)}", end="")
        print()
    print()


def assignment_5_1():
    # MONK-1 by hand for depth of two
    a0 = max([(averageGain(m.monk1, a), a) for a in m.attributes])[1]
    first_subsets = [select(m.monk1, a0, v) for v in a0.values]
    a1 = [max([(averageGain(subset, a), a) for a in m.attributes], key=lambda x: x[0])[1] if entropy(
        subset) else mostCommon(subset) for subset in first_subsets]
    all_leafs = [[(v, mostCommon(select(s, a, v))) for v in a.values] if isinstance(a, Attribute) else [] for a, s in
                 zip(a1, first_subsets)]
    print(f"First Node:{a0}")
    for v, a, leafs in zip(a0.values, a1, all_leafs):
        if isinstance(a, Attribute):
            print(f"\tFor {v} ask for {a}")
            print('\n'.join([f"\t\tFor {v} its {y}" for v, y in leafs]))
        else:
            print(f"\tFor {v} its {a}")
    t = d.buildTree(m.monk1, m.attributes, maxdepth=2)
    print(t)
    drawTree(t)


def assignment_5_2():
    print("Full tree performance on MONK datsets")
    print(f"Dataset,E_train, E_test")
    for name, train, test in datasets:
        t = d.buildTree(train, m.attributes)
        print(f"{name},{1 - d.check(t, train)},{1 - d.check(t, test)}")
    print()


def assignment_5_3(dataset):
    t = d.buildTree(dataset, m.attributes)
    print(t)
    drawTree(t)


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    break_point = int(len(ldata) * fraction)
    return ldata[:break_point], ldata[break_point:]


def train_tree_with_reduced_error_pruning(dataset, test, fraction):
    train, valid = partition(dataset, fraction)

    t = buildTree(train, m.attributes)
    t_acc = check(t, valid)
    no_pruning_acc = check(t, test)

    while True:
        pruned_trees = d.allPruned(t)
        pruned_trees_with_accuracies = [(check(t, valid), t) for t in pruned_trees]
        if not pruned_trees_with_accuracies:
            break

        best_tree = max(pruned_trees_with_accuracies, key=lambda x: x[0])
        if best_tree[0] <= t_acc:
            break

        t_acc, t = best_tree

    return 1 - check(t, train), 1 - check(t, valid), 1 - check(t, test), 1 - no_pruning_acc


def assignment_7(task):
    loops, run_name = task

    datasets_to_use = datasets[0], datasets[2]
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = {}
    no_pruning = {}
    for dataset in datasets_to_use:
        results[dataset[0]] = {}
        results[dataset[0]]["train error"] = []
        results[dataset[0]]["validation error"] = []
        results[dataset[0]]["test error"] = []
        results[dataset[0]]["no pruning"] = []
        no_pruning[dataset[0]] = 1 - check(buildTree(dataset[1], m.attributes), dataset[2])
        for fraction in fractions:
            results[dataset[0]]["train error"].append([])
            results[dataset[0]]["validation error"].append([])
            results[dataset[0]]["test error"].append([])
            results[dataset[0]]["no pruning"].append([])
            for _ in tqdm(range(loops)):
                train_err, valid_err, test_err, no_pruning_err = train_tree_with_reduced_error_pruning(dataset[1],
                                                                                                       dataset[2],
                                                                                                       fraction)
                results[dataset[0]]["train error"][-1].append(train_err)
                results[dataset[0]]["validation error"][-1].append(valid_err)
                results[dataset[0]]["test error"][-1].append(test_err)
                results[dataset[0]]["no pruning"][-1].append(no_pruning_err)

    print(results)
    print(no_pruning)
    for d in results.keys():
        for acc_type in ["test error"]:
            plt.figure(figsize=[6, 6])
            plt.title(f'{d} dataset {acc_type} ({loops} tests per fraction)')
            # plt.boxplot(results[d][acc_type], positions=fractions, meanline=True,
            #             widths=[0.05]*len(results[d][acc_type]))
            plt.plot(fractions, [no_pruning[d] for _ in fractions], color="green", label="no pruning")
            plt.errorbar(fractions, [statistics.mean(x) for x in results[d][acc_type]],
                         yerr=[statistics.stdev(x) for x in results[d][acc_type]],
                         c='red', label='mean and std values w/ pruning', markersize=4, capsize=3, marker='o',
                         capthick=2)
            plt.errorbar(fractions, [statistics.mean(x) for x in results[d]["no pruning"]],
                         yerr=[statistics.stdev(x) for x in results[d]["no pruning"]],
                         c='orange', label='mean and std values w/out pruning when using the fraction', markersize=4,
                         capsize=3, marker='o', capthick=2)

            plt.legend()
            # plt.xlim(left=0.25, right=0.85)
            plt.ylim(bottom=0, top=0.5)
            plt.xlabel('Fraction of training data')
            plt.ylabel("Error (1 - accuracy)")

            dirname = Path(f"{d}")
            if not dirname.is_dir():
                dirname.mkdir(parents=True, exist_ok=False)
            plt.savefig(os.path.join(d, f"{run_name + '_' if run_name else ''}{acc_type}_{loops}.png"), dpi=420)
            plt.show()
            plt.close()


if __name__ == '__main__':
    # assignment_1()
    # assignment_3()
    # assignment_5_1()
    # assignment_5_2()
    # assignment_5_3(datasets[2][1])
    # with multiprocessing.Pool() as pool:
    #     idx = [f"{i + 1:03d}" for i in range(8)]
    #     tasks = list(zip([5000] * len(idx), idx))
    #     pool.map(assignment_7, tasks)
    #
    assignment_7((1000, "009 "))
    # assignment_7(5000, "002 ")
    # assignment_7(5000, "003 ")
    # assignment_7(10000, "004 ")
    # assignment_7(1000, "005 ")
    # assignment_7(5000, "006 ")
    # assignment_7(5000, "007 ")
    # assignment_7(10000, "008 ")
