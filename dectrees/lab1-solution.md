# KTH ML Lab1

## Assignment 0

## Assignment 1

```py
import monkdata as m

datasets = ("MONK-1", m.monk1, m.monk1test), ("MONK-2", m.monk2, m.monk2test), ("MONK-3", m.monk3, m.monk3test)
print("Entropies:")
for dataset, _, _ in datasets:
    print(entropy(dataset))
```

| Dataset |       Entropy      |
|:-------:|:------------------:|
| MONK-1  |                1.0 |
| MONK-2  |  0.957117428264771 |
| MONK-3  | 0.9998061328047111 |

## Assignment 2

Entropy is the expectence of information. The entropy of a uniform distribution is always higher than the entropy of a non-uniform distrubution. This is because of the fact that probabilites for some outcomes are higher than other in a non-uniform distribution. When coin flip taken for example, the entropy of a fair coin with 50%-50% probability is equal to 1, for a 80%-20% it lowers to 0.72 as the outcome is more certain and carries less information. For a 100%-0% unfair coin, the entropy is 0 as there is no new information if an event happens.

## Assignment 3

```py
print(f"Dataset,{','.join([str(a) for a in m.attributes])}")
for name, dataset, _ in datasets:
    print(name, end="")
    for a in m.attributes:
        print(f",{averageGain(dataset, a)}", end="")
    print()
```

Information gains:
| Dataset | A1                | A2                | A3                | A4                | A5                | A6                |
|---------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| MONK-1  | 0.075272555608319 | 0.005838429962909 | 0.004707566617297 | 0.026311696507682 | 0.287030749715784 | 0.000757855715864 |
| MONK-2  | 0.003756177377512 | 0.002458498666083 | 0.001056147715892 | 0.015664247292644 | 0.017277176937918 | 0.006247622236881 |
| MONK-3  | 0.007120868396072 | 0.293736173508389 | 0.000831114044534 | 0.002891817288654 | 0.255911724619728 | 0.007077026074097 |

One should choose the highest information gain, which means that A5, A5 and A2 need to be chosen for MONK-1, MONK-2, MONK-3 repsectively.

## Assignment 4

The information gain is used as a heuristic because of its intention of maximazing the information gained by picking a particular attribute. This is, firstly, placing the most information impactful features at the top of the tree, which could enhance the generalization ability, and secondly, expected to make the subtrees more simple to construct, as their relative entropies are lowest possible.

## Assignment 5

```py
import dtree as d
from drawtree_qt5 import drawTree
from dtree import entropy, averageGain, mostCommon, select
from monkdata import Attribute

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

# MONK-2 of max depth two using provided ID3 implementation
t = d.buildTree(m.monk1, m.attributes, maxdepth=2)
print(t)
drawTree(t)
```

```py
print(f"Dataset,E_train, E_test")
for name, train, test in datasets:
    t = d.buildTree(train, m.attributes)
    print(f"{name},{d.check(t, train)},{d.check(t, test)}")
print()
```

| Dataset | E_train | E_test |
|---------|---------|--------|
| MONK-1  | 0       | 0.1713 |
| MONK-2  | 0       | 0.3079 |
| MONK-3  | 0       | 0.0556 |

As a complete tree is being constructed, it is not suprising that train reached accuracy of 100%.

## Assignment 6

More pruning leads to lower variance and higher bias. No or not su much pruning gives low bias and high variance. The golden mean generalizes usually best. One must experiment to find the best combination, as it is a tradeoff where one cannot have both low bias and low variance at once.

## Assignment 7

```py
TODO
```

![](monk1test.png)

![](monk3test.png)

TBD
