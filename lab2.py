import os

import matplotlib.pyplot as plt
import numpy as np
from coverage.misc import ensure_dir
from scipy.optimize import minimize


def generate_dataset_f():
    classA = np.concatenate((
        # np.random.randn(10, 2) * 0.1 + [-1.5, 1.5],
        np.random.randn(1, 2) * 0.1 + [-0.5, 1.5],
        # np.random.randn(10, 2) * 0.1 + [-1, -1.5],
        # np.random.randn(10, 2) * 0.2 + [-4.5, -1.5],
        np.array([[-4, -4], [-4, -3], [-4, -2], [-4, -1], [-4, 0], [-4, 1], [-4, 2], [-4, 3],
                  [-4, 4], [-1, 4], [1, 4], [1, 2], [1, -4], [0, 0.25]
                  # , [-2, -0.5]
                  ])
    ))
    classB = np.concatenate((
        np.random.randn(20, 2) * 0.2 + [0, 2.5],
        np.random.randn(20, 2) * 0.2 + [-3, 2.5],
        np.random.randn(20, 2) * 0.2 + [-2, 2.5],
        np.random.randn(20, 2) * 0.2 + [-1, 2.5],
        np.random.randn(20, 2) * 0.2 + [-3, 1.5],
        np.random.randn(20, 2) * 0.2 + [-3, 0.5],
        np.random.randn(20, 2) * 0.2 + [-2, 0.5],
        np.random.randn(20, 2) * 0.2 + [-1, 0.5],
        np.random.randn(20, 2) * 0.2 + [-3, -0.5],
        np.random.randn(20, 2) * 0.2 + [-3, -1.5],
        np.random.randn(20, 2) * 0.2 + [-3, -2.5]
    ))

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((
        np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])
    ))

    shuffle_indices = np.arange(inputs.shape[0])
    np.random.shuffle(shuffle_indices)
    inputs, targets = inputs[shuffle_indices], targets[shuffle_indices]
    return inputs, targets


def generate_dataset():
    classA = np.concatenate((
        np.random.randn(10, 2) * 0.4 + [-1.5, 0.5],
        np.random.randn(10, 2) * 0.4 + [-1.2, -1.2],
        np.random.randn(10, 2) * 0.4 + [1.5, 0.5],
    ))
    classB = np.concatenate((
        np.random.randn(10, 2) * 0.4 + [0, -0.3],
        np.random.randn(10, 2) * 0.5 + [-2.2, -0.2],
    ))

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((
        np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])
    ))

    shuffle_indices = np.arange(inputs.shape[0])
    np.random.shuffle(shuffle_indices)
    inputs, targets = inputs[shuffle_indices], targets[shuffle_indices]
    return inputs, targets

def generate_dataset_3():
    from sklearn.datasets.samples_generator import make_circles
    X, y = make_circles(90, factor=0.2, noise=0.1)
    inputs = X
    targets = (y-0.5)*2

    shuffle_indices = np.arange(inputs.shape[0])
    np.random.shuffle(shuffle_indices)
    inputs, targets = inputs[shuffle_indices], targets[shuffle_indices]
    return inputs, targets


def scatter_plot_2d_features(inputs, targets, name, indicator=None, svm_indices=tuple([]),
                             save_folder=None, show_plot=True, close_plot=True, luft=2, colormesh=False):
    plt.title(name)
    if indicator is not None:
        xgrid = np.linspace(np.min(inputs[:, 0]) - luft, np.max(inputs[:, 0]) + luft)
        ygrid = np.linspace(np.min(inputs[:, 1]) - luft, np.max(inputs[:, 1]) + luft)
        xy_values = np.array([[indicator([x, y]) for x in xgrid] for y in ygrid])
        if colormesh:
            plt.pcolormesh(xgrid, ygrid, xy_values)
        plt.contour(xgrid, ygrid, xy_values, (-1, 0, 1), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    markers = [("b" if ti.item() == 1 else "r") + ("x" if i in svm_indices[0] else "o") for i, ti in enumerate(targets)]
    for (x, y), m in zip(inputs, markers):
        plt.plot(x, y, m)
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, name) + ".png", dpi=300)
    if show_plot:
        plt.show()
    if close_plot:
        plt.close()


def linear_kernel(x, y):
    return x @ y


def polynomial_kernel(x, y, p):
    return (x @ y + 1) ** p


def rbf_kernel(x, y, sigma):
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))


def demo(name, save_folder, dataset, show_plots=True, kernel=linear_kernel, c=None, threshold=1e-5):
    inputs, targets = dataset
    scatter_plot_2d_features(inputs, targets, f"{name} A - start", None, ([],), save_folder, show_plots)

    print(f"{'#' * 20}  {name}  {'#' * 20}")

    alpha_start = np.zeros(len(inputs))
    b = [(0, c)] * len(inputs)
    xc = {'type': 'eq', 'fun': lambda alpha: np.sum(targets * alpha)}

    precomputed_ttK_matrix = np.zeros((targets.shape[0], targets.shape[0]))
    for i, (xi, ti) in enumerate(zip(inputs, targets)):
        for j, (xj, tj) in enumerate(zip(inputs, targets)):
            precomputed_ttK_matrix[i, j] = ti * tj * kernel(xi, xj)

    def objective(a):
        return np.sum(np.outer(a, a) * precomputed_ttK_matrix) / 2 - np.sum(a)

    print("Minimization started.")
    result = minimize(objective, alpha_start, bounds=b, constraints=xc)
    print(f"Minimization results:\n{result}")
    if not result["success"]:
        print("Minimization was not successful")
        return result

    alpha = result["x"]
    support_vectors_indices = np.where((alpha > threshold) * (alpha < (np.inf if c is None else c - threshold)))
    if len(support_vectors_indices[0]) == 0:
        print("No support vectors found!")
        return result
    indicator_vectors_indices = np.where((alpha > threshold) * (alpha <= (np.inf if c is None else c)))
    indicator_alphas = alpha[indicator_vectors_indices]
    indicator_x = inputs[indicator_vectors_indices]
    indicator_t = targets[indicator_vectors_indices]

    b = 0

    def indicator(x):
        return np.sum(indicator_alphas * indicator_t * np.array([kernel(x, sv_x) for sv_x in indicator_x])) - b

    b = indicator(inputs[support_vectors_indices[0]][0]) - targets[support_vectors_indices[0]][0]

    scatter_plot_2d_features(inputs, targets, f"{name} B - result", indicator, support_vectors_indices, save_folder,
                             show_plots, colormesh=False)
    scatter_plot_2d_features(inputs, targets, f"{name} B - result with colormesh", indicator, support_vectors_indices,
                             save_folder,
                             show_plots, colormesh=True)

    return result


if __name__ == '__main__':
    np.random.seed(72)
    show_plots = False
    save_folder = "imgs06"

    # c = 90
    # c = 9
    c = None
    # c = 1

    kernels = {
        "linear_kernel": lambda x, y: linear_kernel(x, y),
        "poly_kernel_1": lambda x, y: polynomial_kernel(x, y, 1),
        "poly_kernel_2": lambda x, y: polynomial_kernel(x, y, 2),
        "poly_kernel_3": lambda x, y: polynomial_kernel(x, y, 3),
        "poly_kernel_4": lambda x, y: polynomial_kernel(x, y, 4),
        "poly_kernel_5": lambda x, y: polynomial_kernel(x, y, 5),
        "poly_kernel_6": lambda x, y: polynomial_kernel(x, y, 6),
        "poly_kernel_10": lambda x, y: polynomial_kernel(x, y, 10),
        "rbf_kernel_0_1": lambda x, y: rbf_kernel(x, y, 0.1),
        "rbf_kernel_0_2": lambda x, y: rbf_kernel(x, y, 0.2),
        "rbf_kernel_0_5": lambda x, y: rbf_kernel(x, y, 0.5),
        "rbf_kernel_1": lambda x, y: rbf_kernel(x, y, 1),
        "rbf_kernel_2": lambda x, y: rbf_kernel(x, y, 2)
    }
    kernel_name = "rbf_kernel_0_5"
    kernel = kernels[kernel_name]
    # datasets = [generate_dataset() for _ in range(10)]
    datasets = [generate_dataset_3()    ]
    for kernel_name, kernel in kernels.items():
        print(f"--{'#' * 46}--")
        print(f"{(kernel_name + ' ') * 10}")
        print(f"--{'#' * 46}--")
        for i, dataset in enumerate(datasets):
            name = f"c={c}--kernel={kernel_name}--i={i}"
            ensure_dir(save_folder)
            result = demo(
                name,
                dataset=dataset,
                kernel=kernel,
                c=c,
                save_folder=save_folder,
                show_plots=show_plots
            )
