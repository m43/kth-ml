import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def pairplot(train, save_filename, show=False, close=True):
    sns.set_style("whitegrid")
    sns.pairplot(train, hue="y", size=3)
    if show:
        plt.show()
    if save_filename:
        plt.savefig(save_filename)
    if close:
        plt.close()


if __name__ == '__main__':
    train, test = pd.read_csv("train_clean_2.csv", comment="#"), pd.read_csv("eval.csv")
    pairplot(train, "/home/user72/Desktop/myplot3.png")
