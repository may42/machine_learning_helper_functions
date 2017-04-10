# source: https://github.com/may42/machine_learning_helper_functions

import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing


class FeaturePreparer:
    """Prepares features:
    1) Adds powers, permutations, logarithms and exponents;
    2) Normalizes all features mean and std;
    """

    def __init__(self, powers):
        self.powers = powers or [1]
        self.scaler = preprocessing.StandardScaler()
        self.__orig_std = None # needed for scaling features before taking exp

    def apply_powers(self, x):
        f_list = [x ** p for p in self.powers if type(p) is int]
        if "perm" in self.powers:
            f_list += [x[:, i:i+1] * x[:, j:j+1] for i in range(x.shape[1]) for j in range(i + 1)]
        if "log" in self.powers:
            f_list += [np.log(x + 1)]
        if "exp" in self.powers:
            f_list += [np.exp(x / self.__orig_std)]
        return np.concatenate(f_list, axis=1) if len(f_list) > 1 else f_list[0]

    def fit(self, x):
        self.__orig_std = x.std(axis=0)
        x_new = self.apply_powers(x)
        return self.scaler.fit(x_new)

    def transform(self, x):
        x_new = self.apply_powers(x)
        return self.scaler.transform(x_new)

    def fit_transform(self, x):
        self.__orig_std = x.std(axis=0)
        x_new = self.apply_powers(x)
        return self.scaler.fit_transform(x_new)


def plot_learning_curves(est, X, y, title, ylim=(.55, 1.005), cv=3, train_sizes=np.linspace(.05, 1.0, 10)):
    """
    plot the test and training learning curves
    est: estimator - must implement "fit" and "predict" methods
    X: features
    y: target
    title: title for the chart
    ylim: defines minimum and maximum yvalues plotted
    cv: cv strategy
    steps: train portions sizes
    """
    steps, train_sc, test_sc = model_selection.learning_curve(est, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    create_learning_curves_plot(steps, train_sc, test_sc, title, ylim).show()


def create_learning_curves_plot(train_sizes, train_scores, test_scores, title, ylim=(.55, 1.005)):
    plt.figure()
    plt.title(title)
    plt.ylim(*ylim)
    plt.xlabel("number of tr. examples")
    plt.ylabel("score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, '-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, '-', color="g", label="Cv score")

    plt.legend(loc="best")
    return plt

