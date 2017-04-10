import numpy as np
from sklearn import datasets, model_selection, linear_model
from __init__ import plot_learning_curves


def demonstrate():
    X, y = datasets.load_boston(return_X_y=True)
    est = linear_model.Ridge(alpha=0.1)
    cv = model_selection.ShuffleSplit(n_splits=50, test_size=0.2)
    plot_learning_curves(est, X, y, "Ridge, alpha=0.1", cv=cv, train_sizes=np.linspace(.15, 1.0, 10))

if __name__ == '__main__':
    demonstrate()
