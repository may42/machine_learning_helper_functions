from sklearn.preprocessing import scale, StandardScaler
import numpy as np


class FeaturePreparer:
    """Prepares features:
    1) Adds powers, permutations, logarithms and exponents;
    2) Normalizes all features mean and std;
    """

    def __init__(self, powers):
        self.powers = powers or [1]
        self.scaler = StandardScaler()

    def apply_powers(self, x):
        f_list = [x ** p for p in self.powers if type(p) is int]
        if "perm" in self.powers:
            f_list += [x[:, i:i+1] * x[:, j:j+1] for i in range(x.shape[1]) for j in range(i + 1)]
        if "log" in self.powers:
            f_list += [np.log(x + 1)]
        if "exp" in self.powers:
            f_tmp = scale(x, with_std=True)
            f_list += [np.exp(f_tmp)]
        return np.concatenate(f_list, axis=1) if len(f_list) > 1 else f_list[0]

    def fit(self, x):
        x_new = self.apply_powers(x)
        return self.scaler.fit(x_new)

    def transform(self, x):
        x_new = self.apply_powers(x)
        return self.scaler.transform(x_new)

    def fit_transform(self, x):
        x_new = self.apply_powers(x)
        return self.scaler.fit_transform(x_new)


def give_learning_curves(reg, x_tr, y_tr, x_test, y_test, steps=20):
    train_scores = np.zeros(steps)
    test_scores = np.zeros(steps)
    steps = np.round(np.linspace(6, len(x_tr), steps)).astype(int)
    for i, n in enumerate(steps):
        reg.fit(x_tr[:n], y_tr[:n])
        train_scores[i] = reg.score(x_tr[:n], y_tr[:n])
        test_scores[i] = reg.score(x_test, y_test)
    return train_scores, test_scores

