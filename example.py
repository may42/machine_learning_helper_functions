from __init__ import give_learning_curves
from sklearn import datasets, model_selection, linear_model
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def demonstrate():
    x_orig, y_orig = datasets.load_boston(return_X_y=True)
    x_tr, x_test, y_tr, y_test = model_selection.train_test_split(x_orig, y_orig, test_size=0.2)

    reg = linear_model.Ridge(alpha=0.1)
    trainScores, testScores = give_learning_curves(reg, x_tr, y_tr, x_test, y_test)

    plt.plot(trainScores, 'r', testScores, 'g')
    plt.gca().set_ylim([1, 0.5])
    plt.title('Learning curves')
    plt.show()

if __name__ == '__main__':
    demonstrate()
