# Machine learning helper functions
Some useful functions that I use for machine learning: learning curves, feature generation etc.

## Learning curves example:
```
reg = sklearn.linear_model.Ridge(alpha=0.1)
trainScores, testScores = give_learning_curves(reg, x_tr, y_tr, x_test, y_test)
plt.plot(trainScores, 'r', testScores, 'g')
```
Run example.py to see learning curves in action.

## FeaturePreparer example:
```
fp = FeaturePreparer([0.5, 1, "log", "perm", "exp"])
X_tr_scaled = fp.fit_transform(X_tr)
X_test_scaled = fp.transform(X_test)
```
