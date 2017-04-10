# Machine learning helper functions
Some useful functions that I use for machine learning: learning curves, feature generation etc.

## Learning curves example:
```
est = sklearn.linear_model.Ridge(alpha=0.1)
cv = model_selection.ShuffleSplit(n_splits=10, test_size=0.2)
plot_learning_curves(est, X, y, "Ridge, alpha=0.1", cv=cv).show()
```
Run example.py to see learning curves in action.

## FeaturePreparer example:
```
fp = FeaturePreparer([0.5, 1, "log", "perm", "exp"])
X_tr_scaled = fp.fit_transform(X_tr)
X_test_scaled = fp.transform(X_test)
```
