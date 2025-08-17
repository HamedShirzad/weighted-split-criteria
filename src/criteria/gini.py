import numpy as np

def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities ** 2)

def gini_criterion(y_left, y_right):
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    gini_left = gini_impurity(y_left)
    gini_right = gini_impurity(y_right)
    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    return weighted_gini
