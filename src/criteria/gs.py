import numpy as np
from scipy import stats

def create_contingency_table(y_left, y_right):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    all_classes = np.unique(np.concatenate([y_left, y_right]))
    k = len(all_classes)
    
    observed = np.zeros((2, k))
    for i, class_label in enumerate(all_classes):
        observed[0, i] = np.sum(y_left == class_label)
        observed[1, i] = np.sum(y_right == class_label)
    
    n_total = len(y_left) + len(y_right)
    row_totals = [len(y_left), len(y_right)]
    col_totals = [np.sum(observed[:, j]) for j in range(k)]
    
    expected = np.zeros((2, k))
    for i in range(2):
        for j in range(k):
            expected[i, j] = (row_totals[i] * col_totals[j]) / n_total if n_total > 0 else 0.0
            
    return observed, expected, all_classes

def g_statistic(y_left, y_right, alpha=0.05):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    
    observed, expected, all_classes = create_contingency_table(y_left, y_right)
    k = len(all_classes)
    
    g_value = 0.0
    for i in range(2):
        for j in range(k):
            if observed[i, j] > 0 and expected[i, j] > 0:
                ratio = observed[i, j] / expected[i, j]
                g_value += observed[i, j] * np.log(ratio)
    g_value *= 2
    
    # بازگشت فقط مقدار g_value برای معیار
    return g_value

# تابع معیار نهایی برای custom-tree-classifier
def g_statistic_criterion(y_left, y_right):
    return g_statistic(y_left, y_right)
