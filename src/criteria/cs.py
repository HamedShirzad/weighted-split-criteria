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

    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    grand_total = np.sum(observed)

    return observed, row_totals, col_totals, grand_total

def calculate_expected_frequencies(row_totals, col_totals, grand_total):
    expected = np.outer(row_totals, col_totals) / grand_total
    return expected

def chi_squared_statistic(y_left, y_right):
    observed, row_totals, col_totals, grand_total = create_contingency_table(y_left, y_right)
    expected = calculate_expected_frequencies(row_totals, col_totals, grand_total)

    mask = expected > 0
    expected_safe = np.where(expected == 0, 1e-10, expected)
    chi2_stat = np.sum(((observed - expected)**2 / expected_safe)[mask])

    return chi2_stat

def chi_squared_criterion(y_left, y_right):
    return chi_squared_statistic(y_left, y_right)
