import numpy as np

def calculate_g_value(q):
    if q <= 0 or q >= 1:
        return 0.0
    return np.sqrt(q * (1 - q))

def calculate_g_for_branch(y):
    if len(y) == 0:
        return 0.0
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    total_g = 0.0
    for i, target_class in enumerate(unique_classes):
        n_positive = counts[i]
        q = n_positive / n_samples
        class_g = calculate_g_value(q)
        total_g += class_g
    return total_g / len(unique_classes) if len(unique_classes) > 0 else 0.0

def dkm_criterion(y_left, y_right):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    weight_left = n_left / n_total
    weight_right = n_right / n_total
    g_left = calculate_g_for_branch(y_left)
    g_right = calculate_g_for_branch(y_right)
    return weight_left * g_left + weight_right * g_right

def dkm_binary_criterion(y_left, y_right):
    y_left = np.asarray(y_left, dtype=int)
    y_right = np.asarray(y_right, dtype=int)
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    q_left = np.mean(y_left)
    q_right = np.mean(y_right)
    g_left = calculate_g_value(q_left)
    g_right = calculate_g_value(q_right)
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    weight_left = n_left / n_total
    weight_right = n_right / n_total
    return weight_left * g_left + weight_right * g_right
