import numpy as np

def calculate_entropy(y):
    if len(y) == 0:
        return 0.0
    
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    
    entropy = 0.0
    for count in counts:
        if count > 0:
            probability = count / n_samples
            entropy -= probability * np.log2(probability)
    return entropy

def normalized_gain(y_left, y_right, n_branches=2):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    y_parent = np.concatenate([y_left, y_right])
    n_total = len(y_parent)
    
    if n_total == 0 or n_branches <= 1:
        return 0.0
    
    n_left = len(y_left)
    n_right = len(y_right)
    
    if n_left == 0 or n_right == 0:
        return 0.0
    
    parent_entropy = calculate_entropy(y_parent)
    left_entropy = calculate_entropy(y_left)
    right_entropy = calculate_entropy(y_right)
    
    weighted_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy
    
    information_gain = parent_entropy - weighted_entropy
    
    if information_gain <= 0:
        return 0.0
    
    normalized_gain_value = information_gain / np.log2(n_branches)
    
    return normalized_gain_value

def normalized_gain_criterion(y_left, y_right):
    return normalized_gain(y_left, y_right, n_branches=2)
