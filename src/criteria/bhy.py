import numpy as np

def bhattacharyya_coefficient(y_left, y_right, epsilon=1e-10):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0

    all_classes = np.unique(np.concatenate([y_left, y_right]))
    k = len(all_classes)

    counts_left = np.zeros(k)
    counts_right = np.zeros(k)

    for i, class_label in enumerate(all_classes):
        counts_left[i] = np.sum(y_left == class_label)
        counts_right[i] = np.sum(y_right == class_label)

    n_left = len(y_left)
    n_right = len(y_right)

    p_smooth = (counts_left + epsilon) / (n_left + k * epsilon)
    q_smooth = (counts_right + epsilon) / (n_right + k * epsilon)

    bc = np.sum(np.sqrt(p_smooth * q_smooth))

    return bc

def bhattacharyya_distance(y_left, y_right, epsilon=1e-10):
    bc = bhattacharyya_coefficient(y_left, y_right, epsilon)
    if bc > 0:
        bd = -np.log(bc)
    else:
        bd = float('inf')
    return bd

def bhattacharyya_criterion(y_left, y_right):
    return bhattacharyya_distance(y_left, y_right)
