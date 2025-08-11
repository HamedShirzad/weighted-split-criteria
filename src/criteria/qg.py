import numpy as np

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))

def information_gain(y_parent, branches):
    h_parent = entropy(y_parent)
    n_total = len(y_parent)
    children_entropy = 0
    for y_child in branches:
        n_child = len(y_child)
        if n_child == 0:
            continue
        children_entropy += n_child / n_total * entropy(y_child)
    return h_parent - children_entropy

def intrinsic_value(branches):
    n_total = sum(len(y_child) for y_child in branches)
    iv = 0.0
    for y_child in branches:
        p = len(y_child) / n_total if n_total > 0 else 0.
        if p > 0:
            iv -= p * np.log2(p)
    return iv

def gain_ratio(y_parent, branches):
    ig = information_gain(y_parent, branches)
    iv = intrinsic_value(branches)
    if iv == 0:
        return 0.
    return ig / iv

class GainRatioCriterion:
    def calculate_score(self, y_left, y_right):
        branches = [y_left, y_right]  # هر کدام آرایه‌ای از لیبل‌هاست
        y_parent = np.concatenate(branches)
        return gain_ratio(y_parent, branches)
