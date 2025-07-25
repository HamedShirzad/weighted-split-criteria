import numpy as np
from .base import BaseCriterion

def entropy(y):
    """
    محاسبه Entropy برای یک آرایه برچسب‌های کلاس
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    nonzero_probs = probabilities[probabilities > 0]
    return -np.sum(nonzero_probs * np.log2(nonzero_probs))

def information_gain(y_parent, branches):
    """
    محاسبه Information Gain برای تقسیم parent به چند شاخه
    """
    n_total = len(y_parent)
    if n_total == 0:
        return 0.0
    parent_entropy = entropy(y_parent)
    weighted_child_entropy = 0.0
    for y_child in branches:
        n_child = len(y_child)
        if n_child > 0:
            weighted_child_entropy += (n_child / n_total) * entropy(y_child)
    return parent_entropy - weighted_child_entropy

def intrinsic_value(branches):
    """
    محاسبه Intrinsic Value یا Split Info برای تقسیم به چند شاخه
    """
    n_total = sum(len(branch) for branch in branches)
    if n_total == 0:
        return 0.0
    values = [len(branch) / n_total for branch in branches if len(branch) > 0]
    if len(values) == 0:
        return 0.0
    return -np.sum([p * np.log2(p) for p in values if p > 0])

def gain_ratio(y_parent, branches):
    """
    محاسبه معیار Gain Ratio (Quinlan Gain)
    """
    ig = information_gain(y_parent, branches)
    iv = intrinsic_value(branches)
    if iv == 0.0:
        return 0.0   # تقسیمات با فقط یک شاخه پر: نادیده گرفته می‌شوند
    return ig / iv

class GainRatioCriterion(BaseCriterion):
    """
    کلاس معیار Gain Ratio (Quinlan) برای استفاده در درخت‌های تصمیم چندشاخه‌ای
    """
    def __init__(self):
        super().__init__()
        self.name = "gain_ratio"

    def calculate_score(self, y_parent, branches):
        # branches: لیست [y_left, y_right, ...] در تقسیم چندشاخه‌ای
        return gain_ratio(y_parent, branches)

    def get_description(self):
        return "Gain Ratio (Quinlan): معیار بهبودیافته Information Gain با تقسیم بر Intrinsic Value برای حذف تعصب ویژگی‌های چندمقداره."
