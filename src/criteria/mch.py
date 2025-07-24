import numpy as np
from .base import BaseCriterion

def multi_class_hellinger(y_left, y_right):
    """
    محاسبه امتیاز معیار Multi-Class Hellinger
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    
    Returns:
    float
        امتیاز MCH (بین 0 و 1)
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        return 0.0

    # یافتن کلاس‌های موجود
    all_classes = np.unique(np.concatenate([y_left, y_right]))
    hellinger_sum = 0.0
    
    for class_j in all_classes:
        # احتمال کلاس در هر شاخه
        p_Lj = np.sum(y_left == class_j) / n_left
        p_Rj = np.sum(y_right == class_j) / n_right
        
        # مجذور تفاضل جذرها
        hellinger_sum += (np.sqrt(p_Lj) - np.sqrt(p_Rj)) ** 2
    
    return 0.5 * hellinger_sum

class MultiClassHellingerCriterion(BaseCriterion):
    def __init__(self):
        super().__init__()
        self.name = "mch"
    
    def calculate_score(self, y_left, y_right):
        y_left, y_right = self.validate_input(y_left, y_right)
        return multi_class_hellinger(y_left, y_right)
    
    def get_description(self):
        return "Multi-Class Hellinger: مقاوم در برابر عدم تعادل کلاس‌ها"
