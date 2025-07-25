import numpy as np
from .base import BaseCriterion

def gini_impurity(y):
    """
    محاسبه Gini impurity برای آرایه برچسب‌های کلاس
    
    Parameters:
    y : array-like, لیست برچسب‌ها
    
    Returns:
    float : مقدار Gini impurity
    """
    if len(y) == 0:
        return 0.0
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities ** 2)

def gini_split(y_left, y_right):
    """
    محاسبه Gini splitting criterion (امتیاز تقسیم) برای دو شاخه
    
    Parameters:
    y_left, y_right : array-like, لیست برچسب‌ها برای شاخه چپ و راست
    
    Returns:
    float : gini score: impurity با وزن متوسط هر دو شاخه
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    gini_left = gini_impurity(y_left)
    gini_right = gini_impurity(y_right)
    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    return weighted_gini

class GiniCriterion(BaseCriterion):
    """
    کلاس معیار Gini Impurity برای استفاده در تقسیم‌های درخت تصمیم
    """
    def __init__(self):
        super().__init__()
        self.name = "gini"

    def calculate_score(self, y_left, y_right):
        y_left, y_right = self.validate_input(y_left, y_right)
        return gini_split(y_left, y_right)

    def get_description(self):
        return "Gini impurity: معیار ساده و استاندارد برای تقسیم گره‌های درخت تصمیم، مناسب داده‌های چندکلاسه و پرکاربرد در CART."
