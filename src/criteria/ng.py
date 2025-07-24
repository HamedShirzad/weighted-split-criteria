import numpy as np
from .base import BaseCriterion

def calculate_entropy(y):
    """
    محاسبه آنتروپی یک مجموعه داده
    
    Parameters:
    y : array-like
        برچسب‌های کلاس
        
    Returns:
    float
        مقدار آنتروپی
    """
    if len(y) == 0:
        return 0.0
    
    # شمارش فراوانی کلاس‌ها
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    
    # محاسبه آنتروپی
    entropy = 0.0
    for count in counts:
        if count > 0:
            probability = count / n_samples
            entropy -= probability * np.log2(probability)
    
    return entropy

def normalized_gain(y_left, y_right, n_branches=2):
    """
    محاسبه امتیاز معیار Normalized Gain
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    n_branches : int
        تعداد شاخه‌های تقسیم (پیش‌فرض: 2)
    
    Returns:
    float
        امتیاز Normalized Gain
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    # ترکیب داده‌های والد
    y_parent = np.concatenate([y_left, y_right])
    n_total = len(y_parent)
    
    # بررسی شرایط خاص
    if n_total == 0 or n_branches <= 1:
        return 0.0
    
    n_left = len(y_left)
    n_right = len(y_right)
    
    # اگر یکی از شاخه‌ها خالی باشد
    if n_left == 0 or n_right == 0:
        return 0.0
    
    # محاسبه آنتروپی والد
    parent_entropy = calculate_entropy(y_parent)
    
    # محاسبه آنتروپی وزن‌دار زیرمجموعه‌ها
    left_entropy = calculate_entropy(y_left)
    right_entropy = calculate_entropy(y_right)
    
    weighted_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy
    
    # محاسبه Information Gain استاندارد
    information_gain = parent_entropy - weighted_entropy
    
    # نرمال‌سازی با log_2(n)
    if information_gain <= 0:
        return 0.0
    
    normalized_gain_value = information_gain / np.log2(n_branches)
    
    return normalized_gain_value

class NormalizedGainCriterion(BaseCriterion):
    """
    کلاس معیار Normalized Gain برای استفاده در درخت‌های تصمیم
    """
    
    def __init__(self):
        super().__init__()
        self.name = "normalized_gain"
    
    def calculate_score(self, y_left, y_right):
        """محاسبه امتیاز Normalized Gain"""
        y_left, y_right = self.validate_input(y_left, y_right)
        # فرض تقسیم باینری (n=2)
        return normalized_gain(y_left, y_right, n_branches=2)
    
    def get_description(self):
        """توضیح معیار"""
        return "Normalized Gain: رفع تعصب نسبت به ویژگی‌های چندمقداره"
