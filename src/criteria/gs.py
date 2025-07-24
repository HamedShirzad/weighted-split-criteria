import numpy as np
from scipy import stats
from .base import BaseCriterion

def create_contingency_table(y_left, y_right):
    """
    ایجاد جدول contingency برای دو شاخه
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
        
    Returns:
    tuple
        (observed_table, expected_table, classes)
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    # یافتن کلاس‌های موجود
    all_classes = np.unique(np.concatenate([y_left, y_right]))
    k = len(all_classes)
    
    # محاسبه فراوانی‌های مشاهده شده
    observed = np.zeros((2, k))
    for i, class_label in enumerate(all_classes):
        observed[0, i] = np.sum(y_left == class_label)
        observed[1, i] = np.sum(y_right == class_label)
    
    # محاسبه فراوانی‌های مورد انتظار
    n_total = len(y_left) + len(y_right)
    row_totals = [len(y_left), len(y_right)]
    col_totals = [np.sum(observed[:, j]) for j in range(k)]
    
    expected = np.zeros((2, k))
    for i in range(2):
        for j in range(k):
            if n_total > 0:
                expected[i, j] = (row_totals[i] * col_totals[j]) / n_total
            else:
                expected[i, j] = 0.0
    
    return observed, expected, all_classes

def g_statistic(y_left, y_right, alpha=0.05):
    """
    محاسبه G Statistic و بررسی معناداری آماری
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    alpha : float
        سطح معناداری (پیش‌فرض: 0.05)
    
    Returns:
    dict
        شامل G value, معناداری، و p-value
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    # بررسی شرایط اولیه
    if len(y_left) == 0 or len(y_right) == 0:
        return {
            'g_value': 0.0,
            'is_significant': False,
            'p_value': 1.0,
            'degrees_of_freedom': 0
        }
    
    # ایجاد جدول contingency
    observed, expected, all_classes = create_contingency_table(y_left, y_right)
    k = len(all_classes)
    
    # محاسبه G Statistic
    g_value = 0.0
    for i in range(2):
        for j in range(k):
            if observed[i, j] > 0 and expected[i, j] > 0:
                ratio = observed[i, j] / expected[i, j]
                g_value += observed[i, j] * np.log(ratio)
    
    g_value *= 2
    
    # محاسبه درجه آزادی
    degrees_of_freedom = (2 - 1) * (k - 1)
    
    # آزمون معناداری
    if degrees_of_freedom > 0:
        p_value = 1 - stats.chi2.cdf(g_value, degrees_of_freedom)
        critical_value = stats.chi2.ppf(1 - alpha, degrees_of_freedom)
        is_significant = g_value > critical_value
    else:
        p_value = 1.0
        is_significant = False
    
    return {
        'g_value': g_value,
        'is_significant': is_significant,
        'p_value': p_value,
        'degrees_of_freedom': degrees_of_freedom
    }

class GStatisticCriterion(BaseCriterion):
    """
    کلاس معیار G Statistic برای استفاده در درخت‌های تصمیم
    """
    
    def __init__(self, alpha=0.05):
        super().__init__()
        self.name = "g_statistic"
        self.alpha = alpha
    
    def calculate_score(self, y_left, y_right):
        """محاسبه امتیاز G Statistic"""
        y_left, y_right = self.validate_input(y_left, y_right)
        result = g_statistic(y_left, y_right, self.alpha)
        return result['g_value']
    
    def evaluate_split(self, y_left, y_right):
        """
        ارزیابی کامل تقسیم شامل معناداری آماری
        
        Returns:
        dict
            اطلاعات کامل تقسیم
        """
        y_left, y_right = self.validate_input(y_left, y_right)
        return g_statistic(y_left, y_right, self.alpha)
    
    def get_description(self):
        """توضیح معیار"""
        return f"G Statistic: کنترل overfitting با معناداری آماری (α={self.alpha})"
    
    def set_alpha(self, alpha):
        """تنظیم سطح معناداری"""
        if 0 < alpha < 1:
            self.alpha = alpha
        else:
            raise ValueError("Alpha باید بین 0 و 1 باشد")
