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
        (observed_table, row_totals, col_totals, grand_total)
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
    
    # محاسبه مجموع ردیف‌ها و ستون‌ها
    row_totals = np.sum(observed, axis=1)
    col_totals = np.sum(observed, axis=0)
    grand_total = np.sum(observed)
    
    return observed, row_totals, col_totals, grand_total, all_classes

def calculate_expected_frequencies(row_totals, col_totals, grand_total):
    """
    محاسبه فراوانی‌های مورد انتظار
    
    Parameters:
    row_totals : array
        مجموع هر ردیف
    col_totals : array
        مجموع هر ستون
    grand_total : float
        مجموع کل
        
    Returns:
    ndarray
        ماتریس فراوانی‌های مورد انتظار
    """
    expected = np.outer(row_totals, col_totals) / grand_total
    return expected

def chi_squared_test(y_left, y_right, alpha=0.05):
    """
    اجرای کامل آزمون Chi-Squared
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    alpha : float
        سطح معناداری
        
    Returns:
    dict
        نتایج کامل آزمون
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    # بررسی شرایط اولیه
    if len(y_left) == 0 or len(y_right) == 0:
        return {
            'chi2_statistic': 0.0,
            'p_value': 1.0,
            'degrees_of_freedom': 0,
            'is_significant': False,
            'expected_freq_valid': False
        }
    
    # ایجاد جدول contingency
    observed, row_totals, col_totals, grand_total, classes = create_contingency_table(y_left, y_right)
    
    # محاسبه فراوانی‌های مورد انتظار
    expected = calculate_expected_frequencies(row_totals, col_totals, grand_total)
    
    # بررسی شرط E_ij >= 5
    expected_freq_valid = np.all(expected >= 5)
    
    # محاسبه Chi-Squared statistic
    # جلوگیری از تقسیم بر صفر
    mask = expected > 0
    chi2_stat = np.sum(((observed - expected) ** 2 / expected)[mask])
    
    # درجه آزادی
    r, c = observed.shape
    df = (r - 1) * (c - 1)
    
    # محاسبه p-value
    if df > 0:
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        is_significant = p_value < alpha
    else:
        p_value = 1.0
        is_significant = False
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': df,
        'is_significant': is_significant,
        'expected_freq_valid': expected_freq_valid,
        'observed': observed,
        'expected': expected
    }

def chi_squared_criterion(y_left, y_right):
    """
    محاسبه امتیاز Chi-Squared برای تقسیم
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
        
    Returns:
    float
        مقدار Chi-Squared statistic
    """
    result = chi_squared_test(y_left, y_right)
    return result['chi2_statistic']

class ChiSquaredCriterion(BaseCriterion):
    """
    کلاس معیار Chi-Squared برای استفاده در درخت‌های تصمیم
    """
    
    def __init__(self, alpha=0.05):
        super().__init__()
        self.name = "chi_squared"
        self.alpha = alpha
    
    def calculate_score(self, y_left, y_right):
        """محاسبه امتیاز Chi-Squared"""
        y_left, y_right = self.validate_input(y_left, y_right)
        return chi_squared_criterion(y_left, y_right)
    
    def evaluate_split(self, y_left, y_right):
        """
        ارزیابی کامل تقسیم شامل معناداری آماری
        
        Returns:
        dict
            اطلاعات کامل آزمون Chi-Squared
        """
        y_left, y_right = self.validate_input(y_left, y_right)
        return chi_squared_test(y_left, y_right, self.alpha)
    
    def get_description(self):
        """توضیح معیار"""
        return f"Chi-Squared: classical statistical test for independence (α={self.alpha})"
    
    def set_alpha(self, alpha):
        """تنظیم سطح معناداری"""
        if 0 < alpha < 1:
            self.alpha = alpha
        else:
            raise ValueError("Alpha باید بین 0 و 1 باشد")
