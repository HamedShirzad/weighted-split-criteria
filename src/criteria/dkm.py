import numpy as np
from .base import BaseCriterion

def calculate_g_value(q):
    """
    محاسبه G(q) = sqrt(q * (1-q)) برای نسبت q
    
    Parameters:
    q : float
        نسبت کلاس مثبت (بین 0 و 1)
        
    Returns:
    float
        مقدار G(q)
    """
    if q <= 0 or q >= 1:
        return 0.0
    
    return np.sqrt(q * (1 - q))

def dkm_criterion(y_left, y_right):
    """
    محاسبه معیار Dietterich-Kearns-Mansour (G-criterion)
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    
    Returns:
    float
        امتیاز DKM وزنی
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    # بررسی شرایط اولیه
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    # محاسبه وزن‌های شاخه‌ها
    weight_left = n_left / n_total
    weight_right = n_right / n_total
    
    # محاسبه G برای هر شاخه
    g_left = calculate_g_for_branch(y_left)
    g_right = calculate_g_for_branch(y_right)
    
    # ترکیب وزنی
    dkm_score = weight_left * g_left + weight_right * g_right
    
    return dkm_score

def calculate_g_for_branch(y):
    """
    محاسبه G برای یک شاخه (چندکلاسه)
    
    Parameters:
    y : array-like
        برچسب‌های کلاس
        
    Returns:
    float
        مقدار G برای این شاخه
    """
    if len(y) == 0:
        return 0.0
    
    # یافتن کلاس‌های موجود
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    
    # روش One-vs-Rest برای چندکلاسه
    total_g = 0.0
    
    for i, target_class in enumerate(unique_classes):
        # تبدیل به مسئله باینری
        n_positive = counts[i]  # تعداد نمونه‌های کلاس هدف
        n_negative = n_samples - n_positive  # تعداد سایر کلاس‌ها
        
        # محاسبه نسبت کلاس مثبت
        q = n_positive / n_samples
        
        # محاسبه G(q) برای این کلاس
        class_g = calculate_g_value(q)
        total_g += class_g
    
    # میانگین G برای همه کلاس‌ها
    return total_g / len(unique_classes) if len(unique_classes) > 0 else 0.0

def dkm_binary_criterion(y_left, y_right):
    """
    نسخه باینری محض معیار DKM (مطابق مقاله اصلی)
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ (0 یا 1)
    y_right : array-like
        برچسب‌های کلاس سمت راست (0 یا 1)
    
    Returns:
    float
        امتیاز DKM باینری
    """
    y_left = np.asarray(y_left, dtype=int)
    y_right = np.asarray(y_right, dtype=int)
    
    # بررسی شرایط اولیه
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    
    # محاسبه نسبت کلاس مثبت در هر شاخه
    q_left = np.mean(y_left)  # نسبت کلاس 1 در شاخه چپ
    q_right = np.mean(y_right)  # نسبت کلاس 1 در شاخه راست
    
    # محاسبه G(q) برای هر شاخه
    g_left = calculate_g_value(q_left)
    g_right = calculate_g_value(q_right)
    
    # محاسبه وزن‌های شاخه‌ها
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    
    weight_left = n_left / n_total
    weight_right = n_right / n_total
    
    # ترکیب وزنی
    dkm_score = weight_left * g_left + weight_right * g_right
    
    return dkm_score

class DKMCriterion(BaseCriterion):
    """
    کلاس معیار Dietterich-Kearns-Mansour برای استفاده در درخت‌های تصمیم
    """
    
    def __init__(self, binary_mode=False):
        super().__init__()
        self.name = "dkm"
        self.binary_mode = binary_mode
    
    def calculate_score(self, y_left, y_right):
        """محاسبه امتیاز DKM"""
        y_left, y_right = self.validate_input(y_left, y_right)
        
        if self.binary_mode:
            return dkm_binary_criterion(y_left, y_right)
        else:
            return dkm_criterion(y_left, y_right)
    
    def get_description(self):
        """توضیح معیار"""
        mode_str = "باینری" if self.binary_mode else "چندکلاسه"
        return f"DKM Criterion: بهبود یافته Information Gain با concavity بهتر ({mode_str})"
    
    def set_binary_mode(self, binary_mode):
        """تنظیم حالت باینری یا چندکلاسه"""
        self.binary_mode = binary_mode
