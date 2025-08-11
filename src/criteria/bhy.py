import numpy as np
from .base import BaseCriterion

def bhattacharyya_coefficient(y_left, y_right, epsilon=1e-10):
    """
    محاسبه ضریب Bhattacharyya بین دو توزیع
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    epsilon : float
        مقدار smoothing برای جلوگیری از احتمالات صفر
        
    Returns:
    float
        ضریب Bhattacharyya (بین 0 و 1)
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    # بررسی شرایط اولیه
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    
    # یافتن کلاس‌های موجود
    all_classes = np.unique(np.concatenate([y_left, y_right]))
    k = len(all_classes)
    
    # شمارش فراوانی کلاس‌ها
    counts_left = np.zeros(k)
    counts_right = np.zeros(k)
    
    for i, class_label in enumerate(all_classes):
        counts_left[i] = np.sum(y_left == class_label)
        counts_right[i] = np.sum(y_right == class_label)
    
    n_left = len(y_left)
    n_right = len(y_right)
    
    # اعمال Laplace smoothing برای جلوگیری از احتمالات صفر
    p_smooth = (counts_left + epsilon) / (n_left + k * epsilon)
    q_smooth = (counts_right + epsilon) / (n_right + k * epsilon)
    
    # محاسبه Bhattacharyya Coefficient
    bc = np.sum(np.sqrt(p_smooth * q_smooth))
    
    return bc

def bhattacharyya_distance(y_left, y_right, epsilon=1e-10):
    """
    محاسبه فاصله Bhattacharyya بین دو توزیع
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    epsilon : float
        مقدار smoothing
        
    Returns:
    float
        فاصله Bhattacharyya (0 تا +inf)
    """
    bc = bhattacharyya_coefficient(y_left, y_right, epsilon)
    
    # محاسبه Bhattacharyya Distance
    if bc > 0:
        bd = -np.log(bc)
    else:
        bd = float('inf')  # حداکثر فاصله برای توزیع‌های کاملاً متفاوت
    
    return bd

def robust_bhattacharyya_distance(y_left, y_right, min_samples=5, epsilon=1e-10):
    """
    نسخه robust از Bhattacharyya Distance با کنترل حداقل نمونه
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    min_samples : int
        حداقل تعداد نمونه مورد نیاز در هر شاخه
    epsilon : float
        مقدار smoothing
        
    Returns:
    float
        فاصله Bhattacharyya یا 0 در صورت عدم اعتبار
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    if len(y_left) < min_samples or len(y_right) < min_samples:
        return 0.0  # تقسیم نامعتبر
    
    return bhattacharyya_distance(y_left, y_right, epsilon)

class BhattacharyyaCriterion(BaseCriterion):
    """
    کلاس معیار Bhattacharyya Distance برای استفاده در درخت‌های تصمیم
    
    این معیار بر اساس فاصله بین توزیع‌های احتمال کلاس‌ها عمل می‌کند
    و upper bound برای خطای طبقه‌بندی ارائه می‌دهد.
    """
    
    def __init__(self, epsilon=1e-10, min_samples=5, robust=True):
        super().__init__()
        self.name = "bhattacharyya"
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.robust = robust
    
    def calculate_score(self, y_left, y_right):
        """محاسبه امتیاز Bhattacharyya Distance"""
        y_left, y_right = self.validate_input(y_left, y_right)
        
        if self.robust:
            return robust_bhattacharyya_distance(
                y_left, y_right, self.min_samples, self.epsilon
            )
        else:
            return bhattacharyya_distance(y_left, y_right, self.epsilon)
    
    def calculate_coefficient(self, y_left, y_right):
        """
        محاسبه ضریب Bhattacharyya (برای تحلیل تشابه)
        
        Returns:
        float
            ضریب تشابه (1 = یکسان، 0 = کاملاً متفاوت)
        """
        y_left, y_right = self.validate_input(y_left, y_right)
        return bhattacharyya_coefficient(y_left, y_right, self.epsilon)
    
    def evaluate_split(self, y_left, y_right):
        """
        ارزیابی کامل تقسیم شامل فاصله و ضریب
        
        Returns:
        dict
            اطلاعات کامل تقسیم
        """
        y_left, y_right = self.validate_input(y_left, y_right)
        
        bd = self.calculate_score(y_left, y_right)
        bc = self.calculate_coefficient(y_left, y_right)
        
        # تخمین upper bound برای خطای طبقه‌بندی
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        
        if n_total > 0:
            p1 = n_left / n_total
            p2 = n_right / n_total
            error_bound = np.sqrt(p1 * p2) * np.exp(-bd) if bd != float('inf') else 0.0
        else:
            error_bound = 0.5
        
        return {
            'bhattacharyya_distance': bd,
            'bhattacharyya_coefficient': bc,
            'similarity': bc,  # alias برای تفسیر آسان‌تر
            'error_upper_bound': error_bound,
            'is_valid': bd != float('inf')
        }
    
    def get_description(self):
        """توضیح معیار"""
        mode = "robust" if self.robust else "standard"
        return f"Bhattacharyya Distance: probability distance with error bounds ({mode})"
    
    def set_epsilon(self, epsilon):
        """تنظیم مقدار smoothing"""
        if epsilon > 0:
            self.epsilon = epsilon
        else:
            raise ValueError("Epsilon باید مثبت باشد")
    
    def set_min_samples(self, min_samples):
        """تنظیم حداقل نمونه برای حالت robust"""
        if min_samples >= 1:
            self.min_samples = min_samples
        else:
            raise ValueError("حداقل نمونه باید حداقل 1 باشد")
    
    def set_robust_mode(self, robust):
        """تنظیم حالت robust"""
        self.robust = robust
