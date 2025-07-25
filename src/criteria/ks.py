import numpy as np
from scipy import stats
from .base import BaseCriterion

def empirical_cdf(data):
    """
    محاسبه تابع توزیع تجمعی تجربی برای یک آرایه عددی
    
    Parameters:
    data : array-like
        داده‌های عددی
        
    Returns:
    tuple
        (values, cdf_values) - مقادیر منحصر به فرد و احتمالات تجمعی آنها
    """
    if len(data) == 0:
        return np.array([]), np.array([])
    
    data = np.sort(np.asarray(data))
    n = len(data)
    values, counts = np.unique(data, return_counts=True)
    cum_counts = np.cumsum(counts)
    cdf_values = cum_counts / n
    
    return values, cdf_values

def ks_distance(y_left, y_right):
    """
    محاسبه فاصله Kolmogorov-Smirnov بین دو توزیع
    
    Parameters:
    y_left : array-like
        نمونه‌های شاخه چپ
    y_right : array-like
        نمونه‌های شاخه راست
        
    Returns:
    float
        فاصله K-S (بین 0 و 1)
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    # بررسی شرایط اولیه
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    
    # ترکیب تمام مقادیر منحصر به فرد برای نقاط ارزیابی
    all_values = np.union1d(y_left, y_right)
    
    # محاسبه توزیع تجمعی تجربی برای هر گروه
    x_left, cdf_left = empirical_cdf(y_left)
    x_right, cdf_right = empirical_cdf(y_right)
    
    # محاسبه مقادیر CDF در تمام نقاط ارزیابی
    cdf_left_interp = np.interp(all_values, x_left, cdf_left, left=0, right=1)
    cdf_right_interp = np.interp(all_values, x_right, cdf_right, left=0, right=1)
    
    # محاسبه حداکثر فاصله مطلق
    max_distance = np.max(np.abs(cdf_left_interp - cdf_right_interp))
    
    return max_distance

def ks_test_pvalue(y_left, y_right):
    """
    محاسبه p-value آزمون Kolmogorov-Smirnov دو نمونه‌ای
    
    Parameters:
    y_left : array-like
        نمونه‌های شاخه چپ
    y_right : array-like
        نمونه‌های شاخه راست
        
    Returns:
    tuple
        (ks_statistic, p_value)
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0, 1.0
    
    try:
        # استفاده از scipy برای محاسبه دقیق آزمون K-S
        ks_stat, p_val = stats.ks_2samp(y_left, y_right)
        return ks_stat, p_val
    except:
        # fallback به محاسبه دستی
        return ks_distance(y_left, y_right), 0.5

class KolmogorovSmirnovCriterion(BaseCriterion):
    """
    کلاس معیار Kolmogorov-Smirnov برای استفاده در درخت‌های تصمیم
    
    این معیار بر اساس حداکثر فاصله بین توابع توزیع تجمعی تجربی
    دو کلاس عمل می‌کند و نیازی به فرضیات درباره توزیع داده‌ها ندارد.
    """
    
    def __init__(self, alpha=0.05):
        super().__init__()
        self.name = "kolmogorov_smirnov"
        self.alpha = alpha
    
    def calculate_score(self, y_left, y_right):
        """محاسبه امتیاز K-S distance"""
        y_left, y_right = self.validate_input(y_left, y_right)
        return ks_distance(y_left, y_right)
    
    def evaluate_split(self, y_left, y_right):
        """
        ارزیابی کامل تقسیم شامل معناداری آماری
        
        Returns:
        dict
            اطلاعات کامل تقسیم شامل K-S distance و p-value
        """
        y_left, y_right = self.validate_input(y_left, y_right)
        
        ks_stat, p_val = ks_test_pvalue(y_left, y_right)
        is_significant = p_val < self.alpha
        
        return {
            'ks_distance': ks_stat,
            'p_value': p_val,
            'is_significant': is_significant,
            'alpha': self.alpha
        }
    
    def get_description(self):
        """توضیح معیار"""
        return f"Kolmogorov-Smirnov: nonparametric criterion based on max CDF distance (α={self.alpha})"
    
    def set_alpha(self, alpha):
        """تنظیم سطح معناداری"""
        if 0 < alpha < 1:
            self.alpha = alpha
        else:
            raise ValueError("Alpha باید بین 0 و 1 باشد")
