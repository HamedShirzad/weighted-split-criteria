"""
کلاس پایه برای همه معیارهای تقسیم در درخت‌های تصمیم

این کلاس به عنوان الگوی مشترک برای هر معیار تقسیم مثل Twoing یا Gini استفاده می‌شود.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseCriterion(ABC):
    """
    کلاس پایه انتزاعی برای پیاده‌سازی معیارهای مختلف تقسیم گره
    """
    def __init__(self):
        self.name = None

    @abstractmethod
    def calculate_score(self, y_left, y_right):
        """
        محاسبه امتیاز معیار برای تقسیم یک گره
        
        Parameters:
        -----------
        y_left : array-like
            برچسب‌های نمونه‌های شاخه چپ
        y_right : array-like
            برچسب‌های نمونه‌های شاخه راست

        Returns:
        --------
        float
            امتیاز تقسیم بر اساس معیار مشخص
        """
        pass

    def validate_input(self, y_left, y_right):
        """
        اعتبارسنجی داده‌های ورودی برای تقسیم گره

        - تبدیل لیست‌ها به numpy array
        - کنترل اینکه هر دو شاخه به طور همزمان خالی نباشند

        Parameters:
        -----------
        y_left : array-like
            برچسب‌های نمونه‌های شاخه چپ
        y_right : array-like
            برچسب‌های نمونه‌های شاخه راست

        Returns:
        --------
        tuple of np.ndarray
            آرایه‌های شاخه چپ و راست
        """
        y_left = np.asarray(y_left)
        y_right = np.asarray(y_right)

        if len(y_left) == 0 and len(y_right) == 0:
            raise ValueError("هم شاخه چپ و هم شاخه راست نمی‌توانند همزمان خالی باشند.")
        return y_left, y_right
