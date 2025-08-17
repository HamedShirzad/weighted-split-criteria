import numpy as np

def false_negative(y_true, y_pred, positive_label=1):
    """
    شمارش تعداد False Negativeها:
    - y_true: آرایه برچسب‌های واقعی (1 بعدی یا هر ساختار numpy)
    - y_pred: آرایه برچسب‌های پیش‌بینی‌شده با اندازه مشابه y_true
    - positive_label: مقدار کلاس مثبت (معمولاً 1 در طبقه‌بندی دودویی)
    
    خروجی: int (تعداد FN)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # FN: نمونه‌هایی که باید مثبت پیش‌بینی می‌شدند ولی نشده‌اند
    return np.sum((y_true == positive_label) & (y_pred != positive_label))
