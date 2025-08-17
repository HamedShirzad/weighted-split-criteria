import numpy as np

def twoing_criterion(y_left, y_right):
    """
    محاسبه امتیاز معیار Twoing برای تقسیم داده‌ها
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    
    Returns:
    float
        امتیاز Twoing (بین 0 و 0.5)
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_left == 0 or n_right == 0 or n_total == 0:
        return 0.0

    p_L = n_left / n_total
    p_R = n_right / n_total
    all_classes = np.unique(np.concatenate([y_left, y_right]))
    diff_sum = 0.0
    for c in all_classes:
        p_Lj = np.sum(y_left == c) / n_left
        p_Rj = np.sum(y_right == c) / n_right
        diff_sum += abs(p_Lj - p_Rj)
    return (p_L * p_R / 4.0) * (diff_sum ** 2)
