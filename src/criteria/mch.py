import numpy as np

def multi_class_hellinger(y_left, y_right):
    """
    محاسبه امتیاز معیار Multi-Class Hellinger
    
    Parameters:
    y_left : array-like
        برچسب‌های کلاس سمت چپ
    y_right : array-like
        برچسب‌های کلاس سمت راست
    
    Returns:
    float
        امتیاز MCH (بین 0 و 1)
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        return 0.0

    all_classes = np.unique(np.concatenate([y_left, y_right]))
    hellinger_sum = 0.0
    
    for class_j in all_classes:
        p_Lj = np.sum(y_left == class_j) / n_left
        p_Rj = np.sum(y_right == class_j) / n_right
        
        hellinger_sum += (np.sqrt(p_Lj) - np.sqrt(p_Rj)) ** 2
    
    return 0.5 * hellinger_sum
