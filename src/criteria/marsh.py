import numpy as np

def calculate_g(a, b, c, d, A, B, C, D, N):
    """
    محاسبه کاهش تنوع Gini بر اساس فرمول Marshall
    """
    term1 = (C * D) / N
    term2 = (a * b) / A if A > 0 else 0.0
    term3 = (c * d) / B if B > 0 else 0.0
    return term1 - term2 - term3

def create_contingency(a, b, c, d):
    """
    دریافت فراوانی‌های جدول 2×2 و محاسبه A,B,C,D,N
    """
    A = a + b
    B = c + d
    C = a + c
    D = b + d
    N = A + B
    return A, B, C, D, N

def marsh_criterion(y_left, y_right):
    """
    محاسبه امتیاز Marshall Criterion برای تقسیم باینری

    Parameters:
    y_left : array-like  برچسب‌های شاخه چپ (0 یا 1)
    y_right: array-like  برچسب‌های شاخه راست (0 یا 1)

    Returns:
    float امتیاز G (کاهش تنوع)
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    a = np.sum((y_left == 0))
    b = np.sum((y_left == 1))
    c = np.sum((y_right == 0))
    d = np.sum((y_right == 1))

    A, B, C, D, N = create_contingency(a, b, c, d)
    if N == 0 or A == 0 or B == 0:
        return 0.0

    return calculate_g(a, b, c, d, A, B, C, D, N)
