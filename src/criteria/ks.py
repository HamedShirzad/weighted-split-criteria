import numpy as np
from scipy import stats

def empirical_cdf(data):
    if len(data) == 0:
        return np.array([]), np.array([])
    
    data = np.sort(np.asarray(data))
    n = len(data)
    values, counts = np.unique(data, return_counts=True)
    cum_counts = np.cumsum(counts)
    cdf_values = cum_counts / n
    
    return values, cdf_values

def ks_distance(y_left, y_right):
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)
    
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0
    
    all_values = np.union1d(y_left, y_right)
    x_left, cdf_left = empirical_cdf(y_left)
    x_right, cdf_right = empirical_cdf(y_right)
    
    cdf_left_interp = np.interp(all_values, x_left, cdf_left, left=0, right=1)
    cdf_right_interp = np.interp(all_values, x_right, cdf_right, left=0, right=1)
    
    max_distance = np.max(np.abs(cdf_left_interp - cdf_right_interp))
    return max_distance

def kolmogorov_smirnov_criterion(y_left, y_right):
    return ks_distance(y_left, y_right)
