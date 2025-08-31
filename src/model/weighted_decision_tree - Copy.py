import numpy as np
from custom_tree_classifier import CustomDecisionTreeClassifier
from custom_tree_classifier.metrics.metric_base import MetricBase
from utils.voting_split_manager import FNWeightedSplitManager
import sys
import os

# اضافه کردن پوشه بالاتر (یعنی src) به مسیر جستجوهای پایتون
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import معیارها بدون تغییر
from criteria.gini import gini_criterion
from criteria.qg import gain_ratio_criterion
from criteria.twoing import twoing_criterion
from criteria.ng import normalized_gain_criterion
from criteria.mch import multi_class_hellinger
from criteria.marsh import marsh_criterion
from criteria.gs import g_statistic_criterion
from criteria.dkm import dkm_criterion
from criteria.cs import chi_squared_criterion
from criteria.bhy import bhattacharyya_criterion
from criteria.ks import kolmogorov_smirnov_criterion


class SingleCriterionMetric(MetricBase):
    """
    کلاس ساده برای هر معیار منفرد که سازگار با API متریک کتابخانه است.
    """

    def __init__(self, name, func):
        super().__init__()
        self.name = name
        self.func = func

    def compute_metric(self, metric_data: np.ndarray) -> float:
        y = metric_data[:, 0]
        # این متد صرفاً جهت سازگاری است و معمولاً compute_delta استفاده می‌شود
        return float(self.func(y, np.array([])))

    def compute_delta(self, split: np.ndarray, metric_data: np.ndarray) -> float:
        y_left = metric_data[split]
        y_right = metric_data[~split]
        return float(abs(self.func(y_left, y_right)))


class WeightedVotingMetric(MetricBase):
    """
    متریک ترکیبی با وزن‌دهی داینامیک بر اساس FN معیارها روی داده همان گره جاری
    """

    def __init__(self, criteria, X_data):
        super().__init__()
        self.criteria = criteria  # لیست (نام، تابع)
        self.X_data = X_data      # کل داده ویژگی‌ها
        self.weights_dict = {name: 1.0 for name, _ in self.criteria}

    def estimate_fn_for_criterion(self, name, X_node, y_node):


        if X_node.shape[0] != y_node.shape:
            min_len = min(X_node.shape[0], y_node.shape[0])
           # print(f"Warning: Unequal lengths, trimming to {min_len}")
            X_node = X_node[:min_len]
            y_node = y_node[:min_len]

        metric_obj = None
        for n, f in self.criteria:
            if n == name:
                metric_obj = SingleCriterionMetric(n, f)
                break
        if metric_obj is None:
            return 1.0

        model = CustomDecisionTreeClassifier(max_depth=3, metric=metric_obj)
        metric_data = y_node.reshape(-1, 1)

        model.fit(X_node, y_node, metric_data)

        probas = model.predict_proba(X_node)
        preds = (probas[:, 1] > 0.5).astype(int)  # تبدیل احتمال کلاس مثبت به پیش‌بینی عددی

        fn_count = np.sum((y_node == 1) & (preds == 0))
        return fn_count





    def update_weights_dynamic(self, X_node, y_node):
        weights = {}
        fn_values = []
        for name, _ in self.criteria:
            fn = self.estimate_fn_for_criterion(name, X_node, y_node)
            fn_values.append(fn)
        inv_fn = [1.0/(fn+1e-6) for fn in fn_values]  # معکوس FN
        total = sum(inv_fn)
        for i, (name, _) in enumerate(self.criteria):
            weights[name] = inv_fn[i] / total
        return weights

    def evaluate(self, y_left, y_right, split):
        # تبدیل آرایه بولی split به آرایه اندیس‌های عددی اگر لازم باشد
        if isinstance(split, np.ndarray) and split.dtype == bool and len(split) != len(self.X_data):
            split_indices = np.where(split)[0]
        else:
            split_indices = split  # اگر قبلاً به صورت اندیس عددی است

        # استخراج زیرمجموعه ویژگی‌ها بر اساس اندیس‌ها
        X_node = self.X_data[split_indices]

        # ترکیب y_left و y_right در یک آرایه
        y_node = np.concatenate((y_left, y_right), axis=0)

        # به‌روزرسانی وزن‌ها با داده گره جاری
        self.weights_dict = self.update_weights_dynamic(X_node, y_node)

        scores = []
        for name, func in self.criteria:
            try:
                score = func(y_left, y_right)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        weights = np.array([self.weights_dict.get(name, 0) for name, _ in self.criteria])
        if weights.sum() == 0:
            return np.mean(scores) if scores else 0.0

        weights = weights / weights.sum()
        weighted_score = float(np.dot(scores, weights))

        return weighted_score

    def compute_metric(self, y_left, y_right=None):
        if y_right is None:
            return 0.0
        else:
            # در این سطح split در دسترس نیست، می‌توان فرض کرد همه داده در یک گره است
            return self.evaluate(y_left, y_right, slice(0, len(y_left)+len(y_right)))

    def compute_delta(self, split, metric_data):
        y_left = metric_data[split]
        y_right = metric_data[~split]
        return abs(self.evaluate(y_left, y_right, split))


class CriterionWrapper:
    def __init__(self, name, func):
        self.name = name
        self.func = func
    def calculate_score(self, y_left, y_right):
        return self.func(y_left, y_right)


class WeightedDecisionTreeModel:
    def __init__(self, criteria_funcs_weights=None, positive_label=1, max_depth=5, X_data=None):
        if criteria_funcs_weights is None:
            criteria_funcs_weights = [
                ("gini", gini_criterion, 1.0/11),
                ("gain_ratio", gain_ratio_criterion, 1.0/11),
                ("twoing", twoing_criterion, 1.0/11),
                ("normalized_gain", normalized_gain_criterion, 1.0/11),
                ("multi_class_hellinger", multi_class_hellinger, 1.0/11),
                ("marshall", marsh_criterion, 1.0/11),
                ("g_statistic", g_statistic_criterion, 1.0/11),
                ("dkm", dkm_criterion, 1.0/11),
                ("chi_squared", chi_squared_criterion, 1.0/11),
                ("bhattacharyya", bhattacharyya_criterion, 1.0/11),
                ("kolmogorov_smirnov", kolmogorov_smirnov_criterion, 1.0/11),
            ]
        self.criteria = [(name, func) for name, func, w in criteria_funcs_weights]
        self.weights_dict = {name: w for name, func, w in criteria_funcs_weights}
        self.positive_label = positive_label
        self.max_depth = max_depth

        if X_data is None:
            raise ValueError("X_data باید داده ویژگی‌های کامل را به مدل بدهید.")

        self.X_data = X_data

        self.split_manager = FNWeightedSplitManager(
            criteria_list=[CriterionWrapper(name, func) for name, func in self.criteria],
            positive_label=positive_label,
        )

        self.metric = WeightedVotingMetric(self.criteria, self.X_data)
        self.model = CustomDecisionTreeClassifier(max_depth=self.max_depth, metric=self.metric)

    def fit(self, X, y):
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        metric_data = y.reshape(-1, 1)

        self.model.fit(X, y, metric_data)
'''
    def predict(self, X):
        if hasattr(X, "values"):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values
        return self.model.predict_proba(X)
'''

if __name__ == "__main__":
    # فرض کنید داده آماده است
    # X_train و y_train باید numpy arrays باشند
    X_train = ...
    y_train = ...

    model = WeightedDecisionTreeModel(X_data=X_train)
    model.fit(X_train, y_train)
    print("Model fit successfully.")

