import numpy as np
from custom_tree_classifier import CustomDecisionTreeClassifier
from utils.voting_split_manager import FNWeightedSplitManager

# Import تمام معیارهای تقسیم به صورت توابع معیار آماده
from criteria.gini import gini_criterion
from criteria.gain_ratio import gain_ratio_criterion
from criteria.twoing import twoing_criterion
from criteria.normalized_gain import normalized_gain_criterion
from criteria.multi_class_hellinger import multi_class_hellinger
from criteria.marshall import marsh_criterion
from criteria.g_statistic import g_statistic_criterion
from criteria.dkm import dkm_criterion
from criteria.chi_squared import chi_squared_criterion
from criteria.bhattacharyya import bhattacharyya_criterion
from criteria.kolmogorov_smirnov import kolmogorov_smirnov_criterion



class WeightedDecisionTreeModel:
    """
    مدل درخت تصمیم با استفاده از رای‌گیری وزنی بین چندین معیار تقسیم
    """

    def __init__(self, criteria_funcs_weights, positive_label=1):
        """
        پارامترها:
        -----------
        criteria_funcs_weights: لیست تاپل (نام معیار, تابع معیار, وزن)
        positive_label: مقدار کلاس مثبت برای محاسبه FN
        """
        self.criteria = [(name, func) for name, func, w in criteria_funcs_weights]
        self.weights_dict = {name: w for name, func, w in criteria_funcs_weights}
        self.positive_label = positive_label

        # ساخت آبجکت مدیر رای‌گیری وزنی FN
        self.split_manager = FNWeightedSplitManager(
            criteria_list=[CriterionWrapper(name, func) for name, func in self.criteria],
            positive_label=positive_label
        )

        # این مدل از CustomDecisionTreeClassifier به عنوان پایه استفاده می‌کند
        self.model = CustomDecisionTreeClassifier(custom_metric=self.weighted_voting_criterion)

    def weighted_voting_criterion(self, y_left, y_right):
        """
        تابع امتیازگیری رای‌گیری وزنی برای custom-tree-classifier
        """
        score_list = []
        fn_list = []

        for name, func in self.criteria:
            score = func(y_left, y_right)
            score_list.append(score)

            # FN در این تابع باید از برچسب‌های واقعی و پیش‌بینی جدا باشد.
            # چون در اینجا پیش‌بینی هنوز نداریم، می‌گذاریم فرض FN=0
            # یا می‌شود مقدار FN را خارج از این تابع با داده‌های پیش‌بینی واقعی تعیین کرد.
            fn_list.append(0)

        weights = np.array([self.weights_dict[name] for name, _ in self.criteria])
        weights = weights / weights.sum()

        weighted_score = float(np.dot(score_list, weights))
        return weighted_score

    def fit(self, X, y):
        """
        آموزش مدل درخت تصمیم با رای‌گیری وزنی معیارها
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        پیش‌بینی برچسب‌ها با مدل آموزش دیده
        """
        return self.model.predict(X)


class CriterionWrapper:
    """
    کلاس کمکی برای سازگار کردن توابع معیار با کلاس مورد نیاز FNWeightedSplitManager
    """
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def calculate_score(self, y_left, y_right):
        return self.func(y_left, y_right)


# نمونه استفاده ساده
if __name__ == "__main__":
    # لیست معیارها به همراه وزن دلخواه (جمع وزن‌ها مساوی 1 یا نرمال می‌شود)
    criteria_with_weights = [
        ("gini", gini_criterion, 0.15),
        ("gain_ratio", gain_ratio_criterion, 0.15),
        ("twoing", twoing_criterion, 0.10),
        ("normalized_gain", normalized_gain_criterion, 0.10),
        ("multi_class_hellinger", multi_class_hellinger, 0.10),
        ("marshall", marsh_criterion, 0.05),
        ("g_statistic", g_statistic_criterion, 0.05),
        ("dkm", dkm_criterion, 0.10),
        ("chi_squared", chi_squared_criterion, 0.05),
        ("bhattacharyya", bhattacharyya_criterion, 0.10),
        ("kolmogorov_smirnov", kolmogorov_smirnov_criterion, 0.05),
    ]

    # آماده‌سازی مدل
    model = WeightedDecisionTreeModel(criteria_with_weights, positive_label=1)

    # فرضی: X_train و y_train داده‌های آموزشی مدل
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    print("مدل WeightedDecisionTree آماده است برای آموزش و پیش‌بینی.")
