import numpy as np
#from custom_tree_classifier.models.decision_tree import CustomDecisionTreeClassifier
from custom_tree_classifier.metrics.metric_base import MetricBase
# از آنجایی که دیگر از این کلاس استفاده نمی‌کنیم، می‌توان آن را حذف کرد یا کامنت کرد
# from utils.voting_split_manager import FNWeightedSplitManager

# Import معیارها (بدون تغییر)
#from criteria.gini import gini_criterion
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

# ================================================================
# 🔥 کلاس ۱: SingleCriterionMetric (نسخه نهایی و اصلاح شده) 🔥
# این نسخه به درستی مسئولیت امتیازدهی را به خود تابع معیار می‌سپارد.
# ================================================================
class SingleCriterionMetric(MetricBase):
    """
    یک کلاس کمکی بازطراحی شده که یک تابع معیار کیفیت تقسیم (Split Quality Function)
    را در قالبی که کتابخانه CustomDecisionTreeClassifier می‌پذیرد، بسته‌بندی می‌کند.
    """
    def __init__(self, name, func):
        super().__init__()
        self.name = name
        self.func = func  # این تابع باید (y_left, y_right) را بپذیرد

    def compute_metric(self, metric_data: np.ndarray) -> float:
        """
        این متد برای معیارهای تقسیم-محور معنای خاصی ندارد.
        برگرداندن مقدار ثابت 0.0 امن و صحیح است.
        """
        return 0.0

    def compute_delta(self, split, metric_data):
        """
        این متد به سادگی تابع معیار را با داده‌های تقسیم شده فراخوانی می‌کند.
        مقدار بازگشتی از تابع معیار، مستقیماً به عنوان امتیاز تقسیم استفاده می‌شود.
        """
        try:
            y = metric_data[:, 0] if metric_data.ndim > 1 else metric_data
            
            # جدا کردن داده‌ها بر اساس تقسیم
            if split.dtype == bool:
                y_left, y_right = y[split], y[~split]
            else:
                mask = np.ones(len(y), dtype=bool)
                mask[split] = False
                y_left, y_right = y[split], y[mask]

            # اگر یکی از فرزندان خالی باشد، تقسیم بی‌معناست
            if len(y_left) == 0 or len(y_right) == 0:
                return 0.0

            # 🔥 تغییر کلیدی: فراخوانی مستقیم تابع معیار 🔥
            # به جای محاسبه دستی information gain، خود تابع معیار (مثلاً gain_ratio_criterion)
            # مسئول محاسبه امتیاز است.
            return float(self.func(y_left, y_right))
        
        except Exception as e:
            # در صورت بروز خطا در تابع معیار، امتیاز صفر برگردان
            # print(f"[Debug] Error in criterion '{self.name}': {e}") # برای دیباگ می‌توانید این خط را فعال کنید
            return 0.0


# ================================================================
# کلاس ۲: WeightedVotingMetric (با متد evaluate اصلاح شده)
# این مغز متفکر مدل شماست.
# ================================================================
class WeightedVotingMetric(MetricBase):
    """
    یک متریک سفارشی که از ترکیب وزن‌دار چندین معیار برای ارزیابی تقسیم‌ها استفاده می‌کند.
    وزن‌ها به صورت پویا در هر گره بر اساس عملکرد هر معیار در کاهش False Negatives محاسبه می‌شوند.
    """
    def __init__(self, criteria, a):
        super().__init__()
        self.criteria = criteria  # لیستی از (نام، تابع) معیارها
        self.a = a  # کل دیتاست ویژگی‌ها (X)
        self.weights_dict = {}  # دیکشنری برای نگهداری وزن‌های محاسبه شده در هر گره
        self._fn_cache = {}  # کش برای جلوگیری از محاسبات تکراری FN

    def estimate_fn_for_criterion(self, name, x_data, y_node):
        """برای یک معیار مشخص، با ساختن یک درخت موقت، مقدار FN را تخمین می‌زند."""
        from custom_tree_classifier.models.decision_tree import CustomDecisionTreeClassifier
        cache_key = f"{name}_{len(x_data)}_{hash(y_node.tobytes())}_{hash(x_data.tobytes())}"
        if cache_key in self._fn_cache:
            return self._fn_cache[cache_key]

        if x_data.shape[0] != y_node.shape[0]:
            return np.inf # در صورت عدم تطابق داده، یک پنالتی بزرگ در نظر می‌گیریم

        try:
            criterion_func = dict(self.criteria)[name]
            metric_obj = SingleCriterionMetric(name, criterion_func)
            
            model = CustomDecisionTreeClassifier(max_depth=10, metric=metric_obj)
            metric_data = y_node.reshape(-1, 1)
            model.fit(x_data, y_node, metric_data)
            
            probas = model.predict_proba(x_data)
            preds = (probas[:, 1] > 0.5).astype(int) if probas.shape[1] > 1 else np.zeros_like(y_node)
            
            fn_count = np.sum((y_node == 1) & (preds == 0))
            self._fn_cache[cache_key] = fn_count
            return fn_count
        except Exception as e:
            print(f"[ERROR] در محاسبه FN برای {name}: {e}")
            return np.inf

    # در کلاس WeightedVotingMetric

    def update_weights_dynamic(self, x_data, y_node):
        """
        🔥 نسخه نهایی با Softmax 🔥
        FNها را برای تمام معیارها محاسبه و دیکشنری وزن‌ها را برای گره فعلی به‌روز می‌کند.
        """
        if len(x_data) < 2:
            return self.weights_dict 

        fn_values = np.array([self.estimate_fn_for_criterion(name, x_data, y_node) for name, _ in self.criteria], dtype=float)
        print(f"  FNs calculated: {dict(zip([c[0] for c in self.criteria], fn_values))}")

        # ===== بخش کلیدی: محاسبه وزن با Softmax =====

        # پارامتر دما (قابل تنظیم)
        # مقدار بالاتر باعث توزیع یکنواخت‌تر وزن‌ها می‌شود.
        # مقدار پایین‌تر باعث تمرکز بیشتر روی بهترین معیارها می‌شود.
        temperature = 1.0 

        # برای جلوگیری از مقادیر بی‌نهایت در تابع نمایی (exp)، امتیازها را شیفت می‌دهیم.
        # این یک تکنیک استاندارد برای پایداری عددی Softmax است و روی نتیجه نهایی تأثیری ندارد.
        scores = -fn_values  # FN کمتر، امتیاز بالاتر
        scores -= np.max(scores) # شیفت دادن برای پایداری

        # محاسبه Softmax
        exp_scores = np.exp(scores / temperature)
        
        # اطمینان از اینکه مخرج صفر نشود
        sum_exp_scores = np.sum(exp_scores)
        if sum_exp_scores > 1e-9:
            normalized_weights = exp_scores / sum_exp_scores
        else:
            # اگر همه امتیازها بسیار منفی باشند، وزن مساوی بده
            num_criteria = len(self.criteria)
            normalized_weights = np.full(num_criteria, 1.0 / num_criteria)
        
        # ============================================
        
        self.weights_dict = {name: weight for (name, _), weight in zip(self.criteria, normalized_weights)}
        print(f"  New weights set (using Softmax): {self.weights_dict}")
        return self.weights_dict



# ================================================================
# 🔥 بلوک نهایی کد برای کلاس WeightedVotingMetric 🔥
# ================================================================

    def evaluate(self, y_left, y_right, split_info=None):
        """
        نسخه نهایی: امتیاز نهایی یک تقسیم را با نرمال‌سازی امتیازات معیارها (Min-Max) و سپس
        اعمال میانگین وزنی، محاسبه می‌کند.
        """
        if len(y_left) == 0 or len(y_right) == 0:
            return 0.0

        # اگر به هر دلیلی وزن‌ها محاسبه نشده باشند، از وزن مساوی استفاده کن
        if not self.weights_dict:
            print("[WARNING] `evaluate` فراخوانی شد در حالی که وزن‌ها تنظیم نشده بودند. استفاده از وزن مساوی.")
            self.weights_dict = {name: 1.0 / len(self.criteria) for name, _ in self.criteria}
        
        # ۱. محاسبه امتیاز خام برای هر معیار با فراخوانی مستقیم تابع آن
        raw_scores = []
        for name, func in self.criteria:
            try:
                # امتیازدهی مستقیم توسط خود تابع معیار
                raw_scores.append(float(func(y_left, y_right)))
            except Exception:
                raw_scores.append(0.0)

        # ۲. نرمال‌سازی امتیازات به بازه [0, 1] برای مقایسه عادلانه (Min-Max Scaling)
        scores_np = np.array(raw_scores)
        min_score, max_score = np.min(scores_np), np.max(scores_np)
        
        # جلوگیری از تقسیم بر صفر اگر همه امتیازات یکسان باشند
        if (max_score - min_score) > 1e-9:
            normalized_scores = (scores_np - min_score) / (max_score - min_score)
        else:
            # اگر همه امتیازات یکسان بودند، همه نرمالایز شده‌ها 0.5 می‌شوند
            normalized_scores = np.full_like(scores_np, 0.5)

        # ۳. خواندن وزن‌های از پیش محاسبه شده
        weights = np.array([self.weights_dict.get(name, 0.0) for name, _ in self.criteria])

        # ۴. محاسبه امتیاز نهایی با استفاده از امتیازات نرمالایز شده
        if weights.sum() > 0:
            # np.dot حاصلضرب داخلی دو بردار را محاسبه می‌کند که همان میانگین وزنی است
            weighted_score = np.dot(normalized_scores, weights)
        else:
            weighted_score = np.mean(normalized_scores) if len(normalized_scores) > 0 else 0.0
            
        return float(weighted_score)


    def compute_metric(self, metric_data: np.ndarray) -> float:
        """
        این متد با معماری جدید، دیگر کاربرد مستقیمی ندارد، اما برای حفظ سازگاری
        با ساختار کلی، یک مقدار پایه برمی‌گرداند.
        """
        # می‌توانید این بخش را برای سادگی خالی بگذارید یا یک میانگین ساده برگردانید.
        # برگرداندن صفر امن‌ترین گزینه است.
        return 0.0


    def compute_delta(self, split: np.ndarray, metric_data: np.ndarray) -> float:
        """
        این متد به درستی وظیفه تقسیم داده‌ها را انجام داده و سپس `evaluate` را
        برای محاسبه امتیاز نهایی فراخوانی می‌کند.
        """
        y = metric_data[:, 0] if metric_data.ndim > 1 else metric_data
        
        if split.dtype == bool:
            y_left, y_right = y[split], y[~split]
        else:
            mask = np.ones(len(y), dtype=bool)
            mask[split] = False
            y_left, y_right = y[split], y[mask]
            
        # فراخوانی متد evaluate نهایی و اصلاح شده
        return self.evaluate(y_left, y_right)



# ================================================================
# کلاس ۳: CriterionWrapper (بدون تغییر)
# ================================================================
class CriterionWrapper:
    # این کلاس اگر در جای دیگری استفاده نمی‌شود، می‌تواند حذف شود.
    # در این معماری جدید، نقش مستقیمی ندارد.
    def __init__(self, name, func):
        self.name = name
        self.func = func
        
    def calculate_score(self, y_left, y_right):
        try:
            return self.func(y_left, y_right)
        except Exception:
            return 0.0

# ================================================================
# کلاس ۴: WeightedDecisionTreeModel (بدون تغییر)
# این کلاس ارکستراتور اصلی است و نیازی به تغییر ندارد.
# ================================================================
class WeightedDecisionTreeModel:
# در فایل src/model/weighted_decision_tree.py
# در کلاس WeightedDecisionTreeModel

    def __init__(self, criteria_funcs_weights=None, positive_label=1, max_depth=5, a=None):
        if criteria_funcs_weights is None:
            criteria_funcs_weights = [
                #("gini", gini_criterion), ("gain_ratio", gain_ratio_criterion),
                ("twoing", twoing_criterion), ("normalized_gain", normalized_gain_criterion),
                ("multi_class_hellinger", multi_class_hellinger), ("marshall", marsh_criterion),
                ("g_statistic", g_statistic_criterion), ("dkm", dkm_criterion),
                ("chi_squared", chi_squared_criterion), ("bhattacharyya", bhattacharyya_criterion),
                ("kolmogorov_smirnov", kolmogorov_smirnov_criterion),
            ]
        self.criteria = criteria_funcs_weights
        self.positive_label = positive_label
        
        # 🔥 خط اصلاح شده و بسیار مهم که فراموش شده بود
        self.max_depth = max_depth
        
        if a is None:
            raise ValueError("a باید داده ویژگی‌های کامل را به مدل بدهید.")
        
        self.a = np.array(a.values) if hasattr(a, "values") else np.array(a)
        
        self.metric = WeightedVotingMetric(self.criteria, self.a)
        self.model = None


    def compute_initial_weights(self, X, y):
        from custom_tree_classifier.models.decision_tree import CustomDecisionTreeClassifier

        """وزن‌های اولیه را برای گره ریشه محاسبه می‌کند."""
        print("[INIT] شروع محاسبه وزن‌های اولیه برای گره ریشه...")
        # از همان منطق `update_weights_dynamic` استفاده می‌کنیم
        initial_weights = self.metric.update_weights_dynamic(X, y)
        self.metric.weights_dict = initial_weights
        print(f"وزن‌های اولیه تنظیم شد: {initial_weights}")

    # در کلاس WeightedDecisionTreeModel
    
# در فایل src/model/weighted_decision_tree.py

    def fit(self, x, y):
        """
        مدل را با استفاده از داده‌های ورودی آموزش می‌دهد.
        """
        # 🔥 اصلاحیه ۱: ایمپورت محلی برای شکستن چرخه وابستگی
        from custom_tree_classifier.models.decision_tree import CustomDecisionTreeClassifier

        # 🔥 اصلاحیه ۲ (بسیار مهم): ساخت مدل در لحظه نیاز
        # اگر مدل ساخته نشده (None است)، آن را بساز
        if self.model is None:
            # در متد __init__ کلاس WeightedDecisionTreeModel
            #self.model = CustomDecisionTreeClassifier() # درست: یک نمونه از کلاس ساخته‌اید

            self.model = CustomDecisionTreeClassifier(
                max_depth=self.max_depth,
                metric=self.metric
            )
        
        # مرحله ۱: آماده‌سازی داده‌ها (بدون تغییر)
        X_fit = x.values if hasattr(x, "values") else x
        y_fit = y.values if hasattr(y, "values") else y
        
        print(f"شروع آموزش مدل - شکل داده: x={X_fit.shape}, y={y_fit.shape}")
        
        # مرحله ۲: محاسبه و تنظیم وزن‌های اولیه (بدون تغییر)
        self.compute_initial_weights(X_fit, y_fit)
        
        # مرحله ۳: آماده‌سازی داده متریک (بدون تغییر)
        metric_data = y_fit.reshape(-1, 1)

 

        import custom_tree_classifier.models.decision_tree as dt

        print("Path of the loaded module:", dt.__file__)


        # مرحله ۴: فراخوانی fit اصلی کتابخانه (حالا بدون خطا اجرا می‌شود)
        self.model.fit(X_fit, y_fit, metric_data)



        print("آموزش تکمیل شد!")


    def predict(self, x):
        X_pred = x.values if hasattr(x, "values") else x
        probas = self.model.predict_proba(X_pred)
        if probas.shape[1] > 1:
            return np.argmax(probas, axis=1)
        else:
            return (probas[:, 0] > 0.5).astype(int)

    def predict_proba(self, x):
        X_pred = x.values if hasattr(x, "values") else x
        return self.model.predict_proba(X_pred)

# ================================================================
# بخش اجرایی (بدون تغییر)
# ================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # ساخت داده نمونه
    X_data, y_data = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, n_classes=2, random_state=42)
    
    # ساخت و آموزش مدل
    model = WeightedDecisionTreeModel(a=X_data, max_depth=5)
    model.fit(X_data, y_data)
    
    # پیش‌بینی
    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data)
    
    print("\n--- نتایج نهایی ---")
    print(f"تعداد پیش‌بینی‌ها: {len(predictions)}")
    print(f"نمونه پیش‌بینی‌ها: {predictions[:10]}")
    print(f"نمونه احتمالات: \n{probabilities[:5]}")
    print("\nمدل با موفقیت اجرا شد!")
