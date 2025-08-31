import numpy as np

class FNBasedWeighting:
    """
    کلاس کمکی برای محاسبه وزن معیارها براساس FN روی داده
    """
    def __init__(self, criteria_list):
        self.criteria_list = criteria_list
    
    def compute_weights(self, X, y, tree_builder):
        """
        برای هر معیار یک مدل با آن معیار می‌سازد و FN را برای کل داده حساب می‌کند.
        """
        fns = []
        epsilon = 1e-8  # برای جلوگیری از تقسیم بر صفر
        
        for name, criterion_func in self.criteria_list:
            # مدل درخت با این معیار بساز
            model = tree_builder(criterion_func)   # تابعی که مدل سفارشی می‌سازد
            model.fit(X, y)
            y_pred = model.predict(X)
            fn = np.sum((y == 1) & (y_pred == 0))
            fns.append(fn + epsilon)  # جمع کردن اپسیلون برای جلوگیری از صفر
        
        inv_fns = 1 / np.array(fns)
        weights = inv_fns / inv_fns.sum()
        return {name: weight for (name, _), weight in zip(self.criteria_list, weights)}

class DynamicWeightedDecisionTree:
    """
    مدل درخت تصمیم چند معیار با وزن‌دهی بر اساس FN –
    هم حالت وزن‌دهی داده‌محور و هم وزن‌دهی پویا در هر نود
    """
    def __init__(self, criteria_list, tree_builder, local_weighting=False):
        """
        criteria_list: لیست [(name, criterion_func)]
        tree_builder: تابع می‌گیرد criterion_func و مدل سفارشی می‌سازد
        local_weighting: اگر True باشد، در هر نود وزن‌ها بر اساس FN محلی داده محاسبه می‌شود
        """
        self.criteria_list = criteria_list
        self.tree_builder = tree_builder
        self.local_weighting = local_weighting
        self.root = None
    
    def fit(self, X, y):
        """
        آموزش درخت اصلی با رای‌گیری وزنی تقدم معیارها
        """
        if self.local_weighting:
            self.root = self._fit_node_local(X, y)
        else:
            # وزن داده‌محور (کل داده) – حالت اول
            weight_calc = FNBasedWeighting(self.criteria_list)
            weights = weight_calc.compute_weights(X, y, self.tree_builder)
            self.root = self._fit_node_fixed(X, y, weights)
    
    def _fit_node_local(self, X, y):
        """
        ساخت نود با وزن‌دهی پویا (محلی) FN
        """
        # در هر نود داده کم باشد یا pure باشد، برگ بساز
        if len(np.unique(y)) == 1 or len(y) < 5:
            return {"leaf": True, "class": y[0]}
        
        # وزن معیارها با داده‌های محلی این نود
        weight_calc = FNBasedWeighting(self.criteria_list)
        weights = weight_calc.compute_weights(X, y, self.tree_builder)
        
        # پیدا کردن بهترین split با رای‌گیری وزنی امتیاز معیارها
        best_score, best_feat, best_thresh = -np.inf, None, None
        n_features = X.shape[1]
        for feat in range(n_features):
            values = np.unique(X[:, feat])
            for thresh in values:
                left_idx = X[:, feat] <= thresh
                right_idx = ~left_idx
                if np.sum(left_idx)==0 or np.sum(right_idx)==0:
                    continue
                score = self._combine_criteria(X[left_idx], y[left_idx], X[right_idx], y[right_idx], weights)
                if score > best_score:
                    best_score, best_feat, best_thresh = score, feat, thresh
        
        # تقسیم نود یا ساخت برگ
        if best_feat is None:
            return {"leaf": True, "class": np.bincount(y).argmax()}
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        return {
            "leaf": False,
            "feature": best_feat,
            "thresh": best_thresh,
            "left": self._fit_node_local(X[left_idx], y[left_idx]),
            "right": self._fit_node_local(X[right_idx], y[right_idx])
        }
    
    def _fit_node_fixed(self, X, y, weights):
        """
        ساخت نود با وزن ثابت FN (کل داده)
        """
        # مانند تابع بالا اما فقط weights ثابت داریم
        if len(np.unique(y)) == 1 or len(y) < 5:
            return {"leaf": True, "class": y[0]}
        best_score, best_feat, best_thresh = -np.inf, None, None
        n_features = X.shape[1]
        for feat in range(n_features):
            values = np.unique(X[:, feat])
            for thresh in values:
                left_idx = X[:, feat] <= thresh
                right_idx = ~left_idx
                if np.sum(left_idx)==0 or np.sum(right_idx)==0:
                    continue
                score = self._combine_criteria(X[left_idx], y[left_idx], X[right_idx], y[right_idx], weights)
                if score > best_score:
                    best_score, best_feat, best_thresh = score, feat, thresh
        if best_feat is None:
            return {"leaf": True, "class": np.bincount(y).argmax()}
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        return {
            "leaf": False,
            "feature": best_feat,
            "thresh": best_thresh,
            "left": self._fit_node_fixed(X[left_idx], y[left_idx], weights),
            "right": self._fit_node_fixed(X[right_idx], y[right_idx], weights)
        }
    
    def _combine_criteria(self, X_left, y_left, X_right, y_right, weights):
        """
        ترکیب امتیاز معیارها با وزن‌دهی برای یک split
        """
        scores = []
        for name, criterion_func in self.criteria_list:
            score = criterion_func(y_left, y_right)
            scores.append(weights[name] * score)
        return sum(scores)
    
    def predict(self, X):
        """
        پیش‌بینی با حرکت درخت
        """
        preds = []
        for x in X:
            node = self.root
            while not node["leaf"]:
                if x[node["feature"]] <= node["thresh"]:
                    node = node["left"]
                else:
                    node = node["right"]
            preds.append(node["class"])
        return np.array(preds)

# --- نمونه استفاده:
# فرض مثال، criterion_func باید فقط روی برچسب‌ها اجرا شود
def gini_criterion(y_left, y_right):
    # یک مثال ساده (واقعی را جایگزین کن)
    def gini(y):
        probs = np.bincount(y) / len(y)
        return 1 - np.sum(probs**2)
    n_l = len(y_left)
    n_r = len(y_right)
    total_n = n_l + n_r
    left_gini = gini(y_left)
    right_gini = gini(y_right)
    return total_n - (n_l * left_gini + n_r * right_gini)

criteria_list = [("gini", gini_criterion)]  # معیارهات را مشابه همین اضافه کن...

def tree_builder(criterion_func):
    # یک مدل بسیار ساده (جایگزین کن با مدل سفارشی خودت)
    class SimpleTree:
        def __init__(self, criterion):
            self.criterion = criterion
        def fit(self, X, y): self.X, self.y = X, y
        def predict(self, X): return np.random.choice(np.unique(self.y), len(X))
    return SimpleTree(criterion_func)

# مدل با وزن پویا در هر نود:
model_local = DynamicWeightedDecisionTree(criteria_list, tree_builder, local_weighting=True)
# مدل با وزن ثابت داده‌محور:
model_fixed = DynamicWeightedDecisionTree(criteria_list, tree_builder, local_weighting=False)

# آموزش و تست:
# X, y را با داده واقعی جایگزین کن مثلاً:
# X = np.random.randint(0, 5, (100, 4))
# y = np.random.randint(0, 2, 100)
# model_local.fit(X, y)
# y_pred = model_local.predict(X)
