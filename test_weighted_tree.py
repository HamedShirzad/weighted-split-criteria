import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# افزودن src به sys.path برای ایمپورت صحیح تمام ماژول‌ها و معیارها
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# ------ وارد کردن تمام معیارهای واقعی پروژه ------
from criteria.gini import GiniCriterion
from criteria.qg import GainRatioCriterion
from criteria.bhy import BhattacharyyaCriterion
from criteria.dkm import DKMCriterion
from criteria.cs import ChiSquaredCriterion
from criteria.gs import GStatisticCriterion
from criteria.marsh import MarshallCriterion
from criteria.mch import MultiClassHellingerCriterion
from criteria.ng import NormalizedGainCriterion
from criteria.ks import KolmogorovSmirnovCriterion
from criteria.twoing import TwoingCriterion

from model.weighted_decision_tree import WeightedDecisionTree

# --- 1. بارگذاری داده آماده و تبدیل برچسب‌ها به عددی ---
data_path = os.path.join("data", "dataset.csv")
df = pd.read_csv(data_path)

label_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
# اگر داده iris نباشد، همین قالب را با مقادیر درست خودت جایگزین کن

X = df.drop(columns=['label']).values            # ویژگی‌ها
y = df['label'].map(label_mapping).values        # برچسب عددی (اجباری!)

# --- 2. تقسیم داده به آموزش و تست ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- 3. تعریف همه معیارهای پروژه در لیست ---
criteria_list = [
    GiniCriterion(),
    GainRatioCriterion(),
    BhattacharyyaCriterion(),
    DKMCriterion(),
    ChiSquaredCriterion(),
    GStatisticCriterion(),
    MarshallCriterion(),
    MultiClassHellingerCriterion(),
    NormalizedGainCriterion(),
    KolmogorovSmirnovCriterion(),
    TwoingCriterion(),
]

tree = WeightedDecisionTree(
    criteria_list=criteria_list,
    max_depth=3,            # قابل تنظیم (مثلاً ۴ یا ۵)
    min_samples_split=5,    # بسته به داده و نیازت تغییر بده
    positive_label=1        # کلاس مثبت پروژه (مثلاً برای iris کلاس وسط)
)

# --- 4. آموزش مدل و تست روی داده تست ---
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

# --- 5. گزارش نهایی ---
print("=== Test Results: Weighted Voting With All Custom Criteria ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
fn_total = np.sum((y_test == 1) & (y_pred != 1))
print("Total False Negatives (FN):", fn_total)
print(f"Sample Test 0: predicted={y_pred[0]}, true={y_test[0]}")
print("\nپایپ‌لاین کامل بدون نیاز به هیچ مرحله دستی دیگر اجرا شد.")

input('Press Enter to exit')
