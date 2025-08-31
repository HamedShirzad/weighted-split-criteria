import sys
import os
import importlib
import pandas as pd
from model.weighted_decision_tree import WeightedDecisionTreeModel

# اضافه کردن مسیر src پروژه به sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# بارگذاری مجدد ماژول‌ها برای اطمینان از بارگذاری تغییرات
import custom_tree_classifier.models.decision_tree
import model.weighted_decision_tree

importlib.reload(custom_tree_classifier.models.decision_tree)
importlib.reload(model.weighted_decision_tree)

# خواندن دیتاست
csv_path = r'C:\Users\HsH-HsH\weighted_split_criteria\sample_dataset_30_numeric.csv'
df = pd.read_csv(csv_path)

# آماده‌سازی داده‌ها
y = df['Churn'].values  # نام ستون هدف را بررسی کنید
X = df.drop(columns=['Churn']).values

# ساخت مدل و آموزش
# فرض می‌کنیم X_data داده‌های ویژگی شماست
model = WeightedDecisionTreeModel(
    a=X_data,
    criteria_funcs_weights=all_criteria,
    max_depth=3
)

print("شروع آموزش مدل...")
model.fit(X, y)
print("آموزش مدل با موفقیت به پایان رسید.")
