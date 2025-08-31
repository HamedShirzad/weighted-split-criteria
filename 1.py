# ================================================================
# سلول ۱: ایمپورت کتابخانه‌های پایه
# ================================================================
# این سلول فقط کتابخانه‌های اصلی مورد نیاز برای مدیریت داده و
# تعامل با سیستم فایل را وارد می‌کند.

import pandas as pd
import numpy as np
import sys
import os

print("✅ کتابخانه‌های پایه (pandas, numpy, sys, os) با موفقیت ایمپورت شدند.")

import sys
import os

project_root = os.path.abspath(os.getcwd())
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"✅ مسیر '{src_path}' به sys.path اضافه شد.")
else:
    print(f"✅ مسیر '{src_path}' از قبل در sys.path وجود دارد.")

# ================================================================
# سلول ۲: افزودن مسیر پروژه به sys.path
# ================================================================
# این سلول مسیر پوشه 'src' را به لیست مسیرهایی که پایتون برای
# پیدا کردن ماژول‌ها جستجو می‌کند، اضافه می‌کند.

import sys
import os

try:
    # مسیر ریشه پروژه را به صورت خودکار از روی محل نوت‌بوک پیدا می‌کنیم.
    project_root = os.path.abspath(os.getcwd())
    # مسیر کامل پوشه src را می‌سازیم.
    src_path = os.path.join(project_root, 'src')

    # برای جلوگیری از اضافه شدن تکراری، ابتدا وجود مسیر را چک می‌کنیم.
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"✅ مسیر '{src_path}' با موفقیت به sys.path اضافه شد.")
    else:
        print(f"✅ مسیر '{src_path}' از قبل در sys.path وجود دارد.")

except Exception as e:
    print(f"❌ خطا در تنظیم مسیر پروژه: {e}")
    print("لطفاً مطمئن شوید که این نوت‌بوک در پوشه ریشه پروژه (C:\\Users\\HsH-HsH\\weighted_split_criteria) قرار دارد.")

# ================================================================
# سلول ۲ (جدید): بارگذاری مجدد و اجباری ماژول‌ها
# ================================================================
# این سلول برای حل مشکل کش شدن (caching) ماژول‌ها در ژوپیتر ضروری است.
# با این کار، مطمئن می‌شویم که همیشه آخرین نسخه از فایل‌های کد شما بارگذاری می‌شود.

import importlib
import sys

# برای جلوگیری از خطاهای احتمالی، مسیرها را دوباره چک می‌کنیم
# (اینجا رشته به شکل raw تعریف شده تا بک‌اسلش‌ها درست در نظر گرفته شوند)
module_path = r'C:\Users\HsH-HsH\weighted_split_criteria\src'
if module_path not in sys.path:
    sys.path.append(module_path)

# ایمپورت ماژول‌هایی که در حال ویرایش آن‌ها هستیم
# توجه کنید که ما خود ماژول را ایمپورت می‌کنیم، نه کلاس را
import custom_tree_classifier.models.decision_tree
import model.weighted_decision_tree

# بارگذاری مجدد و اجباری این ماژول‌ها با استفاده از importlib
# این دستور به پایتون می‌گوید که حافظه کش را نادیده بگیرد و فایل را از نو بخواند
importlib.reload(custom_tree_classifier.models.decision_tree)
importlib.reload(model.weighted_decision_tree)

# حالا می‌توانیم با اطمینان از کلاس‌های آپدیت شده استفاده کنیم
# این ایمپورت باید حتماً بعد از reload انجام شود
from model.weighted_decision_tree import WeightedDecisionTreeModel

print("✅ ماژول‌های سفارشی با موفقیت و با آخرین تغییرات بارگذاری مجدد شدند.")

# ================================================================
# سلول ۳: ایمپورت ماژول‌های اصلی پروژه
# ================================================================
# در این سلول، مدل اصلی و تمام توابع معیار تقسیم را که برای ساخت
# و آموزش مدل نیاز داریم، از مسیرهای صحیح درون پوشه src ایمپورت می‌کنیم.

try:
    # ایمپورت کلاس اصلی مدل از مسیر: src/model/weighted_decision_tree.py
    from model.weighted_decision_tree import WeightedDecisionTreeModel

    # ایمپورت تمام ۱۱ تابع معیار از مسیر: src/criteria/
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

    print("✅ مدل اصلی و تمام ۱۱ معیار تقسیم با موفقیت ایمپورت شدند.")
    print("آماده برای بارگذاری داده‌ها و ساخت مدل.")

except ImportError as e:
    print(f"❌ خطا در ایمپورت ماژول‌های پروژه: {e}")
    print("این خطا معمولاً به دلیل عدم اجرای موفقیت‌آمیز سلول قبلی (تنظیم مسیر) رخ می‌دهد.")
except Exception as e:
    print(f"❌ یک خطای غیرمنتظره رخ داد: {e}")

# ================================================================
# سلول ۴: بارگذاری و آماده‌سازی داده‌ها
# ================================================================
# در این سلول، دیتاست انتخاب شده توسط شما (sample_dataset_30_numeric.csv)
# از فایل CSV خوانده و برای آموزش مدل آماده می‌شود.

import pandas as pd
import numpy as np

try:
    # مسیر دیتاست را مشخص می‌کنیم.
    # با توجه به انتخاب شما، این فایل باید در ریشه پروژه قرار داشته باشد.
    dataset_path = 'sample_dataset_30_numeric.csv'

    # خواندن فایل CSV با استفاده از pandas
    df = pd.read_csv(dataset_path)

    print(f"✅ دیتاست '{dataset_path}' با موفقیت بارگذاری شد.")
    print("\n--- اطلاعات اولیه دیتاست ---")
    print(f"شکل دیتاست (تعداد سطرها و ستون‌ها): {df.shape}")
    print("\n۵ سطر اول دیتاست:")
    # نمایش ۵ سطر اول برای بررسی صحت داده‌ها
    print(df.head())

    # --- آماده‌سازی داده‌ها برای مدل ---
    # تمام ستون‌ها به جز ستون آخر به عنوان ویژگی‌ها (X) در نظر گرفته می‌شوند.
    X = df.iloc[:, :-1]
    # ستون آخر به عنوان متغیر هدف (y) در نظر گرفته می‌شود.
    y = df.iloc[:, -1]

    # تبدیل دیتافریم‌های pandas به آرایه‌های NumPy
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    print("\n✅ داده‌ها با موفقیت به فرمت NumPy تبدیل شدند و برای آموزش آماده هستند.")
    print(f"شکل ماتریس ویژگی‌ها (X): {X_np.shape}")
    print(f"شکل بردار هدف (y): {y_np.shape}")

except FileNotFoundError:
    print(f"❌ خطا: فایل دیتاست در مسیر '{dataset_path}' پیدا نشد.")
    print("لطفاً مطمئن شوید که نام فایل صحیح است و در پوشه ریشه پروژه قرار دارد.")
except Exception as e:
    print(f"❌ یک خطای غیرمنتظره در هنگام پردازش داده‌ها رخ داد: {e}")

# ================================================================
# سلول ۵: آماده‌سازی لیست معیارهای ترکیبی
# ================================================================
# در این سلول، لیستی از تاپل‌ها (نام معیار، تابع معیار) را می‌سازیم.
# این لیست به عنوان ورودی به مدل ما داده خواهد شد تا بداند از کدام
# معیارها برای ساخت درخت تصمیم استفاده کند.

try:
    all_criteria = [
        ("gini", gini_criterion),
        ("gain_ratio", gain_ratio_criterion),
        ("twoing", twoing_criterion),
        ("normalized_gain", normalized_gain_criterion),
        ("multi_class_hellinger", multi_class_hellinger),
        ("marshall", marsh_criterion),
        ("g_statistic", g_statistic_criterion),
        ("dkm", dkm_criterion),
        ("chi_squared", chi_squared_criterion),
        ("bhattacharyya", bhattacharyya_criterion),
        ("kolmogorov_smirnov", kolmogorov_smirnov_criterion),
    ]

    print("✅ لیست معیارها با موفقیت ساخته شد.")
    print(f"   - تعداد کل معیارها: {len(all_criteria)}")
    print("این لیست آماده است تا به عنوان ورودی به مدل داده شود.")

except:
    pass  # هر اتفاقی افتاد، نادیده بگیر

# ================================================================
# سلول ۶ (نسخه دیباگ - بدون try-except و با print اصلاح شده)
# ================================================================
# این نسخه از سلول، بلوک try-except را ندارد تا بتوانیم
# گزارش خطای کامل پایتون را مشاهده کنیم.

print("\n--- شروع فرآیند ساخت نمونه مدل (حالت دیباگ) ---")

# یک نمونه از مدل شما را می‌سازیم.
model = WeightedDecisionTreeModel(
    criteria_funcs_weights=all_criteria,
    max_depth=3,
    a=X_np
)

print("✅ یک نمونه از مدل WeightedDecisionTreeModel با موفقیت ساخته شد.")
# 🔥 اصلاحیه: مسیر دسترسی به max_depth اصلاح شد
print(f"   - حداکثر عمق تنظیم شده: {model.max_depth}")
print(f"   - تعداد معیارهای ترکیبی: {len(model.criteria)}")

print("\nمدل آماده برای آموزش در سلول بعدی است.")

# ================================================================
# سلول ۷ (نسخه دیباگ - بدون try-except): آموزش مدل
# ================================================================
# این نسخه برای دیباگ دقیق است و هیچ خطایی را مدیریت نمی‌کند.

print("--- شروع فرآیند آموزش مدل (fit) ---")

# فراخوانی متد fit برای شروع فرآیند ساخت درخت تصمیم.
# اگر خطایی رخ دهد، در این خط متوقف خواهد شد.
model.fit(X_np, y_np)

print("\n--- ✅ آموزش مدل با موفقیت به پایان رسید ---")
