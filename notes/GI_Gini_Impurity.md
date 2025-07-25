# Gini Impurity (GI)

## اطلاعات کلی
- **منبع اصلی:** Classification and Regression Trees (CART) - Breiman et al.
- **سال معرفی:** 1984
- **نویسندگان:** Leo Breiman, Jerome Friedman, Richard Olshen, Charles Stone
- **مجله:** Chapman & Hall/CRC
- **انگیزه:** معیار ساده و کارآمد برای اندازه‌گیری خلوص گره‌ها در درخت‌های تصمیم
- **مزیت کلیدی:** سادگی محاسباتی و مقاومت در برابر عدم تعادل کلاس‌ها

## هدف و کاربرد

معیار Gini Impurity به عنوان **معیار پایه‌ای و استاندارد** برای اندازه‌گیری ناخلوصی گره‌ها در درخت‌های تصمیم طراحی شده است:

1. **اندازه‌گیری عدم قطعیت** - محاسبه میزان مخلوط بودن کلاس‌ها در هر گره
2. **سادگی محاسباتی** - فرمول ساده بدون نیاز به محاسبه لگاریتم
3. **benchmark استاندارد** - مرجع اصلی برای مقایسه سایر معیارها
4. **مقاومت نسبی** - کمتر متأثر از عدم تعادل کلاس‌ها نسبت به entropy

این معیار در **الگوریتم CART** و **scikit-learn** به عنوان معیار پیش‌فرض استفاده می‌شود و پرکاربردترین معیار تقسیم در یادگیری ماشین محسوب می‌شود.

## فرمول ریاضی

### فرمول اصلی Gini Impurity
$$\text{Gini}(t) = 1 - \sum_{i=1}^{c} p_i^2$$

### فرمول Gini Gain برای تقسیم
$$\text{Gini Gain} = \text{Gini}(parent) - \sum_{j} \frac{|t_j|}{|t|} \text{Gini}(t_j)$$

### فرمول weighted Gini برای دو شاخه
$$\text{Weighted Gini} = \frac{n_L}{n} \times \text{Gini}(S_L) + \frac{n_R}{n} \times \text{Gini}(S_R)$$

### تعریف متغیرها

| نماد | تعریف |
|------|--------|
| **$\text{Gini}(t)$** | مقدار Gini Impurity برای گره $t$ |
| **$p_i$** | نسبت نمونه‌های کلاس $i$ در گره |
| **$c$** | تعداد کلاس‌های موجود |
| **$n_L, n_R$** | تعداد نمونه‌ها در شاخه‌های چپ و راست |
| **$n$** | تعداد کل نمونه‌ها در گره والد |
| **$S_L, S_R$** | مجموعه نمونه‌ها در شاخه‌های چپ و راست |

### محدوده مقادیر
- **حداقل:** $\text{Gini} = 0$ (خلوص کامل - همه نمونه‌ها از یک کلاس)
- **حداکثر:** $\text{Gini} = 1 - \frac{1}{c}$ (حداکثر نابرابری برای $c$ کلاس)
- **برای دو کلاس:** $0 \leq \text{Gini} \leq 0.5$

## مثال محاسبه

فرض کنید یک گره 100 نمونه را به دو شاخه تقسیم می‌کند:

### داده‌های نمونه:

| شاخه | کلاس A | کلاس B | کلاس C | مجموع |
|------|---------|---------|---------|-------|
| چپ | 30 | 20 | 10 | 60 |
| راست | 10 | 15 | 15 | 40 |
| **کل** | 40 | 35 | 25 | 100 |

### گام‌های محاسبه:

**گام ۱:** محاسبه Gini والد
$$p_A = \frac{40}{100} = 0.4, \quad p_B = \frac{35}{100} = 0.35, \quad p_C = \frac{25}{100} = 0.25$$

$$\text{Gini}(parent) = 1 - (0.4^2 + 0.35^2 + 0.25^2) = 1 - (0.16 + 0.1225 + 0.0625) = 0.655$$

**گام ۲:** محاسبه Gini شاخه چپ
$$p_{A,L} = \frac{30}{60} = 0.5, \quad p_{B,L} = \frac{20}{60} = 0.333, \quad p_{C,L} = \frac{10}{60} = 0.167$$

$$\text{Gini}(left) = 1 - (0.5^2 + 0.333^2 + 0.167^2) = 1 - (0.25 + 0.111 + 0.028) = 0.611$$

**گام ۳:** محاسبه Gini شاخه راست
$$p_{A,R} = \frac{10}{40} = 0.25, \quad p_{B,R} = \frac{15}{40} = 0.375, \quad p_{C,R} = \frac{15}{40} = 0.375$$

$$\text{Gini}(right) = 1 - (0.25^2 + 0.375^2 + 0.375^2) = 1 - (0.0625 + 0.141 + 0.141) = 0.656$$

**گام ۴:** محاسبه Weighted Gini
$$\text{Weighted Gini} = \frac{60}{100} \times 0.611 + \frac{40}{100} \times 0.656 = 0.6 \times 0.611 + 0.4 \times 0.656 = 0.629$$

**گام ۵:** محاسبه Gini Gain
$$\text{Gini Gain} = 0.655 - 0.629 = 0.026$$

## ویژگی‌های فنی

- **پیچیدگی محاسباتی:** $O(c)$ جایی که $c$ تعداد کلاس‌هاست
- **حافظه مورد نیاز:** $O(c)$ برای نگهداری شمارنده کلاس‌ها
- **پایداری عددی:** بالا - فقط عملیات جمع و ضرب
- **قابلیت vectorization:** عالی با NumPy

### خصوصیات ریاضی:
- **تابع محدب:** $\text{Gini}(p)$ تابع محدب نسبت به $p$ است
- **متقارن:** مستقل از ترتیب کلاس‌ها
- **additivity:** قابل تجمیع برای گره‌های مختلف
- **sensitivity:** حساس به تغییرات در توزیع کلاس‌ها

### تفسیر مقادیر:
- **Gini = 0:** گره خالص (همه نمونه‌ها از یک کلاس)
- **Gini بالا:** گره مخلوط (نمونه‌های متنوع از کلاس‌های مختلف)
- **Gini Gain بالا:** تقسیم مفید (کاهش قابل توجه ناخلوصی)

## مقایسه با سایر معیارها

| جنبه | Gini Impurity | Information Gain | Chi-Squared | Bhattacharyya |
|------|---------------|------------------|-------------|---------------|
| **فرمول** | $1 - \sum p_i^2$ | $-\sum p_i \log p_i$ | $\sum \frac{(O-E)^2}{E}$ | $-\ln(\sum \sqrt{p_i q_i})$ |
| **پیچیدگی محاسبه** | $O(c)$ | $O(c \log c)$ | $O(c)$ | $O(c)$ |
| **پایداری عددی** | بالا | متوسط | متوسط | متوسط |
| **تفسیرپذیری** | بالا | بالا | متوسط | پایین |
| **استفاده صنعتی** | بسیار زیاد | زیاد | متوسط | کم |
| **کاربرد پیش‌فرض** | CART, sklearn | ID3, C4.5 | آماری | research |

## مزایا

### 1. سادگی محاسباتی
- **عدم نیاز به لگاریتم:** محاسبه سریع‌تر نسبت به entropy
- **عملیات ساده:** فقط جمع و ضرب
- **پایداری عددی:** عدم مشکل با مقادیر صفر یا منفی
- **قابلیت بهینه‌سازی:** vectorization آسان با NumPy

### 2. تفسیرپذیری بالا
- **مفهوم شهودی:** احتمال طبقه‌بندی غلط تصادفی
- **محدوده مشخص:** بین 0 و 0.5 (برای دو کلاس)
- **تناسب با هدف:** مستقیماً مرتبط با دقت طبقه‌بندی
- **مقایسه آسان:** امکان مقایسه مستقیم بین گره‌ها

### 3. مقاومت نسبی در برابر عدم تعادل
- **symmetric:** رفتار یکسان با کلاس‌های اقلیت و اکثریت
- **smooth behavior:** تغییرات تدریجی با تغییر توزیع
- **robust:** کمتر متأثر از نقاط پرت نسبت به entropy
- **balanced split preference:** ترجیح تقسیم‌های متعادل

### 4. استاندارد صنعت
- **پیش‌فرض scikit-learn:** معیار استاندارد در کتابخانه محبوب
- **CART algorithm:** اصلی‌ترین معیار در CART
- **benchmark مقبول:** مرجع مقایسه در تحقیقات
- **پشتیبانی گسترده:** موجود در همه ابزارهای ML

## محدودیت‌ها

### 1. تعصب نسبت به تقسیم‌های متعادل
- **balanced split preference:** گاهی تقسیم‌های نامتعادل بهتر هستند
- **local optimization:** ممکن است optimum سراسری را از دست دهد
- **depth preference:** ممکن است درخت‌های عمیق‌تر تولید کند
- **feature bias:** کمتر نسبت به Information Gain اما همچنان موجود

### 2. حساسیت به توزیع کلاس‌ها
- **class imbalance:** در مجموعه داده‌های نامتعادل مشکل‌ساز است
- **minority class:** ممکن است کلاس‌های اقلیت را نادیده بگیرد
- **threshold sensitivity:** حساس به انتخاب threshold در ویژگی‌های پیوسته
- **noise effect:** متأثر از نویز در برچسب‌ها

### 3. عدم در نظر گیری شدت خطا
- **equal cost assumption:** همه خطاهای طبقه‌بندی را یکسان در نظر می‌گیرد
- **no cost matrix:** قابلیت وزن‌دهی به کلاس‌های مختلف ندارد
- **binary focus:** بهینه‌سازی شده برای مسائل دوکلاسه
- **loss function mismatch:** ممکن است با هدف نهایی مطابقت نداشته باشد

### 4. محدودیت نظری
- **greedy approach:** تصمیم‌گیری محلی بدون در نظر گیری آینده
- **no global optimization:** عدم تضمین درخت بهینه سراسری
- **overfitting tendency:** در داده‌های پیچیده به overfitting مستعد
- **interpretability loss:** در درخت‌های بزرگ قابلیت تفسیر کاهش می‌یابد

## نکات پیاده‌سازی

### مراقبت‌های ضروری:
- **مدیریت تقسیم بر صفر:** بررسی $n > 0$ قبل از محاسبه نسبت‌ها
- **دقت عددی:** استفاده از float64 برای محاسبات دقیق
- **حافظه:** بهینه‌سازی برای مجموعه داده‌های بزرگ
- **کلاس‌های نادر:** مدیریت کلاس‌هایی با نمونه کم

### بهینه‌سازی:
- استفاده از NumPy vectorized operations برای سرعت بالا
- کش کردن محاسبات برای گره‌هایی که چندین بار ارزیابی می‌شوند
- محاسبه incremental برای ویژگی‌های مرتب‌شده
- parallel processing برای ارزیابی ویژگی‌های متعدد

### الگوی پیاده‌سازی:
محاسبه توزیع کلاس‌ها

class_counts = np.bincount(y)
class_probs = class_counts / len(y)
محاسبه Gini Impurity

gini = 1.0 - np.sum(class_probs ** 2)
محاسبه Weighted Gini برای تقسیم

n_left, n_right = len(y_left), len(y_right)
n_total = n_left + n_right

weighted_gini = (n_left/n_total) * gini_left + (n_right/n_total) * gini_right


## کد شبه

def gini_impurity(y):
"""
محاسبه Gini Impurity برای یک مجموعه برچسب
Parameters:
y: آرایه برچسب‌های کلاس

Returns:
gini: مقدار Gini Impurity (0 تا 1-1/c)
"""

if len(y) == 0:
    return 0.0

# شمارش فراوانی هر کلاس
class_counts = {}
for label in y:
    class_counts[label] = class_counts.get(label, 0) + 1

# محاسبه احتمالات
n_samples = len(y)
gini = 1.0

for count in class_counts.values():
    prob = count / n_samples
    gini -= prob * prob

return gini
def gini_gain(y_left, y_right, y_parent):
"""
محاسبه Gini Gain برای یک تقسیم
Parameters:
y_left: برچسب‌های شاخه چپ
y_right: برچسب‌های شاخه راست
y_parent: برچسب‌های گره والد

Returns:
gain: مقدار Gini Gain (مثبت = تقسیم مفید)
"""

n_left = len(y_left)
n_right = len(y_right)
n_total = n_left + n_right

# محاسبه Gini والد
gini_parent = gini_impurity(y_parent)

# محاسبه Weighted Gini فرزندان
if n_total == 0:
    return 0.0

gini_left = gini_impurity(y_left)
gini_right = gini_impurity(y_right)

weighted_gini = (n_left/n_total) * gini_left + (n_right/n_total) * gini_right

# محاسبه Gain
gain = gini_parent - weighted_gini

return gain
def optimized_gini_impurity(y):
"""نسخه بهینه‌شده با NumPy"""
if len(y) == 0:
    return 0.0

# استفاده از NumPy برای سرعت
unique_classes, counts = np.unique(y, return_counts=True)
probs = counts / len(y)

gini = 1.0 - np.sum(probs ** 2)

return gini
def weighted_gini_impurity(y_left, y_right):
"""محاسبه مستقیم Weighted Gini برای تقسیم"""
n_left = len(y_left)
n_right = len(y_right)
n_total = n_left + n_right

if n_total == 0:
    return 0.0

gini_left = optimized_gini_impurity(y_left)
gini_right = optimized_gini_impurity(y_right)

weighted_gini = (n_left/n_total) * gini_left + (n_right/n_total) * gini_right

return weighted_gini
```
## منابع
- Breiman, L., Friedman, J., Stone, C.J., Olshen, R.A. (1984). Classification and Regression Trees. Chapman & Hall/CRC
- Gini, C. (1912). Variabilità e mutabilità. Reprinted in Memorie di metodologia statistica
- Quinlan, J.R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers
- Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical Learning. Springer
- scikit-learn documentation: Decision Tree Classifier
