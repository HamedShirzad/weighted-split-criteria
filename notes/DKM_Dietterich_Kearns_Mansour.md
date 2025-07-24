# Dietterich-Kearns-Mansour Criterion (DKM)

## اطلاعات کلی
- **منبع اصلی:** Applying the Weak Learning Framework to Understand and Improve C4.5
- **سال معرفی:** 1996
- **نویسندگان:** Tom Dietterich (Oregon State University), Michael Kearns (AT&T Research), Yishay Mansour (Tel Aviv University)
- **مجله:** IEEE Conference on Machine Learning (ICML 1996)
- **انگیزه:** بهبود concavity معیار تقسیم نسبت به Information Gain
- **مزیت کلیدی:** تولید درخت‌های کوچک‌تر با دقت بهتر یا مشابه

## هدف و کاربرد

معیار Dietterich-Kearns-Mansour (همچنین به نام G-criterion شناخته می‌شود) به عنوان **جایگزین بهبود یافته Information Gain** در الگوریتم C4.5 معرفی شده است:

1. **بهبود concavity function** - ایجاد تابع concave بهتر برای اندازه‌گیری impurity
2. **تولید درخت‌های کوچک‌تر** - کاهش اندازه درخت بدون کاهش دقت
3. **بهبود عملکرد تعمیم** - کاهش overfitting در درخت‌های تصمیم
4. **سادگی محاسباتی** - جایگزینی لگاریتم با عملیات جذر

این معیار بر پایه **Weak Learning Framework** طراحی شده و برای **مسائل طبقه‌بندی باینری** بهینه شده است، اما قابل تعمیم به مسائل چندکلاسه نیز هست.

## فرمول ریاضی

### فرمول اصلی G-Criterion
$$G(q) = \sqrt{q(1-q)}$$

### فرمول کلی برای درخت
$$G(T) = \sum_{\ell \in leaves(T)} w(\ell) \cdot G(q(\ell))$$

### فرمول تقسیم در گره
برای تقسیم یک گره به دو شاخه چپ و راست:

$$\text{DKM Score} = \frac{n_L}{n} \cdot G(q_L) + \frac{n_R}{n} \cdot G(q_R)$$

### تعریف متغیرها

| نماد | تعریف |
|------|--------|
| **$G(q)$** | مقدار G-criterion برای نسبت q |
| **$q$** | نسبت نمونه‌های مثبت (کلاس 1) در گره |
| **$q_L$** | نسبت نمونه‌های مثبت در شاخه چپ |
| **$q_R$** | نسبت نمونه‌های مثبت در شاخه راست |
| **$n_L$** | تعداد نمونه‌ها در شاخه چپ |
| **$n_R$** | تعداد نمونه‌ها در شاخه راست |
| **$n$** | تعداد کل نمونه‌ها ($n = n_L + n_R$) |
| **$w(\ell)$** | وزن برگ ℓ (نسبت نمونه‌هایی که به آن می‌رسند) |

### مقایسه با Information Gain
$$H(q) = -q \log_2 q - (1-q) \log_2 (1-q)$$
$$G(q) = \sqrt{q(1-q)}$$

## مثال محاسبه

فرض کنید یک گره 100 نمونه را به دو شاخه تقسیم می‌کند:

### داده‌های نمونه:

| شاخه | کلاس 0 | کلاس 1 | مجموع | q (نسبت کلاس 1) |
|------|---------|---------|-------|------------------|
| چپ | 30 | 10 | 40 | 0.25 |
| راست | 20 | 40 | 60 | 0.67 |
| **کل** | 50 | 50 | 100 | 0.50 |

### گام‌های محاسبه:

**گام ۱:** محاسبه G برای هر شاخه
- $G(q_L) = G(0.25) = \sqrt{0.25 \times 0.75} = \sqrt{0.1875} = 0.433$
- $G(q_R) = G(0.67) = \sqrt{0.67 \times 0.33} = \sqrt{0.221} = 0.470$

**گام ۲:** محاسبه وزن‌های شاخه‌ها
- وزن شاخه چپ: $w_L = \frac{40}{100} = 0.4$
- وزن شاخه راست: $w_R = \frac{60}{100} = 0.6$

**گام ۳:** محاسبه DKM Score نهایی
$$\text{DKM Score} = 0.4 \times 0.433 + 0.6 \times 0.470 = 0.173 + 0.282 = 0.455$$

**گام ۴:** مقایسه با Information Gain
- $H(0.25) = -0.25 \log_2(0.25) - 0.75 \log_2(0.75) = 0.811$
- $H(0.67) = -0.67 \log_2(0.67) - 0.33 \log_2(0.33) = 0.918$
- $\text{Weighted H} = 0.4 \times 0.811 + 0.6 \times 0.918 = 0.875$

نتیجه: DKM = 0.455، Information Gain weighted = 0.875

## ویژگی‌های فنی

- **پیچیدگی محاسباتی:** $O(1)$ برای هر گره (فقط عملیات جذر)
- **محدوده مقادیر:** $0 \leq G(q) \leq 0.5$ (حداکثر در $q = 0.5$)
- **concavity بهتر:** نسبت به Information Gain خصوصاً برای مقادیر کوچک q
- **پایداری عددی:** بهتر از لگاریتم (بدون مشکل تقسیم بر صفر)

### خصوصیات ریاضی:
- **مشتق اول:** $G'(q) = \frac{1-2q}{2\sqrt{q(1-q)}}$
- **مشتق دوم:** $G''(q) = -\frac{1}{4[q(1-q)]^{3/2}} < 0$ (تابع concave)
- **نقطه بیشینه:** $q = 0.5$ که در آن $G(0.5) = 0.5$

### تفسیر مقادیر:
- **$G(q) = 0$**: گره خالص (فقط یک کلاس)
- **$G(q) = 0.5$**: حداکثر عدم اطمینان (تعادل کامل کلاس‌ها)
- **$G(q)$ بالا**: تقسیم مفیدتر برای کاهش impurity

## مقایسه با سایر معیارها

| جنبه | DKM G(q) | Information Gain H(q) | Gini Impurity | Twoing |
|------|----------|----------------------|---------------|--------|
| **فرمول** | $\sqrt{q(1-q)}$ | $-q \log_2 q - (1-q) \log_2 (1-q)$ | $1 - \sum p_i^2$ | پیچیده |
| **محاسبه** | بسیار سریع | متوسط | سریع | متوسط |
| **concavity** | بهتر | استاندارد | خوب | متغیر |
| **اندازه درخت** | کوچک‌تر | استاندارد | استاندارد | متوسط |
| **دقت** | مشابه یا بهتر | مرجع | مشابه | متغیر |
| **تفسیر** | ساده | معمولی | ساده | پیچیده |

## مزایا

### 1. بهبود concavity نسبت به Information Gain
- **تقریب بهتر:** G(q) دارای concavity بهتری برای مقادیر کوچک q است
- **پیشرفت سریع‌تر:** در مراحل اولیه ساخت درخت بهبود بیشتری حاصل می‌شود
- **انتخاب بهتر ویژگی:** ویژگی‌هایی که باعث تغییرات قابل توجه در q می‌شوند ترجیح داده می‌شوند

### 2. سادگی و سرعت محاسباتی
- **عدم نیاز به لگاریتم:** فقط عملیات جذر که سریع‌تر است
- **پایداری عددی:** مشکل $\log(0)$ وجود ندارد
- **قابلیت بهینه‌سازی:** محاسبات قابل vectorization هستند

### 3. نتایج تجربی مثبت
- **عملکرد بهتر:** در 8 از 9 مجموعه داده UCI
- **درخت‌های کوچک‌تر:** در تمام موارد آزمایش شده
- **sign test معنادار:** p < 0.05 برای برتری نسبت به Information Gain

### 4. مبنای نظری محکم
- **ارتباط با Weak Learning:** پایه‌گذاری بر اساس نظریه یادگیری ضعیف
- **اثبات ریاضی:** concavity بهتر از لحاظ تئوری اثبات شده
- **انطباق با PAC learning:** سازگار با چارچوب یادگیری محاسباتی

## محدودیت‌ها

### 1. محدودیت در مسائل چندکلاسه
- **طراحی برای باینری:** فرمول اصلی برای دو کلاس است
- **نیاز به تعمیم:** برای چندکلاسه باید روش‌های ترکیبی استفاده کرد
- **پیچیدگی اضافی:** در حالت چندکلاسه از سادگی اصلی دور می‌شود

### 2. عدم شناخت گسترده
- **جدید بودن:** نسبت به معیارهای کلاسیک کمتر شناخته شده
- **مراجع محدود:** کمتر در ادبیات تحقیقاتی استفاده شده
- **عدم پشتیبانی:** در اکثر کتابخانه‌های آماده موجود نیست

### 3. وابستگی به نوع داده
- **حساسیت به توزیع:** عملکرد بهتر در توزیع‌های خاص
- **نیاز به تنظیم:** ممکن است در برخی مسائل نیاز به fine-tuning داشته باشد
- **عملکرد متغیر:** در برخی datasets ممکن است بهبود قابل توجهی نداشته باشد

### 4. تفسیر متفاوت
- **واحد متفاوت:** مقادیر G با H قابل مقایسه مستقیم نیستند
- **عدم تطابق با entropy:** مفهوم آن با entropy کلاسیک متفاوت است
- **نیاز به آموزش:** کاربران باید با مفهوم جدید آشنا شوند

## نکات پیاده‌سازی

### مراقبت‌های ضروری:
- **مدیریت q=0 یا q=1:** در این موارد $G(q) = 0$ برگردانید
- **تعمیم به چندکلاسه:** از weighted average یا one-vs-rest استفاده کنید
- **محاسبه دقیق جذر:** از کتابخانه‌های عددی بهینه استفاده کنید
- **مقایسه منصفانه:** نتایج را با Information Gain مقایسه کنید

### بهینه‌سازی:
- استفاده از NumPy vectorized operations
- کش کردن محاسبات q برای گره‌های مختلف
- محاسبه parallel برای ویژگی‌های مختلف
- استفاده از fast square root implementations

### الگوی پیاده‌سازی:

برای مسئله باینری

q_left = sum(y_left) / len(y_left)
q_right = sum(y_right) / len(y_right)

g_left = sqrt(q_left * (1 - q_left))
g_right = sqrt(q_right * (1 - q_right))
ترکیب وزنی

dkm_score = (len(y_left)/total) * g_left + (len(y_right)/total) * g_right


## کد شبه

def dkm_criterion(y_left, y_right):
"""
محاسبه معیار Dietterich-Kearns-Mansour (G-criterion)
Parameters:
y_left: آرایه برچسب‌های کلاس برای شاخه چپ (0 یا 1)
y_right: آرایه برچسب‌های کلاس برای شاخه راست (0 یا 1)

Returns:
DKM_score: امتیاز وزنی G-criterion
"""

# بررسی شرایط اولیه
if len(y_left) == 0 or len(y_right) == 0:
    return 0.0

# محاسبه نسبت کلاس مثبت در هر شاخه
q_left = sum(y_left) / len(y_left)
q_right = sum(y_right) / len(y_right)

# محاسبه G(q) برای هر شاخه
g_left = calculate_g_value(q_left)
g_right = calculate_g_value(q_right)

# محاسبه وزن‌ها
n_left = len(y_left)
n_right = len(y_right)
n_total = n_left + n_right

weight_left = n_left / n_total
weight_right = n_right / n_total

# ترکیب وزنی
dkm_score = weight_left * g_left + weight_right * g_right

return dkm_score

def dkm_multiclass(y_left, y_right):
"""تعمیم DKM به مسائل چندکلاسه"""

# روش 1: One-vs-Rest
classes = unique(concatenate([y_left, y_right]))
total_score = 0.0

for target_class in classes:
    # تبدیل به مسئله باینری
    binary_left = [1 if label == target_class else 0 for label in y_left]
    binary_right = [1 if label == target_class else 0 for label in y_right]
    
    # محاسبه DKM برای این کلاس
    class_score = dkm_criterion(binary_left, binary_right)
    total_score += class_score

# میانگین امتیازها
return total_score / len(classes)



```

## منابع
- Dietterich, T.G., Kearns, M., Mansour, Y. (1996). Applying the Weak Learning Framework to Understand and Improve C4.5. Proceedings of the International Conference on Machine Learning (ICML)
- Quinlan, J.R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers
- Freund, Y., Schapire, R.E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences
- مقالات مقایسه‌ای معیارهای تقسیم در درخت‌های تصمیم
