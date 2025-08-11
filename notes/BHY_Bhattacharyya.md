# Bhattacharyya Distance (BD)

## اطلاعات کلی
- **منبع اصلی:** A measure of divergence between two statistical populations defined by their probability distributions
- **سال معرفی:** 1943
- **نویسنده:** Anil Kumar Bhattacharyya
- **مجله:** Bulletin of the Calcutta Mathematical Society, Vol. 35
- **انگیزه:** اندازه‌گیری فاصله بین توزیع‌های احتمال برای تشخیص الگو و طبقه‌بندی
- **مزیت کلیدی:** upper bound برای خطای طبقه‌بندی و مقاومت در برابر نویز

## هدف و کاربرد

معیار Bhattacharyya Distance به عنوان **معیار فاصله‌ای بین توزیع‌های احتمال** برای تقسیم گره‌ها در درخت‌های تصمیم طراحی شده است:

1. **اندازه‌گیری تفاوت توزیع‌ها** - محاسبه دقیق فاصله بین توزیع احتمال کلاس‌ها در هر شاخه
2. **کنترل خطای طبقه‌بندی** - ارائه upper bound برای خطای Bayes
3. **مقاومت در برابر نویز** - کمتر متأثر از outlier و نقاط پرت نسبت به معیارهای آماری
4. **تفسیرپذیری هندسی** - تعبیر هندسی روشن در فضای احتمال

این معیار مخصوصاً برای **مسائل تشخیص الگو**، **پردازش تصویر**، و **طبقه‌بندی با عدم قطعیت بالا** مناسب است.

## فرمول ریاضی

### فرمول اصلی Bhattacharyya Coefficient
$$BC(P,Q) = \sum_{i=1}^{n} \sqrt{p_i \cdot q_i}$$

### فرمول Bhattacharyya Distance
$$BD(P,Q) = -\ln(BC(P,Q))$$

### فرمول برای تقسیم درخت
برای تقسیم یک گره به دو شاخه چپ و راست:

$$BD_{split} = BD(P_{left}, P_{right})$$

جایی که $P_{left}$ و $P_{right}$ توزیع احتمال کلاس‌ها در شاخه‌های چپ و راست هستند.

### تعریف متغیرها

| نماد | تعریف |
|------|--------|
| **$BC$** | ضریب Bhattacharyya (مقدار تشابه) |
| **$BD$** | فاصله Bhattacharyya |
| **$P, Q$** | دو توزیع احتمال |
| **$p_i, q_i$** | احتمال کلاس $i$ در توزیع‌های P و Q |
| **$n$** | تعداد کلاس‌های موجود |

### خصوصیات ریاضی
- **محدوده BC:** $0 \leq BC \leq 1$
- **محدوده BD:** $0 \leq BD \leq +\infty$
- **تفسیر BC:** BC=1 یعنی توزیع‌های یکسان، BC=0 یعنی توزیع‌های کاملاً متفاوت
- **تفسیر BD:** BD=0 یعنی توزیع‌های یکسان، BD→∞ یعنی توزیع‌های کاملاً متفاوت

## مثال محاسبه

فرض کنید یک گره را به دو شاخه تقسیم می‌کنیم:

### داده‌های نمونه:

| شاخه | کلاس A | کلاس B | کلاس C | مجموع |
|------|---------|---------|---------|-------|
| چپ | 40 | 20 | 10 | 70 |
| راست | 10 | 15 | 25 | 50 |

### گام‌های محاسبه:

**گام ۱:** محاسبه توزیع احتمال هر شاخه
- شاخه چپ: $P = [40/70, 20/70, 10/70] = [0.571, 0.286, 0.143]$
- شاخه راست: $Q = [10/50, 15/50, 25/50] = [0.200, 0.300, 0.500]$

**گام ۲:** محاسبه Bhattacharyya Coefficient
$$BC = \sqrt{0.571 \times 0.200} + \sqrt{0.286 \times 0.300} + \sqrt{0.143 \times 0.500}$$

$$BC = \sqrt{0.114} + \sqrt{0.086} + \sqrt{0.072}$$

$$BC = 0.338 + 0.293 + 0.268 = 0.899$$

**گام ۳:** محاسبه Bhattacharyya Distance
$$BD = -\ln(0.899) = -(-0.107) = 0.107$$

**گام ۴:** تفسیر نتیجه
- BC = 0.899 نشان‌دهنده تشابه نسبتاً بالای دو توزیع
- BD = 0.107 نشان‌دهنده فاصله کم بین دو شاخه (تقسیم ضعیف)

## ویژگی‌های فنی

- **پیچیدگی محاسباتی:** $O(k)$ جایی که k تعداد کلاس‌هاست
- **محدوده مقادیر:** $0 \leq BD \leq +\infty$
- **symmetry:** $BD(P,Q) = BD(Q,P)$
- **triangle inequality:** در حالت کلی برقرار نیست
- **پایداری عددی:** نیاز به مدیریت دقیق لگاریتم صفر

### خصوصیات نظری:
- **Upper bound برای خطا:** $P_{error} \leq \sqrt{P_1 P_2} e^{-BD}$
- **ارتباط با Chernoff bound:** BD حد بالای بهینه برای خطای طبقه‌بندی
- **Hellinger distance:** $H^2 = 2(1 - BC)$، رابطه مستقیم با BD

### تفسیر مقادیر:
- **BD = 0**: توزیع‌های یکسان (تقسیم بی‌فایده)
- **BD کوچک**: توزیع‌های مشابه (تقسیم ضعیف)
- **BD بزرگ**: توزیع‌های متفاوت (تقسیم قوی)

## مقایسه با سایر معیارها

| جنبه | Bhattacharyya | Hellinger | Chi-Squared | Information Gain |
|------|---------------|-----------|-------------|------------------|
| **مبنای نظری** | فاصله احتمالی | فاصله احتمالی | آزمون استقلال | نظریه اطلاعات |
| **محدوده مقادیر** | $[0, +\infty)$ | $[0, 1]$ | $[0, +\infty)$ | متغیر |
| **تفسیرپذیری** | upper bound خطا | فاصله هندسی | معناداری آماری | کاهش آنتروپی |
| **مقاومت به نویز** | بالا | بالا | متوسط | پایین |
| **پیچیدگی محاسبه** | $O(k)$ | $O(k)$ | $O(k)$ | $O(k \log k)$ |
| **کاربرد عملی** | تشخیص الگو | چندکلاسه نامتعادل | آزمون آماری | درخت‌های کلاسیک |

## مزایا

### 1. پایه نظری محکم
- **ارتباط با optimal classification:** ارائه upper bound برای خطای Bayes
- **تفسیر هندسی:** فاصله در فضای احتمال با معنای روشن
- **پیوند با سایر معیارها:** ارتباط مستقیم با Hellinger و Chernoff distances

### 2. مقاومت در برابر نویز
- **تأثیر کمتر outlier:** استفاده از جذر در محاسبه BC
- **پایداری در برابر تغییرات کوچک:** تغییرات محلی کمتر بر نتیجه تأثیر می‌گذارند
- **robust estimation:** مناسب برای داده‌های پرنویز

### 3. کاربرد وسیع
- **computer vision:** histogram comparison و feature matching
- **signal processing:** تشخیص الگوهای آماری
- **machine learning:** feature selection و clustering
- **medical diagnosis:** مقایسه پروفایل‌های بیولوژیکی

### 4. ویژگی‌های محاسباتی مطلوب
- **محاسبه سریع:** پیچیدگی خطی نسبت به تعداد کلاس‌ها
- **قابلیت vectorization:** مناسب برای پیاده‌سازی موثر با NumPy
- **عدم نیاز به پارامتر:** بدون نیاز به تنظیم hyperparameter

## محدودیت‌ها

### 1. حساسیت به احتمالات صفر
- **مشکل محاسباتی:** وقتی $p_i = 0$ یا $q_i = 0$، ترم مربوطه صفر می‌شود
- **نیاز به smoothing:** استفاده از Laplace smoothing یا روش‌های مشابه
- **تأثیر بر دقت:** smoothing ممکن است اطلاعات را تحریف کند

### 2. عدم تبعیت از triangle inequality
- **محدودیت متریک:** BD یک pseudo-metric است، نه metric کامل
- **مشکل در clustering:** برخی الگوریتم‌های clustering نیاز به triangle inequality دارند
- **تفسیر هندسی محدود:** نمی‌توان آن را کاملاً به عنوان فاصله هندسی در نظر گرفت

### 3. وابستگی به توزیع کلاس‌ها
- **تأثیر class imbalance:** در مجموعه داده‌های نامتعادل ممکن است مشکل‌ساز باشد
- **حساسیت به اندازه نمونه:** در نمونه‌های کوچک تخمین احتمالات نادقیق است
- **نیاز به نمونه کافی:** برای تخمین موثق احتمالات

### 4. تفسیر پیچیده نسبت به معیارهای ساده
- **نیاز به دانش آماری:** درک مفهوم فاصله احتمالی
- **مقایسه با بصیرت انسانی:** کمتر قابل تطبیق با درک شهودی تقسیم‌ها
- **ارتباط غیرمستقیم با هدف:** ارتباط با accuracy از طریق upper bound

## نکات پیاده‌سازی

### مراقبت‌های ضروری:
- **مدیریت احتمالات صفر:** استفاده از Laplace smoothing یا $\epsilon$-smoothing
- **کنترل overflow/underflow:** محاسبه در log space برای پایداری عددی
- **نرمال‌سازی احتمالات:** اطمینان از $\sum p_i = 1$ و $\sum q_i = 1$
- **مدیریت BC=0:** در صورت BC=0، تنظیم BD به مقدار بزرگ مناسب

### بهینه‌سازی:
- استفاده از NumPy vectorized operations برای محاسبه سریع
- کش کردن محاسبات توزیع احتمال برای استفاده مجدد
- محاسبه در log space: $\ln(BC) = \frac{1}{2}\sum \ln(p_i q_i)$
- تنظیم threshold برای احتمالات خیلی کوچک

### الگوی پیاده‌سازی:
# Bhattacharyya Distance (BD)

## اطلاعات کلی
- **منبع اصلی:** A measure of divergence between two statistical populations defined by their probability distributions
- **سال معرفی:** 1943
- **نویسنده:** Anil Kumar Bhattacharyya
- **مجله:** Bulletin of the Calcutta Mathematical Society, Vol. 35
- **انگیزه:** اندازه‌گیری فاصله بین توزیع‌های احتمال برای تشخیص الگو و طبقه‌بندی
- **مزیت کلیدی:** upper bound برای خطای طبقه‌بندی و مقاومت در برابر نویز

## هدف و کاربرد

معیار Bhattacharyya Distance به عنوان **معیار فاصله‌ای بین توزیع‌های احتمال** برای تقسیم گره‌ها در درخت‌های تصمیم طراحی شده است:

1. **اندازه‌گیری تفاوت توزیع‌ها** - محاسبه دقیق فاصله بین توزیع احتمال کلاس‌ها در هر شاخه
2. **کنترل خطای طبقه‌بندی** - ارائه upper bound برای خطای Bayes
3. **مقاومت در برابر نویز** - کمتر متأثر از outlier و نقاط پرت نسبت به معیارهای آماری
4. **تفسیرپذیری هندسی** - تعبیر هندسی روشن در فضای احتمال

این معیار مخصوصاً برای **مسائل تشخیص الگو**، **پردازش تصویر**، و **طبقه‌بندی با عدم قطعیت بالا** مناسب است.

## فرمول ریاضی

### فرمول اصلی Bhattacharyya Coefficient
$$BC(P,Q) = \sum_{i=1}^{n} \sqrt{p_i \cdot q_i}$$

### فرمول Bhattacharyya Distance
$$BD(P,Q) = -\ln(BC(P,Q))$$

### فرمول برای تقسیم درخت
برای تقسیم یک گره به دو شاخه چپ و راست:

$$BD_{split} = BD(P_{left}, P_{right})$$

جایی که $P_{left}$ و $P_{right}$ توزیع احتمال کلاس‌ها در شاخه‌های چپ و راست هستند.

### تعریف متغیرها

| نماد | تعریف |
|------|--------|
| **$BC$** | ضریب Bhattacharyya (مقدار تشابه) |
| **$BD$** | فاصله Bhattacharyya |
| **$P, Q$** | دو توزیع احتمال |
| **$p_i, q_i$** | احتمال کلاس $i$ در توزیع‌های P و Q |
| **$n$** | تعداد کلاس‌های موجود |

### خصوصیات ریاضی
- **محدوده BC:** $0 \leq BC \leq 1$
- **محدوده BD:** $0 \leq BD \leq +\infty$
- **تفسیر BC:** BC=1 یعنی توزیع‌های یکسان، BC=0 یعنی توزیع‌های کاملاً متفاوت
- **تفسیر BD:** BD=0 یعنی توزیع‌های یکسان، BD→∞ یعنی توزیع‌های کاملاً متفاوت

## مثال محاسبه

فرض کنید یک گره را به دو شاخه تقسیم می‌کنیم:

### داده‌های نمونه:

| شاخه | کلاس A | کلاس B | کلاس C | مجموع |
|------|---------|---------|---------|-------|
| چپ | 40 | 20 | 10 | 70 |
| راست | 10 | 15 | 25 | 50 |

### گام‌های محاسبه:

**گام ۱:** محاسبه توزیع احتمال هر شاخه
- شاخه چپ: $P = [40/70, 20/70, 10/70] = [0.571, 0.286, 0.143]$
- شاخه راست: $Q = [10/50, 15/50, 25/50] = [0.200, 0.300, 0.500]$

**گام ۲:** محاسبه Bhattacharyya Coefficient
$$BC = \sqrt{0.571 \times 0.200} + \sqrt{0.286 \times 0.300} + \sqrt{0.143 \times 0.500}$$

$$BC = \sqrt{0.114} + \sqrt{0.086} + \sqrt{0.072}$$

$$BC = 0.338 + 0.293 + 0.268 = 0.899$$

**گام ۳:** محاسبه Bhattacharyya Distance
$$BD = -\ln(0.899) = -(-0.107) = 0.107$$

**گام ۴:** تفسیر نتیجه
- BC = 0.899 نشان‌دهنده تشابه نسبتاً بالای دو توزیع
- BD = 0.107 نشان‌دهنده فاصله کم بین دو شاخه (تقسیم ضعیف)

## ویژگی‌های فنی

- **پیچیدگی محاسباتی:** $O(k)$ جایی که k تعداد کلاس‌هاست
- **محدوده مقادیر:** $0 \leq BD \leq +\infty$
- **symmetry:** $BD(P,Q) = BD(Q,P)$
- **triangle inequality:** در حالت کلی برقرار نیست
- **پایداری عددی:** نیاز به مدیریت دقیق لگاریتم صفر

### خصوصیات نظری:
- **Upper bound برای خطا:** $P_{error} \leq \sqrt{P_1 P_2} e^{-BD}$
- **ارتباط با Chernoff bound:** BD حد بالای بهینه برای خطای طبقه‌بندی
- **Hellinger distance:** $H^2 = 2(1 - BC)$، رابطه مستقیم با BD

### تفسیر مقادیر:
- **BD = 0**: توزیع‌های یکسان (تقسیم بی‌فایده)
- **BD کوچک**: توزیع‌های مشابه (تقسیم ضعیف)
- **BD بزرگ**: توزیع‌های متفاوت (تقسیم قوی)

## مقایسه با سایر معیارها

| جنبه | Bhattacharyya | Hellinger | Chi-Squared | Information Gain |
|------|---------------|-----------|-------------|------------------|
| **مبنای نظری** | فاصله احتمالی | فاصله احتمالی | آزمون استقلال | نظریه اطلاعات |
| **محدوده مقادیر** | $[0, +\infty)$ | $[0, 1]$ | $[0, +\infty)$ | متغیر |
| **تفسیرپذیری** | upper bound خطا | فاصله هندسی | معناداری آماری | کاهش آنتروپی |
| **مقاومت به نویز** | بالا | بالا | متوسط | پایین |
| **پیچیدگی محاسبه** | $O(k)$ | $O(k)$ | $O(k)$ | $O(k \log k)$ |
| **کاربرد عملی** | تشخیص الگو | چندکلاسه نامتعادل | آزمون آماری | درخت‌های کلاسیک |

## مزایا

### 1. پایه نظری محکم
- **ارتباط با optimal classification:** ارائه upper bound برای خطای Bayes
- **تفسیر هندسی:** فاصله در فضای احتمال با معنای روشن
- **پیوند با سایر معیارها:** ارتباط مستقیم با Hellinger و Chernoff distances

### 2. مقاومت در برابر نویز
- **تأثیر کمتر outlier:** استفاده از جذر در محاسبه BC
- **پایداری در برابر تغییرات کوچک:** تغییرات محلی کمتر بر نتیجه تأثیر می‌گذارند
- **robust estimation:** مناسب برای داده‌های پرنویز

### 3. کاربرد وسیع
- **computer vision:** histogram comparison و feature matching
- **signal processing:** تشخیص الگوهای آماری
- **machine learning:** feature selection و clustering
- **medical diagnosis:** مقایسه پروفایل‌های بیولوژیکی

### 4. ویژگی‌های محاسباتی مطلوب
- **محاسبه سریع:** پیچیدگی خطی نسبت به تعداد کلاس‌ها
- **قابلیت vectorization:** مناسب برای پیاده‌سازی موثر با NumPy
- **عدم نیاز به پارامتر:** بدون نیاز به تنظیم hyperparameter

## محدودیت‌ها

### 1. حساسیت به احتمالات صفر
- **مشکل محاسباتی:** وقتی $p_i = 0$ یا $q_i = 0$، ترم مربوطه صفر می‌شود
- **نیاز به smoothing:** استفاده از Laplace smoothing یا روش‌های مشابه
- **تأثیر بر دقت:** smoothing ممکن است اطلاعات را تحریف کند

### 2. عدم تبعیت از triangle inequality
- **محدودیت متریک:** BD یک pseudo-metric است، نه metric کامل
- **مشکل در clustering:** برخی الگوریتم‌های clustering نیاز به triangle inequality دارند
- **تفسیر هندسی محدود:** نمی‌توان آن را کاملاً به عنوان فاصله هندسی در نظر گرفت

### 3. وابستگی به توزیع کلاس‌ها
- **تأثیر class imbalance:** در مجموعه داده‌های نامتعادل ممکن است مشکل‌ساز باشد
- **حساسیت به اندازه نمونه:** در نمونه‌های کوچک تخمین احتمالات نادقیق است
- **نیاز به نمونه کافی:** برای تخمین موثق احتمالات

### 4. تفسیر پیچیده نسبت به معیارهای ساده
- **نیاز به دانش آماری:** درک مفهوم فاصله احتمالی
- **مقایسه با بصیرت انسانی:** کمتر قابل تطبیق با درک شهودی تقسیم‌ها
- **ارتباط غیرمستقیم با هدف:** ارتباط با accuracy از طریق upper bound

## نکات پیاده‌سازی

### مراقبت‌های ضروری:
- **مدیریت احتمالات صفر:** استفاده از Laplace smoothing یا $\epsilon$-smoothing
- **کنترل overflow/underflow:** محاسبه در log space برای پایداری عددی
- **نرمال‌سازی احتمالات:** اطمینان از $\sum p_i = 1$ و $\sum q_i = 1$
- **مدیریت BC=0:** در صورت BC=0، تنظیم BD به مقدار بزرگ مناسب

### بهینه‌سازی:
- استفاده از NumPy vectorized operations برای محاسبه سریع
- کش کردن محاسبات توزیع احتمال برای استفاده مجدد
- محاسبه در log space: $\ln(BC) = \frac{1}{2}\sum \ln(p_i q_i)$
- تنظیم threshold برای احتمالات خیلی کوچک

### الگوی پیاده‌سازی:

نرمال‌سازی و smoothing

p_smooth = (counts_left + epsilon) / (n_left + k * epsilon)
q_smooth = (counts_right + epsilon) / (n_right + k * epsilon)
محاسبه BC

bc = np.sum(np.sqrt(p_smooth * q_smooth))
محاسبه BD

bd = -np.log(bc) if bc > 0 else float('inf')

## کد شبه

def bhattacharyya_distance(y_left, y_right, epsilon=1e-10):
"""
محاسبه فاصله Bhattacharyya بین دو توزیع کلاس
Parameters:
y_left: آرایه برچسب‌های کلاس برای شاخه چپ
y_right: آرایه برچسب‌های کلاس برای شاخه راست
epsilon: مقدار smoothing برای جلوگیری از احتمالات صفر

Returns:
BD: فاصله Bhattacharyya (مقدار بالاتر = تقسیم بهتر)
"""

# یافتن کلاس‌های موجود
all_classes = unique(concatenate([y_left, y_right]))
k = len(all_classes)

# شمارش فراوانی کلاس‌ها
counts_left = zeros(k)
counts_right = zeros(k)

for i, class_label in enumerate(all_classes):
    counts_left[i] = sum(y_left == class_label)
    counts_right[i] = sum(y_right == class_label)

n_left = len(y_left)
n_right = len(y_right)

# اعمال Laplace smoothing
p_smooth = (counts_left + epsilon) / (n_left + k * epsilon)
q_smooth = (counts_right + epsilon) / (n_right + k * epsilon)

# محاسبه Bhattacharyya Coefficient
bc = sum(sqrt(p_smooth * q_smooth))

# محاسبه Bhattacharyya Distance
if bc > 0:
    bd = -log(bc)
else:
    bd = float('inf')  # حداکثر فاصله برای توزیع‌های کاملاً متفاوت

return bd

def bhattacharyya_coefficient(y_left, y_right, epsilon=1e-10):
"""محاسبه ضریب Bhattacharyya (برای تحلیل تشابه)"""
# مراحل مشابه تا محاسبه p_smooth و q_smooth
# ...

bc = sum(sqrt(p_smooth * q_smooth))
return bc

def robust_bhattacharyya_distance(y_left, y_right, min_samples=5):
"""نسخه robust با کنترل حداقل نمونه"""
if len(y_left) < min_samples or len(y_right) < min_samples:
    return 0.0  # تقسیم نامعتبر

return bhattacharyya_distance(y_left, y_right)

```

## منابع
- Bhattacharyya, A. (1943). On a measure of divergence between two statistical populations defined by their probability distributions. Bulletin of the Calcutta Mathematical Society, 35, 99-109
- Chernoff, H. (1952). A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations. The Annals of Mathematical Statistics, 23(4), 493-507
- Fukunaga, K. (1990). Introduction to Statistical Pattern Recognition. Academic Press
- Comaniciu, D., Ramesh, V., Meer, P. (2003). Kernel-based object tracking. IEEE Transactions on Pattern Analysis and Machine Intelligence
- مقالات کاربردی در computer vision و pattern recognition
