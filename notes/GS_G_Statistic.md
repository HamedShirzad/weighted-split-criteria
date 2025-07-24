# G Statistic (GS)

## اطلاعات کلی
- **منبع اصلی:** Expert Systems in Pattern Recognition - John Mingers  
- **سال معرفی:** 1987
- **نویسنده:** John Mingers
- **مجله:** Pattern Recognition and Artificial Intelligence Research
- **انگیزه:** رفع محدودیت‌های Chi-square و کنترل overfitting در درخت‌های تصمیم
- **مزیت کلیدی:** مبنای Information Theory و کنترل معناداری آماری

## هدف و کاربرد

معیار G Statistic (همچنین به نام G-test یا log-likelihood ratio test شناخته می‌شود) برای رفع مشکلات اساسی معیارهای آماری موجود در درخت‌های تصمیم طراحی شده:

1. **کنترل overfitting** - توقف خودکار زمانی که تقسیم از نظر آماری معنادار نیست
2. **مقاومت در برابر فراوانی‌های کم** - عملکرد بهتر از Chi-square در سلول‌های کم‌تعداد
3. **مدیریت داده‌های تصادفی** - قابلیت کار با داده‌های غیرقطعی و دارای عدم اطمینان
4. **کنترل معناداری آماری** - استفاده از hypothesis testing برای اعتبارسنجی تقسیم‌ها

این معیار مخصوصاً برای **داده‌های real-world** که دارای نویز و عدم قطعیت هستند، و **جلوگیری از ایجاد درخت‌های پیچیده** بسیار مناسب است.

## فرمول ریاضی

### فرمول اصلی G Statistic
$$G = 2 \sum_{i,j} x_{ij} \ln\left(\frac{x_{ij}}{E_{ij}}\right)$$

### رابطه با Information Measure
$$G = 2N \times \text{Information Measure}$$

### فرمول تفصیلی برای تقسیم درخت
برای تقسیم یک گره با $k$ کلاس به $v$ شاخه:

$$G = 2 \sum_{i=1}^{v} \sum_{j=1}^{k} n_{ij} \ln\left(\frac{n_{ij} \cdot N}{n_{i \cdot} \cdot n_{\cdot j}}\right)$$

### تعریف متغیرها

| نماد | تعریف |
|------|--------|
| **$G$** | مقدار آماره G Statistic |
| **$x_{ij}$** | فراوانی مشاهده‌شده در سلول $(i,j)$ |
| **$E_{ij}$** | فراوانی مورد انتظار در سلول $(i,j)$ |
| **$n_{ij}$** | تعداد نمونه‌های کلاس $j$ در شاخه $i$ |
| **$n_{i \cdot}$** | تعداد کل نمونه‌ها در شاخه $i$ |
| **$n_{\cdot j}$** | تعداد کل نمونه‌های کلاس $j$ |
| **$N$** | تعداد کل نمونه‌ها |
| **$v$** | تعداد شاخه‌های تقسیم |
| **$k$** | تعداد کلاس‌های موجود |

### محاسبه Expected Values
$$E_{ij} = \frac{n_{i \cdot} \times n_{\cdot j}}{N}$$

### درجه آزادی
$$df = (v-1) \times (k-1)$$

## مثال محاسبه

فرض کنید یک گره 100 نمونه را به 2 شاخه تقسیم می‌کند:

### داده‌های نمونه:

| شاخه | کلاس A | کلاس B | کلاس C | مجموع |
|------|---------|---------|---------|-------|
| چپ | 25 | 10 | 5 | 40 |
| راست | 15 | 30 | 15 | 60 |
| **مجموع کل** | 40 | 40 | 20 | 100 |

### گام‌های محاسبه:

**گام ۱:** محاسبه Expected Values
- $E_{11} = \frac{40 \times 40}{100} = 16$ (کلاس A، شاخه چپ)
- $E_{12} = \frac{40 \times 40}{100} = 16$ (کلاس B، شاخه چپ)  
- $E_{13} = \frac{40 \times 20}{100} = 8$ (کلاس C، شاخه چپ)
- $E_{21} = \frac{60 \times 40}{100} = 24$ (کلاس A، شاخه راست)
- $E_{22} = \frac{60 \times 40}{100} = 24$ (کلاس B، شاخه راست)
- $E_{23} = \frac{60 \times 20}{100} = 12$ (کلاس C، شاخه راست)

**گام ۲:** محاسبه G Statistic
$$G = 2[25 \ln(\frac{25}{16}) + 10 \ln(\frac{10}{16}) + 5 \ln(\frac{5}{8}) + 15 \ln(\frac{15}{24}) + 30 \ln(\frac{30}{24}) + 15 \ln(\frac{15}{12})]$$

$$G = 2[25 \times 0.451 + 10 \times (-0.470) + 5 \times (-0.470) + 15 \times (-0.470) + 30 \times 0.223 + 15 \times 0.223]$$

$$G = 2[11.275 - 4.700 - 2.350 - 7.050 + 6.690 + 3.345] = 2 \times 7.210 = 14.420$$

**گام ۳:** بررسی معناداری
- درجه آزادی: $df = (2-1) \times (3-1) = 2$
- مقدار بحرانی $\chi^2_{0.05, 2} = 5.991$
- $G = 14.420 > 5.991$ → تقسیم معنادار است

## ویژگی‌های فنی

- **پیچیدگی محاسباتی:** $O(v \times k)$ جایی که $v$ تعداد شاخه‌ها و $k$ تعداد کلاس‌هاست
- **توزیع احتمال:** تقریباً از توزیع $\chi^2$ با درجه آزادی $(v-1)(k-1)$ پیروی می‌کند
- **حساسیت به فراوانی کم:** کمتر از Chi-square معمولی
- **کنترل معناداری:** امکان توقف خودکار بر اساس significance level

### تفسیر مقادیر:
- **$G = 0$**: استقلال کامل بین ویژگی و کلاس (تقسیم بی‌فایده)
- **$G$ بالا**: وابستگی قوی بین ویژگی و کلاس (تقسیم مفید)
- **مقایسه با $\chi^2$ جدول**: تعیین معناداری آماری

## مقایسه با سایر معیارها

| جنبه | G Statistic | Chi-square | Information Gain | Gini |
|------|-------------|------------|------------------|------|
| **مبنای نظری** | Information Theory | آمار کلاسیک | Information Theory | احتمالات |
| **کنترل overfitting** | بالا | متوسط | پایین | پایین |
| **حساسیت به فراوانی کم** | کم | زیاد | متوسط | کم |
| **معناداری آماری** | دارد | دارد | ندارد | ندارد |
| **پیچیدگی محاسبه** | بالا | متوسط | کم | کم |
| **دقت در دم‌های توزیع** | بالا | متوسط | - | - |

## مزایا

### 1. کنترل قدرتمند overfitting
- توقف خودکار زمانی که تقسیم از نظر آماری معنادار نیست
- جلوگیری از ایجاد شاخه‌های اضافی و غیرضروری
- کاهش قابل توجه پیچیدگی درخت نهایی

### 2. مبنای نظری محکم
- استفاده از **likelihood ratio test** که مبنای قوی آماری دارد
- ارتباط مستقیم با **Information Theory** و معیار Information Measure
- تقریب دقیق‌تر به توزیع $\chi^2$ نسبت به Chi-square معمولی

### 3. مقاومت در برابر داده‌های کم
- عملکرد بهتر از Chi-square در فراوانی‌های پایین
- مدیریت بهتر سلول‌های خالی یا کم‌تعداد در جداول contingency
- کمتر متأثر از نویز و outlier

### 4. کنترل کیفیت تقسیم
- هر تقسیم با معیار آماری معتبر ارزیابی می‌شود
- امکان تنظیم سطح اطمینان ($\alpha$) بر اساس نیاز
- شفافیت در تصمیم‌گیری درباره توقف یا ادامه تقسیم

## محدودیت‌ها

### 1. پیچیدگی محاسباتی
- نیاز به محاسبه لگاریتم طبیعی که نسبت به عملیات ساده کندتر است
- محاسبه Expected values برای جداول contingency بزرگ وقت‌گیر
- نیاز به دقت عددی بالا برای جلوگیری از خطاهای تجمعی

### 2. حساسیت به پارامترهای آماری
- انتخاب سطح معناداری ($\alpha$) تأثیر مستقیم بر عملکرد دارد
- در نمونه‌های کوچک ممکن است نتایج reliable نباشد
- فرض مستقل بودن مشاهدات که همیشه در عمل برقرار نیست

### 3. پیچیدگی تفسیر
- درک مفهوم likelihood ratio برای غیرمتخصصان دشوار است
- نیاز به آشنایی با مفاهیم آماری پیشرفته
- تفسیر مقادیر G نسبت به معیارهای ساده پیچیده‌تر

### 4. محدودیت در داده‌های پیوسته
- طراحی شده برای داده‌های گسسته و جداول contingency
- برای ویژگی‌های پیوسته نیاز به discretization
- ممکن است اطلاعات مهمی در فرآیند گسسته‌سازی از دست برود

## نکات پیاده‌سازی

### مراقبت‌های ضروری:
- **مدیریت ln(0):** اگر $x_{ij} = 0$، آن ترم را از مجموع حذف کنید
- **Expected values کوچک:** اگر $E_{ij} < 5$، دقت نتایج کاهش می‌یابد
- **درجه آزادی:** محاسبه صحیح $(v-1)(k-1)$ برای آزمون معناداری
- **سطح معناداری:** انتخاب مناسب $\alpha$ (معمولاً 0.05 یا 0.01)

### بهینه‌سازی:
- کش کردن محاسبات لگاریتم برای استفاده مجدد
- استفاده از log-sum-exp trick برای پایداری عددی
- محاسبه موثر Expected values با عملیات ماتریسی
- تنظیم دقت عددی مناسب برای محاسبات حساس

### الگوی پیاده‌سازی:
محاسبه Contingency Table

observed = create_contingency_table(y_left, y_right, classes)
expected = calculate_expected_values(observed)
محاسبه G Statistic

g_value = 2 * sum(observed[i][j] * ln(observed[i][j] / expected[i][j]))
آزمون معناداری

degrees_of_freedom = (n_branches - 1) * (n_classes - 1)
critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)
is_significant = g_value > critical_value

## کد شبه

def g_statistic(y_left, y_right, alpha=0.05):
"""
محاسبه G Statistic و بررسی معناداری آماری
Parameters:
y_left: آرایه برچسب‌های کلاس برای شاخه چپ
y_right: آرایه برچسب‌های کلاس برای شاخه راست
alpha: سطح معناداری (پیش‌فرض: 0.05)

Returns:
G: مقدار G Statistic
is_significant: آیا تقسیم معنادار است
p_value: مقدار p برای آزمون
"""

# ایجاد جدول contingency
all_classes = unique(concatenate([y_left, y_right]))
k = len(all_classes)  # تعداد کلاس‌ها
v = 2  # تعداد شاخه‌ها (باینری)

# محاسبه فراوانی‌های مشاهده شده
observed = zeros((v, k))
for i, class_label in enumerate(all_classes):
    observed[0, i] = sum(y_left == class_label)
    observed[1, i] = sum(y_right == class_label)

# محاسبه فراوانی‌های مورد انتظار
n_total = len(y_left) + len(y_right)
row_totals = [len(y_left), len(y_right)]
col_totals = [sum(observed[:, j]) for j in range(k)]

expected = zeros((v, k))
for i in range(v):
    for j in range(k):
        expected[i, j] = (row_totals[i] * col_totals[j]) / n_total

# محاسبه G Statistic
g_value = 0.0
for i in range(v):
    for j in range(k):
        if observed[i, j] > 0 and expected[i, j] > 0:
            ratio = observed[i, j] / expected[i, j]
            g_value += observed[i, j] * ln(ratio)

g_value *= 2

# آزمون معناداری
degrees_of_freedom = (v - 1) * (k - 1)

if degrees_of_freedom > 0:
    critical_value = chi2_critical_value(1 - alpha, degrees_of_freedom)
    p_value = chi2_p_value(g_value, degrees_of_freedom)
    is_significant = g_value > critical_value
else:
    p_value = 1.0
    is_significant = False

return g_value, is_significant, p_value
def chi2_critical_value(confidence, df):
"""محاسبه مقدار بحرانی توزیع chi-square"""
# استفاده از جداول chi-square یا کتابخانه آماری
return lookup_chi2_table(confidence, df)

def chi2_p_value(g_value, df):
"""محاسبه p-value برای مقدار G"""
# استفاده از تابع CDF معکوس توزیع chi-square
return 1 - chi2_cdf(g_value, df)


```

## منابع
- Mingers, J. (1987). Expert Systems in Pattern Recognition. Pattern Recognition and Artificial Intelligence Research
- Sokal, R.R., Rohlf, F.J. (1981). Biometry: The Principles and Practice of Statistics in Biological Research. W.H. Freeman
- Agresti, A. (2002). Categorical Data Analysis. John Wiley & Sons
- McDonald, J.H. (2014). Handbook of Biological Statistics. Sparky House Publishing
- مقالات مقایسه‌ای معیارهای آماری در درخت‌های تصمیم
