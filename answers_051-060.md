# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (51-60)

### 51. Neyron tarmog‘ining oldinga tarqalish (forward pass) jarayonini bosqichma-bosqich tushuntirib bering.

**Javob:**
Oldinga tarqalish (Forward Propagation) — bu modelga kirish ma’lumotlari berilgandan boshlab, to natija (bashorat) chiqquncha bo‘lgan jarayondir. Bu jarayonda ma’lumot tarmoqning boshidan oxirigacha oqib o‘tadi.

**Bosqichlari:**
1.  **Kirish (Input):** Ma’lumotlar (masalan, rasm piksellari vektori $x$) kirish qatlamiga uzatiladi.
2.  **Chiziqli Transformatsiya (Linear Transformation):** Birinchi yashirin qatlamda har bir neyron kirish qiymatlarini o‘zining vaznlariga ($w$) ko‘paytiradi va siljish ($b$) ni qo‘shadi: $z = w \cdot x + b$. Bu har bir neyronning "o‘z fikrini" shakllantirishidir. Matritsa ko‘rinishida: $Z = WX + B$.
3.  **Faollashtirish (Activation):** Hosil bo‘lgan qiymat ($z$) chiziqli bo‘lmagan funksiyadan (masalan, ReLU) o‘tkaziladi: $a = \sigma(z)$. Bu qadam neyronni "yoqadi" yoki "o‘chiradi" va modelga murakkablik beradi.
4.  **Qatlamlararo uzatish:** Chiqqan natija ($a$) keyingi qatlam uchun kirish ma’lumoti bo‘lib xizmat qiladi. 2 va 3-bosqichlar barcha yashirin qatlamlar uchun takrorlanadi.
5.  **Chiqish (Output):** Eng oxirgi qatlamda natija olinadi. Agar tasniflash bo‘lsa, Softmax orqali ehtimolliklar ($p$) hisoblanadi. Bu modelning yakuniy bashoratidir ($\hat{y}$). 

---

### 52. Neyron tarmog‘ining orqaga tarqalish (backward pass) jarayonini bosqichma-bosqich izohlab bering.

**Javob:**
Orqaga tarqalish (Backpropagation) — bu modelning "o‘rganish" qismidir. Maqsad: qilingan xatoni tahlil qilib, vaznlarni to‘g‘rilash.

**Bosqichlari:**
1.  **Xatolikni hisoblash (Loss Computation):** Oldinga tarqalish tugagach, bashorat ($\hat{y}$) va haqiqiy javob ($y$) solishtiriladi va xatolik funksiyasi qiymati ($L$) hisoblanadi (masalan, Cross-Entropy).
2.  **Gradientni hisoblash (Output Layer):** Avval eng oxirgi qatlamdagi xatolikning hosilasi (gradienti) hisoblanadi. Biz so‘raymiz: "Natija to‘g‘riroq bo‘lishi uchun oxirgi neyronlar qiymati qanchaga o‘zgarishi kerak?".
3.  **Zanjir qoidasi (Chain Rule):** Xatolik signali orqaga, qatlamdan-qatlamga uzatiladi. Har bir qatlamda mahalliy gradientlar hisoblanadi va ular bir-biriga ko‘paytiriladi: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$. Bu orqali biz har bir ichki vazn ($w$) umumiy xatoga qanchalik "aybdor" ekanini aniqlaymiz.
4.  **Parametrlarni yangilash (Optimization):** Barcha gradientlar hisoblab bo‘lingach, Optimizator (masalan, SGD yoki Adam) ishga tushadi. U har bir vaznni gradientga qarama-qarshi yo‘nalishda biroz o‘zgartiradi: $w_{new} = w_{old} - \eta \cdot \text{gradient}$.

---

### 53. Ma’lumotlarni train/validation/test to‘plamlariga ajratish nima uchun muhim ekanini tushuntirib bering.

**Javob:**
Ma’lumotlarni ajratish modelning **xolisligini** va **haqiqiy samaradorligini** baholashning yagona yo‘lidir.

1.  **Training Set (60-80%):** Model faqat shu ma’lumotlarni ko‘radi va o‘rganadi (vaznlarini o‘zgartiradi). Agar biz hamma ma’lumotni bunga ishlatsak, model "imtihon savollarini oldindan yodlab olgan talaba" kabi bo‘lib qoladi.
2.  **Validation Set (10-20%):** Bu qism modelni o‘qitish davomida uni tekshirib turish uchun kerak. Biz bunga qarab: "O‘qishni to‘xtataymi?", "Learning rateni o‘zgartiraymi?", "Model yodlab olmayaptimi?" degan savollarga javob topamiz. Model bu ma’lumotlarni ko‘rmaydi (vazn o‘zgarmaydi), lekin biz ulardan foydalanib modelni **sozlaymiz**.
3.  **Test Set (10-20%):** Bu "yopiq sandiq". Model to‘liq tayyor bo‘lib, barcha sozlashlar tugagandan keyin, faqat **bir marta** ochiladi. Bu modelning real hayotdagi kutilayotgan aniqligini ko‘rsatadi. Agar Test set bo‘lmasa, biz o‘zimizni o‘zimiz aldab, Validation setga moslashib qolgan (overfitted to validation) model yaratishimiz mumkin.

---

### 54. Modelni baholashda aniqlik (accuracy) va F1-score ko‘rsatkichlarini taqqoslab bering.

**Javob:**
Ikkalasi ham model sifatini o‘lchaydi, lekin turli vaziyatlarda ishlatiladi.

1.  **Accuracy (Aniqlik):**
    *   Formula: $\frac{\text{To‘g‘ri javoblar}}{\text{Jami javoblar}}$.
    *   **Qachon yaxshi?** Sinflar muvozanatli bo‘lganda (masalan, 50 ta it, 50 ta mushuk).
    *   **Kamchiligi:** Sinflar nomutanosib bo‘lsa, u chalg‘itadi. Misol: 95 ta sog‘lom, 5 ta kasal odam bor. Model hammaga "Sog‘lom" desa, Accuracy = 95% chiqadi. Bu ajoyib ko‘rinadi, lekin aslida model **foydasiz**, chunki u birorta ham kasalni topmadi.

2.  **F1-Score:**
    *   Bu **Precision** va **Recall** ko‘rsatkichlarining garmonik o‘rtachasidir.
    *   Formula: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
    *   **Qachon yaxshi?** Nomutanosib sinflar (imbalanced data) bilan ishlaganda.
    *   **Afzalligi:** F1-score yuqori bo‘lishi uchun model ham yolg‘on signal bermasligi (Precision), ham haqiqiy holatlarni o‘tkazib yubormasligi (Recall) kerak. Yuqoridagi misolda "hamma sog‘lom" degan modelning F1-score qiymati 0 ga teng bo‘ladi, bu uning yomonligini yaqqol ko‘rsatadi.

---

### 55. Tasniflash vazifalarida chalkashlik matritsasi (confusion matrix) nima ekanini va qanday axborot berishini tushuntiring.

**Javob:**
Chalkashlik matritsasi (Confusion Matrix) — bu modelning bashoratlari va haqiqiy qiymatlar o‘rtasidagi munosabatni batafsil ko‘rsatuvchi jadvaldir. $N$ ta sinf uchun $N \times N$ jadval tuziladi.
Binar (2 sinfli) tasniflash uchun u 4 ta katakdan iborat:

1.  **True Positive (TP):** Model "Ha" dedi va bu **To‘g‘ri**. (Kasalni topdi).
2.  **True Negative (TN):** Model "Yo‘q" dedi va bu **To‘g‘ri**. (Sog‘lomni topdi).
3.  **False Positive (FP):** Model "Ha" dedi, lekin bu **Xato**. ("Yolg‘on signal" — sog‘lom odamni kasal dedi). I-tur xatolik.
4.  **False Negative (FN):** Model "Yo‘q" dedi, lekin bu **Xato**. (Kasalni o‘tkazib yubordi). II-tur xatolik.

**Axborot:**
Bu jadval orqali biz model aynan qayerda adashayotganini ko‘ramiz. U ko‘proq yolg‘on signal beryaptimi (FP) yoki narsalarni ko‘rmay qolyaptimi (FN)? Bu bizga strategiyani to‘g‘rilashga yordam beradi.

---

### 56. Precision va recall ko‘rsatkichlari nimani ifodalashini va ularning rolini izohlab bering.

**Javob:**
Bu ikki ko‘rsatkich o‘rtasida doimiy "savdo" (trade-off) mavjud.

1.  **Precision (Aniqlik/Sifat):**
    *   Savol: "Model 'bu olma' deganlarning necha foizi haqiqatdan ham olma?"
    *   Formula: $\frac{TP}{TP + FP}$.
    *   **Rol:** Bizga **yolg‘on signallar** (False Positive) qimmatga tushganda muhim. Masalan, spam-filtr. Agar muhim xat "spam" deb ketib qolsa (FP), bu yomon. Biz spamni o‘tkazib yuborsak ham, muhim xatni yo‘qotmasligimiz kerak (High Precision).

2.  **Recall (To‘liqlik/Qamrov):**
    *   Savol: "Dunyodagi barcha olmalarning qanchasini model topa oldi?"
    *   Formula: $\frac{TP}{TP + FN}$.
    *   **Rol:** Bizga **o‘tkazib yuborish** (False Negative) xavfli bo‘lganda muhim. Masalan, saratonni aniqlash. Agar model kasalni "sog‘lom" deb yuborsa (FN), odam o‘lishi mumkin. Bizga yolg‘on bo‘lsa ham shubhalarni topish muhimroq (High Recall).

---

### 57. ROC egri chizig‘i (ROC curve) va AUC ko‘rsatkichlarini tushuntirib bering.

**Javob:**
Model (masalan, Logistik regressiya) javobni "Ha" yoki "Yo‘q" deb emas, balki ehtimollik ko‘rinishida (0.8, 0.3) beradi. Biz qayerdan chegara (threshold) qo‘yishni (masalan, >0.5) o‘zimiz hal qilamiz.

1.  **ROC Curve (Receiver Operating Characteristic):**
    *   Bu grafik bo‘lib, unda biz chegara (threshold) ni 0 dan 1 gacha o‘zgartirib chiqqanimizda modelning **Recall (True Positive Rate)** va **False Positive Rate** qanday o‘zgarishini chizamiz.
    *   Bu egri chiziq modelning barcha mumkin bo‘lgan chegaralardagi "xulq-atvorini" ko‘rsatadi.

2.  **AUC (Area Under the Curve):**
    *   Bu ROC egri chizig‘i ostidagi **yuzadir**.
    *   Qiymati 0 dan 1 gacha bo‘ladi.
    *   **AUC = 0.5:** Model tavakkal (random) ishlamoqda (tanga tashlash bilan teng).
    *   **AUC = 1.0:** Ideal model.
    *   **Ma’nosi:** AUC qanchalik baland bo‘lsa, model musbat va manfiy sinflarni shunchalik yaxshi ajratadi. AUC bizga aniq bir chegara tanlamasdan turib modelning umumiy sifatini baholash imkonini beradi.

---

### 58. Qaysi holatlarda MSE (mean squared error) ni MAE (mean absolute error) ga nisbatan afzal ko‘rish mumkinligini izohlang.

**Javob:**
Regressiya masalalarida (sonni bashorat qilishda) ikkalasi ham ishlatiladi.

*   **MSE (O‘rtacha kvadratik xato):** $\frac{1}{n} \sum (y - \hat{y})^2$.
*   **MAE (O‘rtacha absolyut xato):** $\frac{1}{n} \sum |y - \hat{y}|$.

**MSE ning afzalligi va qo‘llanishi:**
MSE xatolikni kvadratga oshiradi. Bu shuni anglatadiki, **katta xatolar** juda qattiq jazolanadi.
*   Agar bashorat 2 birlikka adashsa, jarima $2^2=4$.
*   Agar 10 birlikka adashsa, jarima $10^2=100$. (Xato 5 barobar oshdi, jarima 25 barobar).
**Xulosa:** Agar siz uchun katta xatolarga yo‘l qo‘ymaslik o‘ta muhim bo‘lsa (masalan, jarrohlik roboti yoki avtonom mashina boshqaruvi), MSE afzal. Kichik xatolar kechirilishi mumkin, lekin bitta katta xato falokatga olib keladi.
Shuningdek, MSE matematik jihatdan silliq (hosilasi oson olinadi), MAE esa 0 nuqtasida silliq emas.

---

### 59. Bias–variance muvozanati (tradeoff) tushunchasini chuqur o‘qitish kontekstida tushuntirib bering.

**Javob:**
Har qanday modelning xatoligi ikki qismdan iborat: Bias (Siljish) va Variance (Tarqoqlik).

1.  **Bias (Underfitting):** Model juda sodda. U ma’lumotlardagi murakkab bog‘liqlikni ilg‘ay olmaydi.
    *   Chuqur o‘qitishda: Qatlamlar kam, neyronlar yetarli emas. Natijada Train va Test xatoligi ikkalasi ham yuqori bo‘ladi.
2.  **Variance (Overfitting):** Model juda murakkab va sezgir. U ma’lumotdagi shovqinni ham o‘rganib oladi.
    *   Chuqur o‘qitishda: Parametrlar juda ko‘p, ma’lumot kam. Natijada Train xatoligi juda past, lekin Test xatoligi yuqori bo‘ladi.

**Trade-off (Muvozanat):**
Bizning maqsadimiz o‘rtalikni topishdir.
*   Klassik MLda biz model murakkabligini ehtiyotkorlik bilan tanlardi.
*   **Deep Learning davrida:** Biz odatda "High Variance" (juda katta model) yo‘lidan boramiz va Variance ni kamaytirish uchun **ko‘proq ma’lumot** va **regulyarizatsiya** (Dropout, L2) ishlatamiz. Zamonaviy yondashuv: "Modelni iloji boricha katta qil (Low Bias), keyin uni jilovla (Low Variance)".

---

### 60. Nima uchun k-fold cross-validation usuli qo‘llanadi? Uning maqsadini izohlab bering.

**Javob:**
**Muammo:**
Agar ma’lumotlarimiz kam bo‘lsa (masalan, 1000 ta rasm), biz uni Train/Test ga ajratganimizda (800/200), tasodifan Test qismiga juda oson yoki juda qiyin misollar tushib qolishi mumkin. Natijada biz olgan baho (Accuracy) modelning haqiqiy kuchini ko‘rsatmasligi mumkin (omadga bog‘liq bo‘lib qoladi).

**Yechim (K-Fold):**
Biz ma’lumotni $K$ ta (masalan, 5 ta) teng qismga bo‘lamiz.
1.  1-qadam: 1-qism Test, qolgan 4 tasi Train bo‘ladi. Baho olamiz.
2.  2-qadam: 2-qism Test, qolganlari Train. Baho olamiz.
3.  ... shu tarzda 5 marta takrorlaymiz.

**Maqsad:**
Oxirida 5 ta bahoning **o‘rtachasini** olamiz. Bu o‘rtacha baho modelning haqiqiy sifatini ancha ishonchli va barqaror ifodalaydi. K-fold usuli tasodifiy bo‘linish ta’sirini yo‘qotadi va barcha ma’lumotlardan ham o‘qitish, ham tekshirish uchun foydalanish imkonini beradi.