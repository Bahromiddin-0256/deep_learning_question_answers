# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (41-50)

### 41. Swish faollashtirish funksiyasi nima va nega u ReLU ning yaxshilangan varianti sifatida ko‘rilishini izohlang.

**Javob:**
**Tushuncha:**
Swish — bu Google Brain jamoasi tomonidan (avtomatik qidiruv - AutoML orqali) topilgan zamonaviy faollashtirish funksiyasi.
Formula: $f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$.
Bu yerda $\sigma(x)$ — sigmoid funksiyasi.

**Nega u ReLU dan yaxshiroq?**
1.  **Silliqlik (Smoothness):** ReLU $x=0$ nuqtasida keskin burchakka ("tirsak") ega, bu nuqtada hosila mavjud emas (uzilish bor). Swish esa butun o‘q bo‘ylab silliq (uzluksiz hosilaga ega). Bu optimizatsiya jarayonida gradientning tekisroq o‘zgarishiga yordam beradi va modelni chuqur minimumga tushishini osonlashtiradi.
2.  **Monoton emaslik (Non-monotonicity):** Swish funksiyasi manfiy sohada (taxminan $-5$ dan $0$ gacha) biroz pasayib, keyin yana $0$ ga qaytadi (kichik "chuqurcha"si bor). Bu xususiyat ReLU da yo‘q. Bu chuqurcha ba’zi murakkab ma’lumotlar strukturasini o‘rganishda va manfiy gradientlarni butunlay o‘chirib tashlamaslikda foydali.
3.  **Dying ReLU muammosi yo‘q:** Manfiy qiymatlarda u nolga teng emas (biroz manfiy qiymat va gradient bor), shuning uchun neyronlar butunlay "o‘lib" qolmaydi.
**Natija:** Tadqiqotlar (masalan, EfficientNet maqolasi) shuni ko‘rsatadiki, chuqur tarmoqlarda Swish oddiy ReLU ga qaraganda barqaror ravishda 0.5-1% yuqori aniqlik beradi.

---\n

### 42. To‘yingan (saturating) faollashtirish funksiyalari muammosini tushuntirib bering.

**Javob:**
**Tushuncha:**
To‘yingan funksiya — bu kirish qiymati ($x$) oshib yoki kamayib ketganda, chiqish qiymati ma’lum bir songa (limitga) yopishib qoladigan funksiyadir.
*   **Sigmoid:** $x \to \infty$ da $y \to 1$, $x \to -\infty$ da $y \to 0$.
*   **Tanh:** $x \to \infty$ da $y \to 1$, $x \to -\infty$ da $y \to -1$.

**Muammo (Vanishing Gradient):**
Funksiya to‘yingan hududda (grafik tekislashgan joyda) uning **hosilasi (gradienti) nolga teng** bo‘ladi.
*   Backpropagation paytida gradient zanjir qoidasi bo‘yicha ko‘paytiriladi: $\frac{\partial L}{\partial w} = \dots \times f'(x) \times \dots$.
*   Agar $f'(x) \approx 0$ bo‘lsa, butun ko‘paytma nolga aylanadi.
*   **Oqibat:** Vaznlar yangilanmaydi ($w_{new} \approx w_{old}$). Modelning o‘sha qismi "qotib qoladi" va o‘rganishdan to‘xtaydi. Bu muammo ayniqsa chuqur tarmoqlarda kuchli seziladi. Shuning uchun zamonaviy modellarda to‘yinmaydigan funksiyalar (ReLU: $x \to \infty$ da $y \to \infty$) afzal ko‘riladi.

---\n

### 43. Qaysi holatlarda tanh faollashtirish funksiyasini sigmoid funksiyasiga nisbatan afzal ko‘rish mumkinligini izohlab bering.

**Javob:**
Tanh (Giperbolik tangens): $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$.
Qiymatlar oralig‘i: $(-1, 1)$.

**Afzalligi (Zero-centered):**
Tanh funksiyasi **nolga nisbatan simmetrikdir**.
*   **Sigmoid:** Chiqishi $[0, 1]$. Barcha qiymatlar musbat. Bu keyingi qatlam neyronlari uchun "Bias Shift" (siljish) muammosini tug‘diradi. Gradientlar har doim bir xil ishorali bo‘lib qoladi (zig-zag optimizatsiya).
*   **Tanh:** O‘rtacha qiymati $0$ atrofida bo‘ladi. Musbat va manfiy qiymatlar bir-birini muvozanatlaydi. Bu keyingi qatlamdagi hisob-kitoblarni barqarorlashtiradi va optimizatsiya jarayonini (Sigmoidga nisbatan) tezlashtiradi.
*   **Gradient kuchi:** Tanhning hosilasi maksimum $1$ ga teng ($x=0$ da). Sigmoidniki esa maksimum $0.25$. Bu Tanhda gradient yo‘qolishi muammosi Sigmoidga qaraganda 4 barobar kamroq ekanini bildiradi.

**Qo‘llanilishi:** Yashirin qatlamlarda (ayniqsa RNN/LSTM larda) Sigmoiddan ko‘ra Tanh deyarli har doim afzaldir. Sigmoid faqat chiqish qatlamida (ehtimollik uchun) ishlatiladi.

---\n

### 44. Nega chiziqli faollashtirish funksiyasi chuqur modellar qurishga imkon bermaydi? Tushuntiring.

**Javob:**
Agar neyron tarmoqdagi barcha faollashtirish funksiyalari chiziqli ($f(x) = cx$) yoki umuman yo‘q bo‘lsa, butun tarmoq matematik jihatdan **bitta qatlamli chiziqli modelga** (Logistic/Linear Regression) ekvivalent bo‘lib qoladi.

**Isbot mantiqi:**
*   1-qatlam: $y_1 = W_1 \cdot x + b_1$
*   2-qatlam: $y_2 = W_2 \cdot y_1 + b_2 = W_2 \cdot (W_1 \cdot x + b_1) + b_2 = (W_2 W_1) \cdot x + (W_2 b_1 + b_2)$.
*   Bu ifodani yangi $W_{new} = W_2 W_1$ va $b_{new} = W_2 b_1 + b_2$ bilan almashtirish mumkin.
*   Ya’ni, 100 ta qatlam qursangiz ham, ular baribir bitta matritsaga yig‘ilib qoladi.

**Xulosa:**
"Chuqurlik" o‘z ma’nosini yo‘qotadi. Model XOR kabi oddiy chiziqli bo‘lmagan muammolarni, rasmdagi egri chiziqlarni yoki matndagi murakkab ma’nolarni o‘rgana olmaydi. Model "aqlli" bo‘lishi uchun unga albatta "bukilishlar" (non-linearity: ReLU, Sigmoid) kerak.

---\n

### 45. Faollashtirish funksiyasi tanlovi o‘qitish tezligi va barqarorligiga qanday ta’sir ko‘rsatishini izohlang.

**Javob:**
Faollashtirish funksiyasi modelning "dvigateli" kabidir. To‘g‘ri tanlov o‘qitishni soatlab emas, daqiqalab o‘lchanadigan darajaga olib chiqishi mumkin.

1.  **O‘qitish Tezligi (Speed):**
    *   **ReLU:** Hisoblash juda oson (faqat taqqoslash: $x > 0 ?$ ), gradienti $1$ (o‘zgarmas va so‘nmaydi). Bu eng tez konvergensiyani ta’minlaydi.
    *   **Sigmoid/Tanh:** Eksponenta ($e^x$) hisoblash kompyuter uchun og‘ir. Bundan tashqari, kichik gradientlar (vanishing gradient) sababli model juda sekin o‘rganadi (qadamlar maydalashib ketadi).
2.  **Barqarorlik (Stability):**
    *   **Sigmoid:** O‘rtacha qiymati $0.5$ bo‘lgani uchun vaznlarni doimiy ravishda musbat tomonga "surib" yuradi.
    *   **ELU/SELU:** O‘zining manfiy qismi hisobiga ma’lumotlar taqsimotini barqaror (o‘rtacha 0, dispersiya 1) ushlab turadi. Bu "Self-Normalizing" xususiyati bo‘lib, Batch Normalization ishlatmasdan ham juda chuqur tarmoqlarni barqaror o‘qitish imkonini beradi.

---\n

### 46. To‘g‘ri tarqaluvchi neyron tarmoq (feedforward neural network) tushunchasini ta’riflab bering.

**Javob:**
To‘g‘ri tarqaluvchi neyron tarmoq (Feedforward Neural Network - FNN) — bu sun’iy neyron tarmoqlarning eng birinchi va eng sodda turidir. Unda axborot oqimi faqat **bir yo‘nalishda** (oldindan orqaga) harakatlanadi.

**Xususiyatlari:**
*   **Yo‘nalish:** Kirish qatlami $\to$ Yashirin qatlamlar $\to$ Chiqish qatlami.
*   **Sirtmoqlar yo‘q:** Tarmoqda hech qanday aylanma yo‘llar (cycles) yoki orqaga qaytish (feedback loops) mavjud emas. Signal hech qachon o‘zi o‘tgan neyronga qaytib kelmaydi.
*   **Xotirasizlik:** Chiqish faqat ayni damdagi kirishga bog‘liq. Oldingi rasmlar yoki holatlar eslab qolinmaydi.
*   **Misollar:** MLP (Multilayer Perceptron), CNN (Convolutional Neural Network).
*   **Taqqoslash:** Bu RNN (Rekurrent tarmoqlar) ning teskarisidir. RNNda ma’lumot aylanib o‘ziga qaytishi va xotira hosil qilishi mumkin.

---\n

### 47. Nega ba’zi vazifalarda ko‘p qatlamli chuqur arxitektura talab qilinadi? Sabablarini tushuntirib bering.

**Javob:**
Dunyo va undagi ma’lumotlar **ierarxik** (pog‘onali) tuzilishga ega. Chuqur arxitektura ana shu tabiiy tuzilishni aks ettiradi.

1.  **Kompozitsionallik (Compositionality):** Murakkab narsalar oddiy narsalarning yig‘indisidan iborat.
    *   **Matn:** Harflar $\to$ So‘zlar $\to$ Gaplar $\to$ Ma’no.
    *   **Rasm:** Piksellar $\to$ Chiziqlar $\to$ Shakllar $\to$ Ob’ektlar $\to$ Sahnasi.
    Chuqur model har bir qatlamda ushbu pog‘onalarning bittasini bosqichma-bosqich o‘rganadi. Sayoz model esa bularning hammasini "bir urinishda" o‘rganishi kerak, bu esa o‘ta murakkab va samarasiz.
2.  **Samaradorlik:** Matematik jihatdan isbotlangan: ba’zi murakkab funksiyalarni ifodalash uchun chuqur tarmoqqa sayoz tarmoqqa qaraganda **eksponentsial darajada kamroq** neyron kerak bo‘ladi.
3.  **Abstraksiya:** Chuqur qatlamlar kirish ma’lumotlarining "invariant" xususiyatlarini (masalan, yuzning yoritilishi yoki burchagi o‘zgarsa ham o‘zgarmaydigan o‘zagi) ajratib olishga qodir.

---\n

### 48. Yashirin (hidden) qatlamlarning vazifasi va chuqur tarmoqlardagi ahamiyatini izohlab bering.

**Javob:**
**Vazifasi:**
Yashirin qatlamlar kirish ma’lumotlarini **transformatsiya** qilish (shaklini o‘zgartirish) bilan shug‘ullanadi. Ularning maqsadi ma’lumotni shunday yangi fazoga (Latent Space) o‘tkazishki, u yerda javobni topish oson bo‘lsin.

**Ahamiyati (Manifold Hypothesis):**
*   Tasavvur qiling, qog‘ozda qizil va ko‘k nuqtalar chalkashib yotibdi va ularni to‘g‘ri chiziq bilan ajratib bo‘lmaydi.
*   Yashirin qatlamlar bu "qog‘ozni" buklab, cho‘zib, g‘ijimlab, shunday holatga keltiradiki, natijada qizil va ko‘k nuqtalar ikki xil tekislikda ajralib qoladi.
*   Har bir yashirin qatlam ma’lumotlar fazosini (manifold) biroz o‘zgartiradi. Chuqur tarmoqlarda bu jarayon bosqichma-bosqich bajarilib, oxirida eng murakkab chalkashliklar ham yechiladi ("linearly separable" bo‘ladi). Yashirin qatlamlarsiz model faqat eng oddiy chiziqli muammolarni yecha olgan bo‘lardi.

---\n

### 49. Nega chuqur neyron tarmoqlar universal approksimatorlar sifatida qaraladi? Tushuntirib bering.

**Javob:**
Bu nazariy matematik teoremaga (Universal Approximation Theorem) asoslanadi.

**Mazmuni:**
Teorema (Cybenko, 1989) shuni aytadiki:
Agar yetarli miqdorda neyronlar bo‘lsa va hech bo‘lmaganda bitta nochiziqli faollashtirish funksiyasi (Sigmoid, ReLU) ishlatilsa, **hatto bitta yashirin qatlamga ega** neyron tarmoq ham har qanday uzluksiz funksiyani (ixtiyoriy aniqlikda) taqlid qilishi (approksimatsiya) mumkin.

**Amaliyotda:**
Garchi bitta qatlam nazariy jihatdan yetarli bo‘lsa-da, amalda bu samarasiz (juda ko‘p, milliardlab neyron kerak bo‘ladi va o‘qitish qiyinlashadi). Chuqur tarmoqlar esa xuddi shu qobiliyatni yanada samaraliroq (kamroq resurs bilan, qatlamlar hisobiga) amalga oshiradi. Lekin asosiy g‘oya o‘zgarmaydi: neyron tarmoqlar — bu juda moslashuvchan "plastilin". Ular har qanday shaklga (funksiyaga) kira oladi, xoh u ovoz to‘lqini bo‘lsin, xoh birja narxlari, xoh inson nutqi.

---\n

### 50. ResNet arxitekturasidagi qoldiq ulanishlar (residual connections) g‘oyasini va ularning ahamiyatini izohlab bering.

**Javob:**
ResNet (Residual Network, 2015) inqilobi 100 dan ortiq qatlamli modellarni o‘qitish imkonini berdi.

**G‘oya:**
Oddiy tarmoqda ma’lumot qatlamdan qatlamga o‘tadi: $y = f(x)$.
ResNetda esa ma’lumot **aylanma yo‘l** orqali ham o‘tadi va funksiya natijasiga qo‘shiladi: $y = f(x) + x$. Bu "Skip Connection" yoki "Residual Connection" deb ataladi.

**Ahamiyati:**
1.  **Gradient "shossesi" (Gradient Highway):** Orqaga tarqalish (backpropagation) paytida gradient $f(x)$ funksiyasi ichida (vaznlar, aktivatsiyalar sababli) kichrayib, yo‘qolib ketishi mumkin. Lekin $+x$ yo‘li orqali gradient hech qanday to‘siqsiz, $1$ ga ko‘paytirilgan holda to‘g‘ridan-to‘g‘ri orqaga oqib o‘tadi. Bu "Vanishing gradient" muammosini deyarli to‘liq hal qiladi.
2.  **Identiklikni o‘rganish:** Agar qo‘shimcha qatlamlar modelga foyda bermasa, model shunchaki $f(x)=0$ qilib qo‘yadi va $y=x$ (kirishni o‘zgarishsiz o‘tkazish) holatiga keladi. Bu shuni anglatadiki, chuqur model hech qachon sayoz modeldan *yomonroq* ishlamaydi, faqat yaxshiroq ishlashi mumkin.