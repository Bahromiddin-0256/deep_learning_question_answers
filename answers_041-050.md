# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (41-50)

### 41. Swish faollashtirish funksiyasi nima va nega u ReLU ning yaxshilangan varianti sifatida ko‘rilishini izohlang.

**Javob:**
**Tushuncha:**
Swish — bu Google Brain jamoasi tomonidan (avtomatik qidiruv orqali) topilgan faollashtirish funksiyasi.
Formula: $f(x) = x \cdot \text{sigmoid}(x) = \frac{x}{1 + e^{-x}}$.

**Nega u ReLU dan yaxshiroq?**
1.  **Silliqlik (Smoothness):** ReLU $x=0$ nuqtasida keskin burchakka ega (hosilasi uziladi). Swish esa silliq egri chiziqdir. Bu optimizatsiya jarayonida gradientning tekisroq o‘zgarishiga yordam beradi va modelni chuqur minimumga tushishini osonlashtiradi.
2.  **Monoton emaslik:** Swish funksiyasi manfiy sohada biroz pasayib, keyin yana 0 ga qaytadi (kichik "chuqurcha"si bor). Bu xususiyat ba’zi murakkab ma’lumotlar strukturasini o‘rganishda foydali bo‘lishi mumkinligi aniqlangan.
3.  **Dying ReLU muammosi yo‘q:** Manfiy qiymatlarda u nolga teng emas (biroz manfiy qiymat va gradient bor), shuning uchun neyronlar butunlay o‘lib qolmaydi.
Tadqiqotlar shuni ko‘rsatadiki, juda chuqur tarmoqlarda Swish oddiy ReLU ga qaraganda 0.5-1% yuqori aniqlik berishi mumkin.

---

### 42. To‘yingan (saturating) faollashtirish funksiyalari muammosini tushuntirib bering.

**Javob:**
**Tushuncha:**
To‘yingan funksiya — bu kirish qiymati ($x$) oshib yoki kamayib ketganda, chiqish qiymati ma’lum bir songa (limitga) yopishib qoladigan funksiyadir.
Misol: Sigmoid ($x \to \infty$ da $y \to 1$) va Tanh ($x \to \infty$ da $y \to 1$).

**Muammo:**
Funksiya to‘yingan hududda (grafik tekislashgan joyda) uning **hosilasi (gradienti) nolga teng** bo‘ladi.
*   Backpropagation paytida biz gradientni vaznlarga ko‘paytiramiz.
*   Agar gradient 0 bo‘lsa, ko‘paytma ham 0 bo‘ladi.
*   Natijada vaznlar yangilanmaydi. Model "qotib qoladi" va o‘rganishdan to‘xtaydi.
Bu muammo ayniqsa chuqur tarmoqlarda "Vanishing Gradient"ga sabab bo‘lishi bilan xavflidir. Shuning uchun zamonaviy modellarda to‘yinmaydigan funksiyalar (ReLU va uning turlari) afzal ko‘riladi.

---

### 43. Qaysi holatlarda tanh faollashtirish funksiyasini sigmoid funksiyasiga nisbatan afzal ko‘rish mumkinligini izohlab bering.

**Javob:**
Tanh (Giperbolik tangens): $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$.
U qiymatni $(-1, 1)$ oraliqqa siqadi.

**Afzalligi:**
Tanh funksiyasi **nolga nisbatan simmetrikdir (Zero-centered)**.
*   Sigmoidning chiqishi $[0, 1]$ (hammasi musbat). Bu keyingi qatlam uchun "bias shift" (siljish) muammosini tug‘diradi.
*   Tanhning o‘rtacha qiymati 0 atrofida bo‘ladi. Bu keyingi qatlamdagi hisob-kitoblarni barqarorlashtiradi va optimizatsiya jarayonini (Sigmoidga nisbatan) tezlashtiradi.
*   Shuningdek, Tanhning hosilasi (maksimum 1) Sigmoidnikidan (maksimum 0.25) kattaroq. Bu gradient yo‘qolishini biroz bo‘lsa-da sekinlashtiradi.

Shu sababli, yashirin qatlamlarda (ayniqsa RNN/LSTM larda) Sigmoiddan ko‘ra Tanh ko‘proq ishlatiladi.

---

### 44. Nega chiziqli faollashtirish funksiyasi chuqur modellar qurishga imkon bermaydi? Tushuntiring.

**Javob:**
Agar neyron tarmoqdagi barcha faollashtirish funksiyalari chiziqli ($f(x) = cx$) bo‘lsa, butun tarmoq matematik jihatdan **bitta qatlamli chiziqli modelga** ekvivalent bo‘lib qoladi.

**Isbot mantiqi:**
*   1-qatlam: $y_1 = W_1 \cdot x$
*   2-qatlam: $y_2 = W_2 \cdot y_1 = W_2 \cdot (W_1 \cdot x) = (W_2 \cdot W_1) \cdot x$.
*   $W_{yangi} = W_2 \cdot W_1$ desak, bu shunchaki bitta matritsa bo‘lib qoladi.
Qatlamlar soni 100 ta bo‘lsa ham, ular bir-biriga ko‘paytirilib, bitta oddiy chiziqli transformatsiyaga aylanib ketadi.

Natijada, "chuqurlik" o‘z ma’nosini yo‘qotadi. Model XOR kabi oddiy chiziqli bo‘lmagan muammolarni ham yecha olmaydi. Model "aqlli" bo‘lishi uchun unga albatta "bukilishlar" (non-linearity) kerak.

---

### 45. Faollashtirish funksiyasi tanlovi o‘qitish tezligi va barqarorligiga qanday ta’sir ko‘rsatishini izohlang.

**Javob:**
To‘g‘ri tanlangan funksiya modelni soatlab emas, daqiqalab o‘qitishga yordam beradi.

1.  **Tezlik:**
    *   **ReLU:** Hisoblash juda oson (faqat taqqoslash), gradienti 1 (o‘zgarmas). Bu eng tez konvergensiyani ta’minlaydi.
    *   **Sigmoid/Tanh:** Eksponenta hisoblash sekin, gradient kichik (vanishing gradient), bu esa o‘qishni juda sekinlashtiradi.
2.  **Barqarorlik:**
    *   **Sigmoid:** O‘rtacha qiymati 0.5 bo‘lgani uchun vaznlarni doimiy ravishda bir tomonga "surib" yuradi (zigzag gradient).
    *   **ELU/SELU:** O‘zining manfiy qismi hisobiga ma’lumotlar taqsimotini barqaror (o‘rtacha 0, dispersiya 1) ushlab turadi. Bu Batch Normalization ishlatmasdan ham modelni barqaror o‘qitish imkonini beradi.

---

### 46. To‘g‘ri tarqaluvchi neyron tarmoq (feedforward neural network) tushunchasini ta’riflab bering.

**Javob:**
To‘g‘ri tarqaluvchi neyron tarmoq (Feedforward Neural Network - FNN) — bu axborot oqimi faqat **bir yo‘nalishda** (oldindan orqaga) harakatlanadigan tarmoq turidir.

**Xususiyatlari:**
*   Ma’lumot Kirish qatlamidan kiradi, Yashirin qatlamlardan o‘tadi va Chiqish qatlamidan chiqadi.
*   Hech qanday **sirtmoqlar (loops)**, aylanma yo‘llar yoki orqaga qaytish (feedback) yo‘q. (Orqaga qaytish faqat o‘qitish paytida xatoni to‘g‘rilash uchun bo‘ladi, signal uzatishda emas).
*   Chiqish faqat joriy kirishga bog‘liq (xotirasi yo‘q).
*   Misollar: MLP (Multilayer Perceptron), CNN (Convolutional Neural Network).
Bu RNN (Rekurrent tarmoqlar) ning teskarisidir, chunki RNNda ma’lumot aylanib o‘ziga qaytishi mumkin.

---

### 47. Nega ba’zi vazifalarda ko‘p qatlamli chuqur arxitektura talab qilinadi? Sabablarini tushuntirib bering.

**Javob:**
Dunyo ierarxik (pog‘onali) tuzilishga ega. Chuqur arxitektura ana shu tuzilishni aks ettiradi.

1.  **Kompozitsionallik:** Murakkab narsalar oddiy narsalarning yig‘indisidan iborat.
    *   Matn: Harflar -> So‘zlar -> Gaplar -> Ma’no.
    *   Rasm: Piksellar -> Chiziqlar -> Shakllar -> Ob’ektlar -> Sahnasi.
Chuqur model har bir qatlamda ushbu pog‘onalarning bittasini o‘rganadi. Sayoz model esa bularning hammasini bir urinishda o‘rganishi kerak, bu esa juda qiyin.
2.  **Samaradorlik:** Funksiyani ifodalash uchun chuqur tarmoqqa sayoz tarmoqqa qaraganda **eksponentsial darajada kamroq** neyron kerak bo‘ladi.
3.  **Abstraksiya:** Chuqur qatlamlar kirish ma’lumotlarining "invariant" xususiyatlarini (masalan, yuzning yoritilishi yoki burchagi o‘zgarsa ham o‘zgarmaydigan o‘zagi) ajratib olishga qodir.

---

### 48. Yashirin (hidden) qatlamlarning vazifasi va chuqur tarmoqlardagi ahamiyatini izohlab bering.

**Javob:**
**Vazifasi:**
Yashirin qatlamlar kirish ma’lumotlarini **transformatsiya** qilish (shaklini o‘zgartirish) bilan shug‘ullanadi. Ularning maqsadi ma’lumotni shunday ko‘rinishga keltirishki, oxirgi qatlam uni osongina ajrata olsin.

**Ahamiyati:**
*   Tasavvur qiling, qog‘ozda qizil va ko‘k nuqtalar aralashib yotibdi va ularni to‘g‘ri chiziq bilan ajratib bo‘lmaydi.
*   Yashirin qatlamlar bu "qog‘ozni" buklab, cho‘zib, g‘ijimlab, shunday holatga keltiradiki, natijada qizil va ko‘k nuqtalar ikki xil tekislikda ajralib qoladi.
*   Har bir yashirin qatlam ma’lumotlar fazosini (manifold) o‘zgartiradi. Chuqur tarmoqlarda bu jarayon bosqichma-bosqich bajarilib, oxirida eng murakkab chalkashliklar ham yechiladi. Yashirin qatlamlarsiz model faqat chiziqli muammolarni yecha olgan bo‘lardi.

---

### 49. Nega chuqur neyron tarmoqlar universal approksimatorlar sifatida qaraladi? Tushuntirib bering.

**Javob:**
Bu nazariy matematik teoremaga (Universal Approximation Theorem) asoslanadi.

**Mazmuni:**
Teorema shuni aytadiki: Agar yetarli miqdorda neyronlar bo‘lsa va nochiziqli faollashtirish funksiyasi (Sigmoid, ReLU) ishlatilsa, **hatto bitta yashirin qatlamga ega** neyron tarmoq ham har qanday uzluksiz funksiyani istalgan aniqlikda taqlid qilishi (approksimatsiya) mumkin.

**Amaliyotda:**
Garchi bitta qatlam nazariy jihatdan yetarli bo‘lsa-da, amalda bu samarasiz (juda ko‘p neyron kerak). Chuqur tarmoqlar esa xuddi shu qobiliyatni yanada samaraliroq (kamroq resurs bilan) amalga oshiradi. Lekin asosiy g‘oya o‘zgarmaydi: neyron tarmoqlar — bu juda moslashuvchan "plastilin". Ular har qanday shaklga (funksiyaga) kira oladi, xoh u ovoz to‘lqini bo‘lsin, xoh birja narxlari.

---

### 50. ResNet arxitekturasidagi qoldiq ulanishlar (residual connections) g‘oyasini va ularning ahamiyatini izohlab bering.

**Javob:**
ResNet (Residual Network) 2015-yilda paydo bo‘lib, 100 dan ortiq qatlamli modellarni o‘qitish imkonini berdi.

**G‘oya:**
Oddiy tarmoqda ma’lumot qatlamdan qatlamga o‘tadi: $y = f(x)$.
ResNetda esa ma’lumot **aylanma yo‘l** orqali ham o‘tadi va funksiya natijasiga qo‘shiladi: $y = f(x) + x$. Bu "Skip Connection" yoki "Residual Connection" deb ataladi.

**Ahamiyati:**
1.  **Gradient "shossesi":** Orqaga tarqalish (backpropagation) paytida gradient $f(x)$ funksiyasi ichida yo‘qolib ketishi mumkin, lekin $+x$ yo‘li orqali u hech qanday to‘siqsiz orqaga oqib o‘tadi. Bu "Vanishing gradient" muammosini deyarli to‘liq hal qiladi.
2.  **Identiklikni o‘rganish:** Agar qo‘shimcha qatlamlar foydasiz bo‘lsa, model shunchaki $f(x)=0$ qilib qo‘yadi va $y=x$ (kirishni o‘zgarishsiz o‘tkazish) holatiga keladi. Bu shuni anglatadiki, chuqur model hech qachon sayoz modeldan *yomonroq* ishlamaydi, faqat yaxshiroq ishlashi mumkin.
