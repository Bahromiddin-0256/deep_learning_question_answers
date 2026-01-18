# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (21-30)

### 21. Adaptiv optimizatorlar (masalan, Adam, RMSProp) tushunchasini va ularning umumiy ustun tomonlarini izohlab bering.

**Javob:**
**Tushuncha:**
Adaptiv optimizatorlar — bu modelni o‘qitish jarayonida "O‘qitish tezligi" (Learning Rate) parametrini avtomatik ravishda o‘zgartirib boradigan algoritmlardir. Klassik SGD usulida Learning Rate (masalan, 0.01) barcha parametrlar uchun bir xil va o‘zgarmas bo‘ladi. Adaptiv usullarda esa tarmoqdagi har bir vazn (weight) o‘zining shaxsiy o‘qish tezligiga ega bo‘ladi va bu tezlik vaqt o‘tishi bilan o‘zgarib turadi.

*   **Adagrad:** Kam uchraydigan parametrlar uchun tezlikni oshiradi, ko‘p uchraydiganlar uchun kamaytiradi.
*   **RMSProp:** Adagradning kamchiligini (tezlikning juda tez pasayib ketishini) to‘g‘irlaydi va so‘nggi gradientlarning o‘rtacha kvadratiga qarab tezlikni moslaydi.
*   **Adam:** RMSProp va Momentum g‘oyalarini birlashtiradi.

**Ustun tomonlari:**
1.  **Tezkor konvergensiya:** Ular modelni optimal nuqtaga SGDga qaraganda ancha tezroq olib keladi, chunki ular "tekis" yo‘nalishlarda katta qadamlar, "tik" va havfli yo‘nalishlarda kichik qadamlar tashlaydi.
2.  **Siyrak (Sparse) ma’lumotlar bilan ishlash:** Tabiiy tilni qayta ishlashda (NLP) ba’zi so‘zlar juda kam uchraydi. SGD ularni o‘rganishga ulgurmay qolishi mumkin. Adaptiv optimizatorlar esa bunday kam uchraydigan belgilarni ko‘rganda "katta sakrash" qilib, ularni darhol o‘rganib oladi.
3.  **Giperparametrlarni sozlash:** SGD uchun eng mukammal Learning Rate-ni topish qiyin. Adaptiv metodlar esa boshlang‘ich qiymatga nisbatan kamroq ta’sirchan bo‘lib, standart sozlamalar (default settings) bilan ham yaxshi ishlayveradi.

---

### 22. Learning rate decay (o‘qitish tezligini bosqichma-bosqich kamaytirish) tushunchasini tushuntirib bering.

**Javob:**
Learning Rate Decay (yoki Learning Rate Scheduling) — bu o‘qitish jarayoni davomida o‘qitish tezligini sun’iy ravishda kamaytirib borish strategiyasidir.

**Mantiqiy asosi:**
*   **Boshlanishda:** Model hali hech narsani bilmaydi, xatolik katta. Bizga global minimumni tezroq topish uchun **katta qadamlar** kerak. Katta tezlik modelga lokal minimumlardan sakrab o‘tishga yordam beradi.
*   **Oxirida:** Model global minimumga yaqinlashganda, katta qadamlar xavfli bo‘lib qoladi. Model minimum atrofida u yoqdan-bu yoqqa sakrab yurishi (oscillate) va aniq eng past nuqtaga tusha olmasligi mumkin. Shuning uchun, jarayon oxirida **juda kichik qadamlar** bilan siljish kerak.

**Usullari:**

1.  **Step Decay:**

    $$\eta_t = \eta_0 \cdot \gamma^{\lfloor \frac{t}{N} \rfloor}$$

    bu yerda $\eta_0$ - boshlang'ich tezlik, $\gamma$ - kamayish koeffitsienti (masalan, 0.1), $N$ - har necha epoxada kamaytirish.

2.  **Exponential Decay:**

    $$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

    bu yerda $\lambda$ - kamayish tezligi.

3.  **Cosine Decay:**

    $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_0 - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

    bu yerda $T$ - umumiy o'qitish qadamlari soni.

Bu usul modelning aniqligini (accuracy) sezilarli darajada oshirishga yordam beradi.

---

### 23. Gradientlarni cheklash (gradient clipping) nima va qaysi holatlarda qo‘llanilishini izohlab bering.

**Javob:**
**Tushuncha:**
Gradient Clipping — bu optimizatsiya jarayonida gradient vektorining qiymati yoki normasi belgilangan chegaradan (threshold) oshib ketsa, uni majburan "qirqib" tashlash (kichraytirish) usulidir.
Masalan, agar chegara 5.0 bo‘lsa va hisoblangan gradient 100.0 chiqsa, biz uni yo‘nalishini saqlagan holda 5.0 ga tenglashtirib qo‘yamiz.

**Qo‘llanilish holatlari:**
Bu usul asosan **"Exploding Gradients" (Portlab ketuvchi gradientlar)** muammosiga qarshi kurashish uchun ishlatiladi.
1.  **Rekurrent Neyron Tarmoqlar (RNN/LSTM):** RNNlarda vaqt bo‘yicha orqaga qaytish (BPTT) sababli gradientlar ko‘pincha juda katta qiymatlarga yetib, modelni buzib yuboradi (vaznlar NaN bo‘lib qoladi). Clipping bu jarayonni jilovlaydi.
2.  **Juda chuqur tarmoqlar:** Ba’zida juda chuqur tarmoqlarda ham notekis landshaft (tik qiyaliklar) sababli gradient sakrab ketishi mumkin.

Bu xuddi tog‘dan tushayotganda tezlikni cheklagich (tormoz) qo‘yishga o‘xshaydi: qanchalik tik jarlik bo‘lmasin, biz ma’lum tezlikdan oshmaymiz, natijada qulab tushishdan saqlanamiz.

---

### 24. Nega chuqur modellarning optimallashtirish jarayoni sayoz modellarnikiga qaraganda murakkabroq bo‘ladi? Tushuntirib bering.

**Javob:**
Sayoz modellar (masalan, Logistik Regressiya yoki SVM) odatda **qavariq (convex)** xatolik funksiyasiga ega. Bu piyolaga o‘xshaydi: qayerdan boshlasangiz ham, pastga qarab yuraversangiz, albatta yagona eng chuqur nuqtaga (global minimumga) yetib borasiz.

Chuqur modellar esa **qavariq bo‘lmagan (non-convex)** funksiyalardir. Ularning xatolik landshafti (landscape) xuddi tog‘ tizmalariga o‘xshaydi:
1.  **Ko‘plab lokal minimumlar:** Model haqiqiy eng yaxshi yechimni topmasdan, kichikroq chuqurlikda qolib ketishi mumkin.
2.  **Egar nuqtalar (Saddle Points):** Bir tomondan qaraganda minimum, boshqa tomondan maksimum bo‘lgan tekis joylar. Bu yerda gradient nolga teng bo‘lib, o‘qitish to‘xtab qolishi mumkin.
3.  **Tekisliklar (Plateaus):** Gradient deyarli nolga teng bo‘lgan ulkan tekis maydonlar. Bu yerda model juda sekin o‘qiydi.
4.  **Tartibsizlik (Chaos):** Parametrlar soni millionlab bo‘lgani uchun, fazo juda ko‘p o‘lchamli bo‘ladi va inson tasavvuriga sig‘maydigan murakkab geometriyaga ega bo‘ladi.

---

### 25. Egar nuqta (saddle point) tushunchasini va uning o‘qitish jarayoniga ta’sirini izohlab bering.

**Javob:**
**Tushuncha:**
Egar nuqta (otning egariga o‘xshagani uchun shunday ataladi) — bu funksiyaning gradienti nolga teng bo‘lgan, lekin u minimum ham, maksimum ham bo‘lmagan nuqtasidir.
Tasavvur qiling: oldinga va orqaga qarasangiz yo‘l yuqoriga ketyapti (xuddi minimumdek), lekin chapga va o‘ngga qarasangiz yo‘l pastga tushib ketyapti (xuddi maksimumdek).

**O‘qitishga ta’siri:**
Ko‘p o‘lchamli fazolarda (chuqur o‘qitishda) lokal minimumlardan ko‘ra egar nuqtalar **juda ko‘p** uchraydi.
*   **Muammo:** Egar nuqtada gradient nolga teng (yoki juda kichik). Optimizator (ayniqsa oddiy SGD) gradient nol ekanini ko‘rib, "men manzilga yetdim" deb o‘ylaydi va to‘xtab qoladi yoki shu tekislikdan chiqib ketishi uchun juda ko‘p vaqt sarflaydi.
*   Bu chuqur tarmoqlarni o‘qitishdagi asosiy to‘siqlardan biridir. Shuning uchun Momentum yoki shovqinli SGD kabi usullar kerak — ular inersiya hisobiga bu "soxta to‘xtash" nuqtalaridan o‘tib ketishga yordam beradi.

---

### 26. Chuqur o‘qitishda nima uchun L1 va L2 muntazamlashtirish (regularizatsiya) usullari qo‘llaniladi? Tushuntirib bering.

**Javob:**
Muntazamlashtirish (Regularization) — bu modelning **overfitting** (haddan tashqari moslashish) bo‘lishini oldini olish uchun ishlatiladigan texnika. U modelga "vaznlarni iloji boricha kichik ushlab tur" degan qo‘shimcha talabni yuklaydi.

**L2 Regularization (Ridge / Weight Decay):**

Xatolik funksiyasi:

$$L_{total} = L_{original} + \lambda \sum_{i=1}^{n} w_i^2$$

Gradient yangilash:

$$w = w - \eta(\nabla L + 2\lambda w) = w(1 - 2\eta\lambda) - \eta\nabla L$$

*   **Maqsadi:** Bitta ham vaznning o'ta katta bo'lib ketishiga yo'l qo'ymaslik. Katta vaznlar modelni beqaror qiladi va kirishdagi kichik shovqinga qattiq reaksiya berishga olib keladi. L2 hamma vaznlarni bir tekisda kichraytiradi.

**L1 Regularization (Lasso):**

$$L_{total} = L_{original} + \lambda \sum_{i=1}^{n} |w_i|$$

*   **Maqsadi:** Bu usul ba'zi keraksiz vaznlarni **butunlay nolga** aylantirib yuboradi. Natijada model "siyrak" (sparse) bo'lib qoladi, ya'ni u faqat eng muhim xususiyatlarnigina tanlab oladi. Bu xususiyatlarni tanlash (feature selection) uchun juda foydali.

**Vizual taqqoslash:**
```
L2: vaznlarni kichraytiradi      L1: vaznlarni nolga aylantiradi
    w → 0.1w → 0.01w                  w → 0.5w → 0 (aniq nol)
```

---

### 27. Dropout usulining mazmunini va uning overfitting ni kamaytirishdagi rolini izohlab bering.

**Javob:**
**Mazmuni:**
Dropout — bu o‘qitish jarayonida (faqat training paytida) neyronlarni tasodifiy ravishda "o‘chirib qo‘yish" usulidir. Har bir iteratsiyada, masalan, neyronlarning 50% i tasodifiy tanlanadi va ularning qiymati nolga tenglashtiriladi. Ular o‘sha qadamda qatnashmaydi. Keyingi qadamda esa boshqa 50% neyronlar o‘chiriladi.

**Overfittingni kamaytirishdagi roli:**
1.  **Bog‘liqlikni yo‘qotish:** Odatda neyronlar bir-biriga qattiq bog‘lanib qolishi (co-adaptation) mumkin. Biri xato qilsa, ikkinchisi uni tuzatishga urinadi. Dropout neyronlarni "yolg‘iz ishlashga" majbur qiladi. Har bir neyron o‘ziga mustaqil ravishda foydali belgilarni topishi kerak bo‘ladi, chunki u qo‘shnisiga tayana olmaydi.
2.  **Ansambl effekti:** Dropout orqali biz aslida bitta katta tarmoqni emas, balki minglab turli xil kichik tarmoqlarni o‘qitgandek bo‘lamiz. Test paytida barcha neyronlar yoqiladi, bu esa o‘sha minglab kichik modellarning "o‘rtacha" fikrini olishdek kuchli natija beradi.

---

### 28. Early stopping (erta to‘xtatish) nima va u qaysi maqsadda qo‘llanilishini tushuntiring.

**Javob:**
**Tushuncha:**
Early Stopping — bu modelni o‘qitishni belgilangan epoxalar soni (masalan, 1000 epoxa) tugamasdan oldin to‘xtatish strategiyasidir.

**Maqsadi va Ishlash tartibi:**

O'qitish paytida biz ikkita grafikni kuzatib boramiz:
1.  **Training Loss:** Bu doimiy ravishda kamayib boradi (model yodlayapti).
2.  **Validation Loss:** Bu modelning haqiqiy imtihondagi bahosi.

**Early Stopping Vizualizatsiyasi:**
```
Loss
  ^
  |  Training Loss
  |    ╲
  |     ╲_______________
  |
  |  Validation Loss    ╱ overfitting boshlandi
  |    ╲              ╱
  |     ╲___________╱
  |                 ↑
  |           Bu yerda to'xtat!
  └────────────────────────────> Epoxalar
```

Boshida ikkalasi ham kamayadi. Lekin ma'lum bir nuqtadan keyin Training Loss kamayishda davom etsa ham, Validation Loss **oshishni** boshlaydi. Bu nuqta **overfitting** boshlanganini bildiradi: model endi o'rganmayapti, balki yodlayapti.
Early Stopping aynan shu burilish nuqtasini aniqlaydi va o'qitishni darhol to'xtatib, o'sha paytdagi eng yaxshi modelni saqlab qoladi. Bu resursni tejaydi va eng yuqori aniqlikni ta'minlaydi.

---

### 29. Batch normalization va layer normalization usullarining farqlarini tushuntirib bering.

**Javob:**
Ikkala usul ham tarmoq ichidagi qiymatlarni (aktivatsiyalarni) standartlashtirish (o‘rtacha qiymatni 0 ga, dispersiyani 1 ga keltirish) uchun xizmat qiladi, lekin ular buni turlicha bajaradi.

1.  **Batch Normalization (Batch Norm):**
    *   **Yo'nalishi:** Normalizatsiya **batch (guruh)** bo'ylab amalga oshiriladi.

    **Formula:**

    $$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$ (batch o'rtachasi)

    $$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$ (batch dispersiyasi)

    $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$ (normalizatsiya)

    $$y_i = \gamma \hat{x}_i + \beta$$ (masshtablash va siljish)

    bu yerda $m$ - batch hajmi, $\gamma$ va $\beta$ - o'rganiladigan parametrlar.

    *   **Qo'llanilishi:** Asosan Kompyuterni Ko'rish (CNN) sohasida standart hisoblanadi.
    *   **Kamchiligi:** Kichik batchlar bilan (masalan, 2 yoki 4 ta rasm) yaxshi ishlamaydi va RNNlarda qo'llash qiyin.

2.  **Layer Normalization (Layer Norm):**
    *   **Yo'nalishi:** Normalizatsiya **har bir namuna (sample)** uchun alohida bajariladi.
    *   U bitta rasmning (yoki matnning) o'zidagi barcha neyronlar qiymatining o'rtachasini oladi. Boshqa namunalarga bog'liq emas.
    *   **Qo'llanilishi:** Asosan Tabiiy Tilni Qayta Ishlash (NLP), Transformerlar va RNNlarda ishlatiladi. Batch o'lchamiga bog'liq emas.

**Vizual farq:**
```
Batch Norm:                  Layer Norm:
[Sample 1] [Sample 2] ...    [Sample 1]
   ↓ ↓ ↓     ↓ ↓ ↓              ← → →
Batch bo'ylab normalize      Sample ichida normalize
```

---

### 30. Nega normalizatsiya (batch/layer norm) o‘qitish jarayonini tezlashtirishi mumkinligini izohlab bering.

**Javob:**
Normalizatsiya (ayniqsa Batch Norm) qo‘shilganda model 10-20 barobar tezroq o‘qishi (konvergensiya qilishi) mumkin. Buning sabablari:

1.  **Internal Covariate Shift:** Har bir qatlam o‘zgarishi bilan keyingi qatlamga boradigan ma’lumotlar taqsimoti o‘zgarib turadi. Keyingi qatlam har safar yangi qoidalarga moslashishga majbur bo‘ladi. Normalizatsiya bu taqsimotni barqaror (o‘rtachasi 0, tarqoqligi 1) ushlab turadi, natijada har bir qatlam mustaqil va tezroq o‘rganadi.
2.  **Katta Learning Rate:** Normalizatsiya bo‘lmasa, katta learning rate modelni "portlatib" yuborishi mumkin. Normalizatsiya gradientlarni jilovlab turadi, bu esa bizga katta tezlikda o‘qitish imkonini beradi.
3.  **Loss Landscape Smoothing:** Tadqiqotlar shuni ko‘rsatadiki, Batch Norm xatolik funksiyasi yuzasini "silliqlaydi". O‘nqir-cho‘nqir tog‘ yo‘li tekis avtobanga aylanadi. Bu optimizatorga adashmasdan va to‘siqlarga uchramasdan to‘g‘ridan-to‘g‘ri minimumga borish imkonini beradi.