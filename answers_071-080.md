# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (71-80)

### 71. AlexNet arxitekturasining asosiy g‘oyalari va chuqur o‘qitish tarixidagi ahamiyatini izohlang.

**Javob:**
**Tarixiy ahamiyati:**
2012-yilgacha kompyuterni ko‘rish (Computer Vision) sohasida an’anaviy algoritmlar hukmronlik qilardi. AlexNet (Alex Krizhevsky tomonidan yaratilgan) ImageNet musobaqasida xatolikni 26% dan 15.3% gacha keskin tushirib, inqilob qildi. Bu voqea "Deep Learning Era"sining rasmiy boshlanishi hisoblanadi.

**Asosiy g‘oyalari va yangiliklari:**
1.  **ReLU (Rectified Linear Unit):** Sigmoid yoki Tanh o‘rniga birinchi marta ReLU ishlatildi. Bu o‘qitish tezligini 6 barobar oshirdi va chuqur tarmoqlarda gradient yo‘qolishi muammosini kamaytirdi.
2.  **GPU dan foydalanish:** Model juda katta bo‘lgani uchun (60 mln parametr), u o‘sha paytdagi ikkita GTX 580 videokartasida parallel o‘qitildi.
3.  **Dropout:** Overfittingga qarshi kurashish uchun Dropout (0.5) texnikasi ilk bor samarali qo‘llanildi.
4.  **Data Augmentation:** Rasmlarni kesish, ko‘zgudek akslantirish va rangini o‘zgartirish orqali ma’lumotlar bazasi sun’iy kengaytirildi.
5.  **Chuqurlik:** 8 qatlamdan (5 ta konvolyutsiya, 3 ta to‘liq bog‘langan) iborat bo‘lib, o‘z davri uchun juda chuqur edi.

---

### 72. VGG arxitekturasi AlexNet ga nisbatan qaysi jihatlarni yaxshilaganini tushuntirib bering.

**Javob:**
VGG (Visual Geometry Group, 2014) arxitekturasi o‘zining **soddaligi va chuqurligi** bilan ajralib turadi.

**Yaxshilanishlar:**
1.  **Kichik filtrlar:** AlexNetda 11x11 va 5x5 kabi katta filtrlar ishlatilgan edi. VGG bularning barchasini **faqat 3x3 filtrlar**ga almashtirdi.
    *   Mantiq: Ikkita ketma-ket 3x3 qatlam bitta 5x5 qatlam bilan bir xil "ko‘rish maydoniga" (receptive field) ega, lekin kamroq parametr talab qiladi va ko‘proq nochiziqlilik (ReLU) qo‘shadi.
2.  **Chuqurlik:** VGG-16 va VGG-19 modellari nomidan ko‘rinib turibdiki, 16 va 19 qatlamgacha chuqurlashdi (AlexNetda 8 ta edi). Bu modelning murakkab xususiyatlarni o‘rganish qobiliyatini oshirdi.
3.  **Standartlashtirish:** Har bir konvolyutsiya qatlamidan keyin o‘lcham saqlanadi (padding=1), o‘lchamni qisqartirish faqat Max Pooling orqali bajariladi. Bu arxitekturani tushunish va qurishni juda osonlashtirdi.

---

### 73. Inception arxitekturasining asosiy g‘oyasini va uning o‘ziga xos tomonlarini izohlab bering.

**Javob:**
Inception (GoogleNet, 2014) arxitekturasi "kengayish" (width) g‘oyasiga asoslangan.

**Asosiy g‘oya (Inception Module):**
VGG kabi bitta o‘lchamdagi filtrni tanlash o‘rniga (3x3 yoki 5x5?), Inception "Nega hammasini birdaniga ishlatmaymiz?" degan savolni qo‘yadi.
Har bir qatlamda (blokda) parallel ravishda:
*   1x1 konvolyutsiya
*   3x3 konvolyutsiya
*   5x5 konvolyutsiya
*   3x3 Max Pooling
bajariladi va ularning natijalari birlashtiriladi (concatenate).

**O‘ziga xosligi:**
1.  **Ko‘p o‘lchamli nigoh:** Model bir vaqtning o‘zida ham mayda detallarni (1x1, 3x3), ham yirik shakllarni (5x5) ko‘ra oladi.
2.  **1x1 Konvolyutsiya (Bottleneck):** Hisoblash hajmini kamaytirish uchun 3x3 va 5x5 filtrlardan oldin 1x1 filtr ishlatib, kanallar soni (depth) qisqartiriladi. Bu modelni VGGga qaraganda 10 barobar kamroq parametrli, lekin kuchliroq qildi.

---

### 74. ResNet va DenseNet arxitekturalarini taqqoslab, ularning farqli jihatlarini tushuntiring.

**Javob:**
Ikkalasi ham juda chuqur tarmoqlarni o‘qitish muammosini hal qiladi, lekin ma’lumotni uzatish usuli bilan farqlanadi.

1.  **ResNet (Residual Network):**
    *   **Ulanish:** Ma’lumotlar qo‘shiladi (Summation): $y = f(x) + x$.
    *   **Maqsad:** Gradientni oson o‘tkazish. Har bir qatlam faqat "qoldiqni" (residual) o‘rganadi, ya’ni kirishni qanchalik o‘zgartirish kerakligini.
    *   **Natija:** Yuzlab, hatto minglab qatlamli modellarni o‘qitish mumkin bo‘ldi.

2.  **DenseNet (Densely Connected Network):**
    *   **Ulanish:** Ma’lumotlar birlashtiriladi (Concatenation). Har bir qatlam o‘zidan oldingi **barcha** qatlamlarning chiqishini to‘g‘ridan-to‘g‘ri qabul qiladi.
    *   **Farqi:** ResNetda signal qo‘shilib o‘zgarsa, DenseNetda eski signallar asl holida saqlanib qoladi.
    *   **Afzalligi:** "Feature Reuse" (Xususiyatlarni qayta ishlatish). Model boshidagi oddiy xususiyatlar oxirgi qatlamgacha yetib boradi. Parametrlar soni ResNetdan kamroq bo‘ladi (chunki tor qatlamlar yetarli), lekin xotira (RAM) ko‘proq talab qilishi mumkin.

---

### 75. Skip connection (qoldiq/uzatib o‘tuvchi ulanish) nima va u nima uchun kerakligini izohlab bering.

**Javob:**
**Tushuncha:**
Skip Connection (yoki Shortcut) — bu neyron tarmoqda bir yoki bir nechta qatlamlarni hatlab o‘tib, signalni to‘g‘ridan-to‘g‘ri keyingi qatlamlarga uzatish yo‘lagidir.

**Nima uchun kerak?**
1.  **Vanishing Gradient muammosi:** Chuqur tarmoqlarda (masalan, 50 qatlam) orqaga qaytayotgan gradient yo‘lda yo‘qolib ketishi mumkin. Skip connection xuddi "tezkor lift" vazifasini bajarib, gradientni tarmoqning boshiga hech qanday o‘zgarishsiz yetkazib beradi.
2.  **Loss Landscape (Xatolik yuzasi):** Skip connectionlar xatolik funksiyasining g‘adir-budur yuzasini silliqlaydi, bu esa optimizatorga global minimumni osonroq topishga yordam beradi.
3.  **Degradatsiya muammosi:** Tarmoq chuqurlashgani sari uning aniqligi tushib ketish holatini oldini oladi (model kerak bo‘lmasa, oraliq qatlamlarni nolga tenglashtirib, faqat skip connectiondan foydalanishi mumkin).

---

### 76. Rekurrent (takroriy) neyron tarmoqlar (RNN) g‘oyasini tushuntirib bering.

**Javob:**
Oddiy neyron tarmoqlar (FNN, CNN) har bir kirish ma’lumotini mustaqil deb hisoblaydi (masalan, bugungi rasm kechagi rasmga bog‘liq emas).
Lekin ko‘p holatlarda (matn, nutq, ob-havo) **ketma-ketlik** va **tartib** muhim.

**RNN g‘oyasi:**
RNN — bu "xotiraga" ega bo‘lgan tarmoqdir.
*   U ketma-ketlikdagi birinchi elementni ($x_1$) oladi, qayta ishlaydi va natija ($h_1$) chiqaradi.
*   Ikkinchi element ($x_2$) kelganda, model faqat unga qaramaydi, balki **oldingi qadamdan qolgan xotirani ($h_1$)** ham qo‘shib ishlatadi.
*   Shunday qilib, har bir qadamdagi qaror o‘zidan oldingi barcha qadamlarga bog‘liq bo‘ladi.
RNN — bu vaqt bo‘yicha yoyilgan bitta neyron tarmoqdir.

---

### 77. Ma’lumotlardagi vaqt bo‘yicha bog‘liqlik (temporal dependence) tushunchasini izohlab bering.

**Javob:**
Bu tushuncha hozirgi holatning faqat hozirgi omillarga emas, balki o‘tmishdagi voqealarga ham bog‘liqligini anglatadi.

**Misollar:**
1.  **Til (NLP):** "Men bugun bozorga borib, u yerdan shirin..." degan gapni davom ettirish uchun oldingi so‘zlarni ("bozor", "shirin") bilish kerak. "Olma" so‘zi mos keladi, "g‘isht" emas. Bu yerda so‘zlar vaqt bo‘yicha bog‘langan.
2.  **Video:** Videoda koptokning keyingi kadrda qayerda bo‘lishi uning oldingi kadrlardagi tezligi va yo‘nalishiga bog‘liq.
3.  **Moliya:** Aksiyaning bugungi narxi uning kechagi narxi va so‘nggi haftadagi trendiga bog‘liq.

RNN va uning turlari (LSTM, GRU) aynan shunday bog‘liqliklarni modellashtirish uchun yaratilgan.

---

### 78. Nega klassik RNN lar yo‘qolib ketuvchi gradient muammosidan kuchli aziyat chekadi? Tushuntiring.

**Javob:**
Klassik RNN har bir vaqt qadamida (time step) **bir xil vazn matritsasini ($W$)** qayta-qayta ishlatadi (ko‘paytiradi).

**Matematik sabab:**
Agar bizda 100 ta so‘zli gap bo‘lsa, gradient 100 qadam orqaga qaytishi kerak.
Bu jarayon $W 	imes W 	imes … 	imes W$ ($W^{100}$) ko‘paytmasiga olib keladi.
*   Agar $W$ ning qiymatlari (eigenvalues) 1 dan kichik bo‘lsa (masalan 0.9), $0.9^{100} ≈ 0.00002$. Gradient nolga aylanadi va model gapning boshidagi so‘zni "unutib qo‘yadi" (Vanishing Gradient).
*   Agar 1 dan katta bo‘lsa, portlab ketadi (Exploding Gradient).

Shu sababli, oddiy RNNlar faqat qisqa masofadagi (3-10 qadam) bog‘liqliklarni o‘rgana oladi, uzun matnlarda esa ojiz qoladi.

---

### 79. LSTM (Long Short-Term Memory) hujayrasining tuzilishini va ishlash prinsipini izohlab bering.

**Javob:**
LSTM — bu RNNning takomillashgan turi bo‘lib, u "Vanishing Gradient" muammosini hal qilish uchun maxsus **Darvozalar (Gates)** mexanizmini ishlatadi.

**Tuzilishi:**
LSTMda ikkita yo‘l bor:
1.  **Cell State ($C_t$):** Bu "super-shosse". Ma’lumot bu yo‘l bo‘ylab deyarli o‘zgarishsiz (faqat qo‘shish va ayirish amallari bilan) oqib o‘tadi. Bu uzoq muddatli xotirani saqlaydi.
2.  **Hidden State ($h_t$):** Qisqa muddatli xotira va chiqish.

**Darvozalar (Sigmoid qatlami - 0 dan 1 gacha):**
1.  **Forget Gate:** Eski xotiraning qaysi qismini unutish (o‘chirish) kerakligini hal qiladi (masalan, mavzu o‘zgardi, eski mavzuni unut).
2.  **Input Gate:** Yangi ma’lumotning qaysi qismini xotiraga yozish kerakligini belgilaydi.
3.  **Output Gate:** Xotiradagi ma’lumot asosida qanday natija chiqarishni belgilaydi.

Bu mexanizm LSTMga kerakli ma’lumotni 1000 qadam nariga ham yetkazib borish imkonini beradi.

---

### 80. GRU (Gated Recurrent Unit) tuzilishi va ishlash prinsipini tushuntirib bering.

**Javob:**
GRU (2014) — bu LSTMning soddalashtirilgan, lekin deyarli bir xil kuchga ega varianti.

**Farqlari va Ishlash prinsipi:**
1.  **Soddalik:** GRUda alohida "Cell State" yo‘q. U hamma narsani bitta **Hidden State ($h_t$)** da saqlaydi.
2.  **2 ta Darvoza:**
    *   **Update Gate ($z_t$):** Eski xotirani qancha saqlab qolish va yangisini qancha qo‘shishni nazorat qiladi (LSTMdagi Forget va Input darvozalarining birlashmasi).
    *   **Reset Gate ($r_t$):** Yangi ma’lumotni hisoblashda o‘tmishning qancha qismini "unutish" kerakligini belgilaydi.

**Natija:**
GRU kamroq parametrga ega, shuning uchun u tezroq o‘qiydi va kichikroq ma’lumotlar bazasida (dataset) ishlaganda LSTMga qaraganda samaraliroq bo‘lishi mumkin. Katta va murakkab vazifalarda esa LSTM va GRU natijalari deyarli teng bo‘ladi.