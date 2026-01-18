# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (61-70)

### 61. Konvolyutsiya (svyortka) amalini CNN kontekstida tushuntirib bering.

**Javob:**
Konvolyutsiya (Convolution) — bu CNN (Convolutional Neural Networks) ning yuragi hisoblanadi. Matematik jihatdan bu ikkita funksiyaning (bizning holatda kirish rasmi va filtrning) o‘zaro ta’siridir.

**Jarayon:**
Tasavvur qiling, bizda katta rasm (matritsa) va kichkina 3x3 o‘lchamli filtr (yadro) bor.
1.  Filtr rasmning chap yuqori burchagiga qo‘yiladi.
2.  Filtrdagi har bir katak rasmning unga mos kelgan katagi bilan ko‘paytiriladi (element-wise multiplication).
3.  Hosil bo‘lgan 9 ta ko‘paytma jamlanib, bitta son (skalyar) hosil qilinadi.
4.  Filtr rasm bo‘ylab bir qadam o‘ngga siljiydi va jarayon takrorlanadi.
5.  Butun rasm aylanib chiqilgach, yangi kichikroq matritsa — **Feature Map** (Xususiyat xaritasi) hosil bo‘ladi.

**Ma’nosi:**
Bu amal rasmning ma’lum bir qismida biz qidirayotgan naqsh (masalan, vertikal chiziq) qanchalik kuchli namoyon bo‘lganini hisoblaydi.

---

### 62. Qabul maydoni (receptive field) tushunchasini va uning CNN lardagi ahamiyatini izohlab bering.

**Javob:**
**Tushuncha:**
Qabul maydoni (Receptive Field) — bu chiqish qatlamidagi bitta neyron (yoki piksel) kirish tasvirining aynan qaysi qismini (nechta pikselini) "ko‘rib turganini" bildiradi.

**Ierarxiya:**
*   Birinchi konvolyutsiya qatlamida 3x3 filtr ishlatilsa, chiqishdagi 1 ta nuqta asl rasmdagi 3x3 (9 ta) pikselga bog‘liq bo‘ladi.
*   Ikkinchi qatlamda yana 3x3 filtr ishlatilsa, u birinchi qatlamning 3x3 qismini ko‘radi. Lekin birinchi qatlamning o‘zi asl rasmning kattaroq qismini qamrab olgani uchun, ikkinchi qatlamdagi neyron asl rasmdagi 5x5 hududni "his qiladi".

**Ahamiyati:**
Qatlamlar chuqurlashgani sari qabul maydoni kengayib boradi.
*   Dastlabki qatlamlar faqat kichik detallarni (chiziq, nuqta) ko‘radi (Receptive field kichik).
*   Oxirgi qatlamlar butun rasmni to‘liq qamrab oladi (Global Receptive Field). Bu modelga nafaqat detallarni, balki ob’ektning umumiy shakli va kontekstini tushunishga imkon beradi.

---

### 63. Filtr (yadro, kernel) ning ishlash mexanizmini tushuntirib bering.

**Javob:**
Filtr (yoki Yadro/Kernel) — bu CNN o‘qitish jarayonida o‘rganadigan asosiy parametrlardir (vaznlar matritsasi).

**Mexanizm:**
Filtr — bu kichik bir shablon. Masalan, "vertikal chiziqni aniqlovchi filtr" quyidagicha ko‘rinishi mumkin:
```
[-1, 1, -1]
[-1, 1, -1]
[-1, 1, -1]
```
Agar bu filtr rasmning vertikal chiziq bor joyiga tushsa, mos keluvchi piksellar (1 va 1) ko‘paytirilib, katta musbat son hosil bo‘ladi (faollashadi).
Agar u gorizontal chiziqqa tushsa, manfiy va musbatlar bir-birini yeb yuborib, natija 0 ga yaqin bo‘ladi (faollashmaydi).

CNNda biz filtr qiymatlarini qo‘lda yozmaymiz. O‘qitish jarayonida (Backpropagation) model o‘zi uchun eng kerakli filtrlarni (kimdir uchun ko‘z detektori, kimdir uchun yung detektori) avtomatik shakllantiradi.

---

### 64. Xususiyat xaritasi (feature map) nima ekanini va uning CNN dagi rolini izohlang.

**Javob:**
**Tushuncha:**
Xususiyat xaritasi (Feature Map yoki Activation Map) — bu konvolyutsiya amali bajarilgandan so‘ng hosil bo‘lgan natijaviy matritsadir.

**Tuzilishi:**
*   Agar kirish rasmida 3 ta kanal bo‘lsa (RGB), konvolyutsiya qatlamida biz 64 ta filtr ishlatishimiz mumkin.
*   Natijada 64 ta alohida Feature Map hosil bo‘ladi (Hajmi: Balandlik $\times$ Eni $\times$ 64).

**Roli:**
Har bir Feature Map rasmning o‘ziga xos xususiyatini ifodalaydi.
*   1-xarita: Rasmning barcha **vertikal** chiziqlari bor joyni yoritadi.
*   2-xarita: Rasmning **yashil** rangli qismlarini yoritadi.
*   Chuqurroq qatlamlardagi xaritalar: **"Ko'z"** bor joyni yoki **"G'ildirak"** bor joyni ko‘rsatadi.
Keyingi qatlamlar ushbu xaritalarni birlashtirib, murakkabroq xulosa chiqaradi.

---

### 65. Padding nima va u qaysi maqsadlarda qo‘llanilishini tushuntirib bering.

**Javob:**
**Muammo:**
Konvolyutsiya paytida rasm chetlari (border) kamroq ishlatiladi va har bir qatlamda rasm o‘lchami kichrayib boradi (3x3 filtr 5x5 rasmni 3x3 qilib qo‘yadi). Chuqur tarmoqlarda rasm oxiriga borib yo‘q bo‘lib ketishi mumkin.

**Yechim (Padding):**
Padding — bu rasmning atrofiga sun’iy piksellar (odatda nollar — Zero Padding) qo‘shib chiqishdir. 5x5 rasm atrofga 1 qator nol qo‘shilsa, 7x7 bo‘ladi. Filtr yurgizilgach, natija yana 5x5 bo‘lib chiqadi.

**Maqsadlari:**
1.  **O‘lchamni saqlash:** Kirish va chiqish o‘lchamini bir xil ushlab turish (Same Padding). Bu juda chuqur tarmoqlar (ResNet) qurish uchun zarur.
2.  **Chetki ma’lumotni saqlash:** Padding bo‘lmasa, rasmning burchagidagi ma’lumotlar faqat bir marta filtrga tushadi (markazdagilar 9 marta). Padding bilan esa chekkalar ham markaz kabi to‘liq qayta ishlanadi.

---

### 66. Stride = 1 va stride = 2 parametrlarining farqlarini va chiqish o‘lchamiga ta’sirini izohlab bering.

**Javob:**
**Stride (Qadam):** Filtr rasm ustida yuranda har safar necha pikselga siljishini belgilaydi.

1.  **Stride = 1:**
    *   Filtr har bir pikseldan sakramay, zich yurib o‘tadi.
    *   **Ta’siri:** Ma’lumot maksimal darajada saqlanadi. Chiqish o‘lchami kirish bilan deyarli bir xil bo‘ladi.
2.  **Stride = 2:**
    *   Filtr 2 ta katak sakrab yuradi (har ikkinchi pikselni o‘tkazib yuboradi).
    *   **Ta’siri:** Chiqish xaritasi eniga va bo‘yiga **2 barobar kichrayadi** (Downsampling). 100x100 rasm 50x50 bo‘lib qoladi.
    *   **Foydasi:** Bu usul hisoblash hajmini 4 barobar kamaytiradi va Pooling (Max Pooling) o‘rnini bosuvchi alternativa sifatida ishlatiladi.

**Formula:**
$$ Output = \lfloor \frac{Input - Filter + 2 \cdot Padding}{Stride} \rfloor + 1 $$

---

### 67. Max pooling qatlamining vazifasi nimadan iborat? Tushuntirib bering.

**Javob:**
Max Pooling (Maksimal hovuzlash) — bu xususiyat xaritasining o‘lchamini kichraytirish (subsampling) usuli.

**Ishlash:**
Rasm mayda bo‘laklarga (masalan, 2x2 o‘lchamli oynalarga) bo‘linadi va har bir oynadan faqat **eng katta qiymat** tanlab olinadi. Qolgan 3 ta qiymat tashlab yuboriladi.

**Vazifasi:**
1.  **O‘lchamni qisqartirish:** Parametrlar va hisoblash hajmini keskin kamaytiradi (2x2 pooling o‘lchamni 4 barobar qisqartiradi).
2.  **Eng muhimini ajratish:** Eng katta qiymat — bu eng kuchli signal (masalan, eng yorqin chiziq). Bizga chiziqning aniq koordinatasi emas, uning "borligi" muhimroq.
3.  **Invariantlik:** Ob’ekt rasmda biroz siljisa ham, Max Pooling natijasi o‘zgarmaydi. Bu modelning kichik siljishlarga chidamliligini oshiradi.

---

### 68. Average pooling qatlamining vazifasini va qachon ishlatilishini tushuntirib bering.

**Javob:**
Average Pooling (O‘rtacha hovuzlash) — 2x2 oynadagi barcha qiymatlarning o‘rtacha arifmetigini hisoblaydi.

**Qachon ishlatiladi?**
*   Avvallari (LeNet davrida) ko‘p ishlatilgan, lekin hozir asosan Max Pooling afzal ko‘riladi, chunki Average Pooling keskin xususiyatlarni (contrast) "yuvib yuboradi" (xiralashtiradi).
*   **Global Average Pooling:** Ammo, zamonaviy tarmoqlarning (ResNet, Inception) eng oxirgi qismida **Global Average Pooling** ishlatiladi. U har bir xususiyat xaritasining (masalan 7x7 o‘lchamli) to‘liq o‘rtachasini olib, bitta songa aylantiradi. Bu juda ko‘p parametrlarni (Fully Connected qatlamlarni) tejashga yordam beradi va overfittingni kamaytiradi.

---

### 69. Dilated convolution (kengaytirilgan svyortka) tushunchasini va uning afzalliklarini izohlab bering.

**Javob:**
**Tushuncha:**
Dilated Convolution (yoki Atrous Convolution) — bu filtrning kataklari orasiga "bo‘shliqlar" (teshiklar) tashlab ishlash usulidir.
Oddiy 3x3 filtr:
```
1 1 1
1 1 1
1 1 1
```
Dilation rate = 2 bo‘lgan 3x3 filtr (aslida 5x5 maydonni egallaydi):
```
1 0 1 0 1
0 0 0 0 0
1 0 1 0 1
0 0 0 0 0
1 0 1 0 1
```

**Afzalliklari:**
1.  **Katta Qabul Maydoni:** Parametrlar sonini ko‘paytirmasdan va tasvir o‘lchamini kichraytirmasdan (Pooling qilmasdan), **qabul maydonini (receptive field) keskin kengaytirish** imkonini beradi.
2.  **Qo‘llanilishi:** Bu ayniqsa **Semantik Segmentatsiya** (DeepLab modeli) yoki yuqori sifatli audio generatsiyasida (WaveNet) juda foydali. Model katta kontekstni ko‘rishi kerak, lekin tasvir detallarini yo‘qotmasligi kerak.

---

### 70. LeNet arxitekturasining asosiy tuzilishi va xususiyatlarini tushuntirib bering.

**Javob:**
LeNet-5 (Yann LeCun, 1998) — bu CNN tarixidagi ilk muvaffaqiyatli va klassik arxitekturadir. U bank cheklaridagi qo‘lyozma raqamlarni (MNIST) tanish uchun yaratilgan.

**Tuzilishi:**
Juda sodda va ixcham:
1.  **Kirish:** 32x32 o‘lchamli oq-qora rasm.
2.  **C1:** Konvolyutsiya (6 ta filtr, 5x5).
3.  **S2:** Subsampling (Average Pooling).
4.  **C3:** Konvolyutsiya (16 ta filtr, 5x5).
5.  **S4:** Subsampling (Average Pooling).
6.  **C5:** Konvolyutsiya (120 ta filtr) - bu yerda u to‘liq bog‘langan qatlamga o‘xshab ishlaydi.
7.  **F6:** Fully Connected (84 ta neyron).
8.  **Chiqish:** RBF (hozirgi Softmax o‘tmishdoshi) - 10 ta sinf (0-9).

**Xususiyatlari:**
U birinchi bo‘lib "Konvolyutsiya -> Pooling -> Non-linearity" ketma-ketligini standartlashtirdi. Garchi u hozirgi standartlarga ko‘ra juda kichik bo‘lsa-da (60 ming parametr), u zamonaviy kompyuterni ko‘rish sohasiga asos soldi.
