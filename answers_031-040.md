# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (31-40)

### 31. Ma’lumotlarni kengaytirish (data augmentation) nima va u nima uchun zarur ekani haqida yozing.

**Javob:**
**Tushuncha:**
Ma’lumotlarni kengaytirish (Data Augmentation) — bu sun’iy intellekt modellarini o‘qitishda ishlatiladigan eng samarali usullardan biri bo‘lib, u mavjud ma’lumotlar to‘plamini (dataset) sun’iy ravishda ko‘paytirishga asoslanadi. Bu jarayonda asl ma’lumotlarga (rasm, matn yoki ovoz) turli xil o‘zgartirishlar kiritilib, yangi, lekin mazmunan o‘xshash namunalar yaratiladi.

**Tasvirlar uchun misollar:**
Bitta mushuk rasmidan quyidagi amallar orqali 10-20 ta yangi rasm olish mumkin:
1.  **Geometrik o‘zgarishlar:** Rasmni ma’lum burchakka burish (rotation), ko‘zgudagi kabi akslantirish (flip), qirqish (crop), masshtabni o‘zgartirish (zoom).
2.  **Rang o‘zgarishlari:** Yorug‘likni (brightness), kontrastni yoki rang to‘yinganligini (saturation) o‘zgartirish.
3.  **Shovqin qo‘shish:** Rasmga Gauss shovqini yoki dog‘lar qo‘shish.

**Matnlar uchun:**
So‘zlarni sinonimlarga almashtirish, gap tuzilishini o‘zgartirish yoki matnni boshqa tilga tarjima qilib, yana qayta o‘z tiliga tarjima qilish (back-translation).

**Zarurati va Foydalari:**
1.  **Overfitting (Yodlab olish) ning oldini olish:** Agar model bir xil rasmni qayta-qayta ko‘rsa, u rasmdagi ob’ektni emas, balki aynan o‘sha piksellar ketma-ketligini yodlab oladi. Kengaytirish orqali model har safar "yangi" rasmni ko‘radi. Bu modelni piksellarga emas, balki ob’ektning **shakli va tuzilishiga** e’tibor qaratishga majbur qiladi.
2.  **Invariantlikni (O‘zgarmaslikni) ta’minlash:** Model uchun mushuk rasmi o‘ngga qaragan bo‘lsa ham, chapga qaragan bo‘lsa ham, yorug‘ yoki qorong‘i xonada bo‘lsa ham — baribir "mushuk" bo‘lib qolishi kerak. Data augmentation modelda ushbu invariantlik xususiyatini shakllantiradi.
3.  **Qimmat ma’lumotlarni tejash:** Haqiqiy ma’lumotlarni yig‘ish va belgilash (labeling) ko‘p vaqt va mablag‘ talab qiladi. Augmentatsiya mavjud ozgina ma’lumotdan maksimal darajada foydalanish imkonini beradi.

---

### 32. Shovqin kiritish (noise injection) orqali muntazamlashtirishning ishlash prinsipi va maqsadini tushuntirib bering.

**Javob:**
**Ishlash Prinsipi:**
Shovqin kiritish (Noise Injection) — bu neyron tarmoqning o‘qitish jarayonini ataylab qiyinlashtirish usulidir. Maqsad modelni "qulay sharoitda" ishlashga o‘rgatish emas, balki har qanday kutilmagan vaziyatlarga tayyorlashdir.

Shovqin uch xil joyga kiritilishi mumkin:
1.  **Kirish ma’lumotlariga (Input Noise):** Masalan, rasm piksellariga tasodifiy "qor" (Gaussian noise) qo‘shiladi. Model bundan shovqin ostidagi haqiqiy tasvirni ajratib olishni o‘rganadi. Bu "Denoising Autoencoders" da asosiy usuldir.
2.  **Vaznlarga (Weight Noise):** Tarmoqning ichki parametrlari ($W$) ga kichik tasodifiy qiymatlar qo‘shiladi. Bu modelning bitta aniq vazn qiymatiga bog‘lanib qolishini oldini oladi va yechimlar fazosida (loss landscape) kengroq va barqarorroq minimumni topishga yordam beradi. Bu Bayescha (Bayesian) yondashuvga yaqin.
3.  **Chiqishga (Label Noise):** "Label Smoothing" deb ham ataladi. Masalan, aniq javob "1" (100% ishonch) o‘rniga "0.9" deb beriladi. Bu modelni o‘z javobiga haddan tashqari qattiq ishonib ketishdan (overconfidence) saqlaydi.

**Maqsadi:**
*   **Robustness (Mustahkamlik):** Model real hayotdagi sifatsiz, xira yoki shovqinli ma’lumotlarda ham to‘g‘ri ishlaydi.
*   **Decision Boundary Smoothing:** Shovqin modelning qaror qabul qilish chegaralarini "silliqlaydi". Bu esa modelning yangi, notanish ma’lumotlarda keskin xatolar qilmasligini ta’minlaydi.
*   **Overfittingga qarshi:** Shovqin xuddi Dropout kabi modelga ma’lumotlarni yodlab olishga xalaqit beradi.

---

### 33. Oversampling (ortiqcha tanlash) usuli nomutanosib (imbalanced) sinflarga ega ma’lumotlar bilan ishlashda qanday yordam berishini izohlang.

**Javob:**
**Muammo (Imbalanced Dataset):**
Real hayotiy masalalarda sinflar ko‘pincha teng taqsimlanmagan bo‘ladi. Masalan, bank firibgarligini aniqlashda: 1 millionta oddiy tranzaksiya va atigi 100 ta firibgarlik (fraud) holati bo‘lishi mumkin. Agar biz modelni shundayligicha o‘qitsak, model shunchaki "Hamma tranzaksiya toza" deb javob berishni o‘rganadi va 99.99% aniqlikka erishadi, lekin bitta ham firibgarni topolmaydi. Bu modelning foydasiz ekanligini anglatadi.

**Oversampling (Up-sampling) Yechimi:**
Bu usul kam sonli sinf (minority class) vakillarini sun’iy ravishda ko‘paytirish orqali muvozanatni tiklaydi.
1.  **Random Oversampling:** Kam uchraydigan sinf namunalarini shunchaki nusxalab, sonini ko‘paytirish. Masalan, 100 ta firibgarlik holatini 10,000 marta takrorlab, milliontaga yetkazish.
2.  **SMOTE (Synthetic Minority Over-sampling Technique):** Bu aqlliroq usul. U mavjud namunalarni shunchaki nusxalamaydi, balki ularga o‘xshash **yangi, sun’iy namunalar** yaratadi. U ikki firibgarlik holati orasidagi masofani o‘lchab, ular orasida joylashgan yangi matematik nuqtalarni hosil qiladi.

**Qanday yordam beradi?**
*   **E’tiborni jalb qilish:** Model o‘qish jarayonida "kamchil" sinfni tez-tez ko‘ra boshlaydi. Bu uning xatolik funksiyasiga (loss) katta ta’sir qiladi. Model endi bu sinfni e’tiborsiz qoldira olmaydi, aks holda katta jarima oladi.
*   **Recall (Sezgirlik)ni oshirish:** Model kam uchraydigan muhim holatlarni (kasallik, firibgarlik) o‘tkazib yubormaslikni o‘rganadi.

---

### 34. Nega juda katta (ko‘p parametrli) modellar overfitting ga kuchliroq moyil bo‘ladi? Izohlab bering.

**Javob:**
**Model Sig‘imi (Capacity):**
Modelning parametrlari soni (vaznlar va siljishlar) uning axborot saqlash sig‘imini belgilaydi.
*   Kichik modellar (kam parametrli) cheklangan "xotira"ga ega bo‘lib, ular ma’lumotlardagi faqat eng kuchli, umumiy qonuniyatlarni (trendlarni) eslab qola oladi. Ular mayda detallarni e’tiborsiz qoldirishga majbur.
*   Katta modellar (millionlab yoki milliardlab parametrli) esa ulkan xotiraga ega. Ular o‘qitish to‘plamidagi (training set) **har bir namunani**, uning barcha shovqinlari va o‘ziga xos xatolari bilan birga **yodlab olish** (memorization) imkoniyatiga ega.

**Overfitting Mexanizmi:**
Tasavvur qiling, siz talabaga dars o‘tyapsiz.
*   Yaxshi talaba (optimal model) mavzuning mohiyatini tushunishga harakat qiladi.
*   "Overfit" bo‘lgan talaba (katta model) esa darslikdagi hamma gaplarni so‘zma-so‘z yodlab oladi. Imtihonda darslikdagi savol tushsa, u 100% javob beradi (Train accuracy yuqori). Lekin savol ozgina o‘zgartirilsa (yangi ma’lumot), u javob berolmaydi, chunki u mantiqni tushunmagan.

Katta modellar ma’lumotlar orasidagi haqiqiy funksional bog‘liqlikni topish o‘rniga, har bir nuqta uchun alohida qoida yaratishga moyil bo‘ladi.
Shu sababli, katta modellarda **regulyarizatsiya** (Dropout, L1/L2, Early Stopping) juda muhim. Qiziqarli jihati, zamonaviy "Double Descent" nazariyasiga ko‘ra, ma’lumotlar hajmi yetarlicha katta bo‘lsa, o‘ta katta modellar yana yaxshi ishlay boshlashi ham mumkin.

---

### 35. Train, validation va test xatoliklari o‘rtasidagi farqlarni tushuntirib bering.

**Javob:**
Mashinaviy o‘qitishda ma’lumotlar to‘plami uch qismga ajratiladi va ularning har biri modelni baholashda alohida rol o‘ynaydi.

1.  **Train Error (O‘qitish xatoligi):**
    *   **Manba:** Model bevosita ko‘rib, o‘rganayotgan ma’lumotlar (Train set).
    *   **Vazifasi:** Model ushbu xatolikni minimallashtirish uchun o‘z vaznlarini (weights) o‘zgartiradi (Gradient Descent shu yerda ishlaydi).
    *   **Xususiyati:** O‘qitish davomida bu xatolik deyarli doim kamayib boradi. Agar u nolga yaqinlashsa, bu model "o‘rgandi" yoki "yodlab oldi" deganidir.

2.  **Validation Error (Tekshirish/Validatsiya xatoligi):**
    *   **Manba:** O‘qitish jarayonida qatnashmaydigan, lekin modelni vaqti-vaqti bilan tekshirib turish uchun ajratilgan qism (Validation set).
    *   **Vazifasi:** Modelning **yangi ma’lumotga moslashuvchanligini** kuzatish va **giperparametrlarni** (learning rate, qatlamlar soni) tanlash.
    *   **Xususiyati:** Agar Train Error kamaysa-yu, Validation Error osha boshlasa, bu **Overfitting** (qayta moslashish) boshlanganining aniq belgisidir. Shu nuqtada o‘qitishni to‘xtatish kerak (Early Stopping).

3.  **Test Error (Sinov xatoligi):**
    *   **Manba:** Model to‘liq tayyor bo‘lgunga qadar umuman ishlatilmaydigan, "seifda saqlangan" ma’lumotlar (Test set).
    *   **Vazifasi:** Modelning real hayotdagi samaradorligini xolis baholash.
    *   **Farqi:** Biz Validation setga qarab modelni sozlaganimiz uchun, model bilvosita bo‘lsa ham Validation ma’lumotlariga moslashib qoladi. Test Error esa mutlaqo toza va yakuniy bahodir.

---

### 36. Neyron tarmog‘ida faollashtirish funksiyasining umumiy roli nimadan iboratligini izohlang.

**Javob:**
**Asosiy Rol: Chiziqli bo‘lmaganlikni (Non-linearity) joriy qilish.**

Agar neyron tarmoqda faollashtirish funksiyasi (activation function) bo‘lmasa yoki faqat chiziqli funksiya ($f(x) = x$) ishlatilsa, butun tarmoq, necha ming qatlamdan iborat bo‘lishidan qat’i nazar, matematik jihatdan **bitta oddiy chiziqli regressiyaga** ($y = Wx + b$) teng bo‘lib qoladi.
*   Sababi: Chiziqli funksiyalarning kompozitsiyasi (funksiya ichida funksiya) baribir chiziqli funksiya bo‘ladi. $f(g(x)) = W_2(W_1x) = W_{new}x$.

**Nima uchun bu muhim?**
Haqiqiy dunyo muammolarining aksariyati (tasvirlarni tanish, ovozni tushunish, til modellari) **chiziqli ajratib bo‘lmaydigan** (non-linearly separable) muammolardir.
Masalan, oddiy **XOR muammosi**:
*   (0, 0) -> 0
*   (1, 1) -> 0
*   (0, 1) -> 1
*   (1, 0) -> 1
Bu nuqtalarni tekislikda bitta to‘g‘ri chiziq bilan ajratishning iloji yo‘q.

Faollashtirish funksiyalari (ReLU, Sigmoid, Tanh) kirish signallarini "egadi", "bukadi" va murakkab shakllarga soladi. Bu neyron tarmoqqa to‘g‘ri chiziqlardan tashqari, egri chiziqlar, aylanalar va har qanday murakkab chegaralarni chizish imkonini beradi. Aynan shu narsa chuqur o‘qitishni "Universal Approksimator"ga aylantiradi.

---

### 37. Nega sigmoid faollashtirish funksiyasi chuqur tarmoqlarda ko‘p hollarda samarasiz hisoblanadi? Tushuntiring.

**Javob:**
Sigmoid funksiyasi: $\sigma(x) = \frac{1}{1 + e^{-x}}$. U har qanday sonni (0, 1) oraliqqa siqib beradi. Tarixiy jihatdan u biologik neyronlarga o‘xshashligi uchun ko‘p ishlatilgan, ammo chuqur tarmoqlarda (Deep Learning) u jiddiy muammolarga sabab bo‘ladi:

1.  **Vanishing Gradient (Gradient yo‘qolishi):**
    *   Sigmoid funksiyasining hosilasi: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.
    *   Bu hosilaning maksimal qiymati **0.25** ga teng (u ham bo‘lsa $x=0$ da).
    *   Agar $x$ juda katta yoki juda kichik bo‘lsa (masalan, $x=10$ yoki $x=-10$), funksiya grafigi tekislashib qoladi (Saturation) va hosila **0 ga teng** bo‘ladi.
    *   Orqaga tarqalish (Backpropagation) paytida gradientlar zanjir qoidasi bo‘yicha ko‘paytiriladi. Ko‘p qatlamli tarmoqda 0.25 dan kichik sonlarni ketma-ket ko‘paytirish gradientni yo‘q qilib yuboradi. Natijada dastlabki qatlamlar o‘qimaydi.

2.  **Not Zero-Centered (Nolga simmetrik emas):**
    *   Sigmoid har doim musbat qiymat qaytaradi. Bu shuni anglatadiki, keyingi qatlamga kiradigan barcha ma’lumotlar musbat bo‘ladi. Bu esa vaznlar gradientining barchasi bir vaqtning o‘zida yo musbat, yo manfiy bo‘lishiga olib keladi. Bu optimizatsiya yo‘lini samarasiz "zig-zag" shaklga keltiradi va o‘qitishni sekinlashtiradi.

3.  **Hisoblash og‘irligi:** Eksponenta ($e^x$) ni hisoblash kompyuter protsessori uchun oddiy qo‘shish yoki maksimum (ReLU) amalidan ko‘ra qimmatroqdir.

---

### 38. ReLU va Leaky ReLU faollashtirish funksiyalarini taqqoslab bering.

**Javob:**
**1. ReLU (Rectified Linear Unit):**
*   **Formula:** $f(x) = \max(0, x)$.
*   **Mantiq:** Agar signal musbat bo‘lsa, uni o‘zgarishsiz o‘tkazadi. Agar manfiy bo‘lsa, to‘sib qo‘yadi (0 ga aylantiradi).
*   **Afzalligi:**
    *   **Sparsity (Siyraklik):** Neyronlarning bir qismi 0 bo‘lib turadi, bu modelni yengil va samarali qiladi.
    *   **Gradient saqlanishi:** Musbat sohada hosila aynan **1 ga teng**. Bu chuqur tarmoqlarda gradient yo‘qolishi (Vanishing Gradient) muammosini hal qiladi va o‘qitishni keskin tezlashtiradi.
*   **Kamchiligi (Dying ReLU):** Agar neyronning kirish qiymati manfiy bo‘lib qolsa, u 0 qaytaradi va uning gradienti ham 0 bo‘ladi. Agar o‘qitish davomida neyron vaznlari shunday o‘zgarib qolsaki, u doim manfiy qiymat oladigan bo‘lsa, bu neyron butunlay "o‘ladi" va qaytib hech qachon ishlamaydi.

**2. Leaky ReLU:**
*   **Formula:** $f(x) = \max(\alpha x, x)$, odatda $\alpha = 0.01$.
*   **Mantiq:** Manfiy qiymatlarni butunlay o‘chirmaydi, balki ularni biroz kamaytirib o‘tkazadi (kichik nishablik).
*   **Farqi:** "Dying ReLU" muammosini hal qiladi. Manfiy sohada ham kichik gradient oqimi mavjud bo‘ladi, shuning uchun "o‘lgan" neyronlar qayta tiklanish imkoniyatiga ega bo‘ladi.
*   **Qachon ishlatiladi:** Agar ReLU bilan o‘qitishda ko‘p neyronlar ishlamay qolayotganini sezsangiz, Leaky ReLU yaxshi alternativ hisoblanadi.

---

### 39. ELU (Exponential Linear Unit) funksiyasi nima va uning asosiy afzalliklari nimada, izohlab bering.

**Javob:**
**Tushuncha:**
ELU — bu ReLU ning kamchiliklarini tuzatish va model barqarorligini oshirish uchun ishlab chiqilgan funksiya.
*   **Musbat qismda ($x > 0$):** $x$ (xuddi ReLU kabi).
*   **Manfiy qismda ($x \le 0$):** $\alpha (e^x - 1)$. Bu qism silliq egri chiziq bo‘lib, sekin-asta $-\alpha$ qiymatiga yaqinlashadi (saturatsiya bo‘ladi).

**Afzalliklari:**
1.  **Zero-Mean Activations (O‘rtacha qiymat nolga yaqin):** ReLU va Sigmoid dan farqli o‘laroq, ELU manfiy qiymatlar ham qabul qiladi. Bu qatlam chiqishlarining o‘rtacha qiymatini 0 ga yaqinlashtiradi. Bu xuddi Batch Normalization effektiga o‘xshab, keyingi qatlamlar uchun o‘qishni osonlashtiradi va konvergensiyani tezlashtiradi.
2.  **Shovqinga chidamlilik:** Leaky ReLU manfiy sohada cheksiz pastga ketishi mumkin ($-\infty$). ELU esa manfiy sohada ma’lum bir qiymatda ($-\alpha$) to‘xtaydi (saturation). Bu modelni kirish ma’lumotlaridagi katta manfiy shovqinlarga nisbatan chidamliroq (robust) qiladi.
3.  **Silliqlik:** $x=0$ nuqtasida ReLU ning hosilasi keskin o‘zgaradi. ELU esa barcha nuqtalarda, shu jumladan nolda ham silliq (diferensiallanuvchi).

**Kamchiligi:**
Eksponenta funksiyasini hisoblash kompyuter uchun biroz og‘irroq bo‘lgani sababli, modelning ishlash vaqti (inference time) ReLU ga nisbatan sekinroq bo‘lishi mumkin.

---

### 40. Softmax funksiyasi nima uchun va qaysi qatlamda odatda qo‘llanilishini tushuntirib bering.

**Javob:**
**Mohiyati:**
Softmax funksiyasi ixtiyoriy haqiqiy sonlar to‘plamini (vektorini) **ehtimolliklar taqsimotiga** aylantirib beruvchi matematik funksiyadir.
Formula:
$$ \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} $$
Bu yerda $z$ — modeldan chiqqan xom ballar (logits).

**Softmax ikkita muhim ishni bajaradi:**
1.  **Normallashtirish (0-1 oraliq):** Barcha chiqish qiymatlarini 0 va 1 oralig‘iga tushiradi.
2.  **Yig‘indini 1 ga tenglash:** Barcha sinflar bo‘yicha ehtimolliklar yig‘indisi aniq 1 ga teng bo‘lishini ta’minlaydi ($\sum p_i = 1$). Bu bizga natijani foizlarda (masalan, 70%, 20%, 10%) talqin qilish imkonini beradi.

**Qo‘llanilishi:**
Softmax deyarli har doim **Ko‘p sinfli tasniflash (Multi-class Classification)** masalalarida neyron tarmoqning **eng oxirgi chiqish qatlamida** ishlatiladi.
**Misol:**
Model rasmda nima borligini aniqlashi kerak (It, Mushuk, Qush).
Oxirgi qatlamdan chiqqan xom ballar (logits): `[2.0, 1.0, 0.1]`.
Softmax qo‘llagandan keyin: `[0.7, 0.2, 0.1]`.
Xulosa: "Bu 70% ehtimol bilan It".

**Nega oddiy bo‘lish emas, eksponenta ($e^x$) ishlatiladi?**
Eksponenta katta qiymatlarni yanada kattalashtiradi ("Winner takes all" effekti). Bu modelning eng ishonchli javobini yaqqol ajratib ko‘rsatishga yordam beradi va Cross-Entropy Loss funksiyasi bilan matematik jihatdan juda yaxshi moslashadi.