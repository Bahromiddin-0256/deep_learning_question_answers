# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (31-40)

### 31. Ma’lumotlarni kengaytirish (data augmentation) nima va u nima uchun zarur ekani haqida yozing.

**Javob:**
**Tushuncha:**
Ma’lumotlarni kengaytirish (Data Augmentation) — bu mavjud o‘qitish ma’lumotlariga (dataset) sun’iy o‘zgartirishlar kiritish orqali ma’lumotlar hajmini va xilma-xilligini oshirish usulidir.
Masalan, bizda bitta mushuk rasmi bor. Biz bu rasmni:
1.  Biroz buramiz (rotation).
2.  Ko‘zgudagi aksini olamiz (flip).
3.  Yaqinlashtiramiz (zoom).
4.  Rangini o‘zgartiramiz (color jitter).
Natijada bitta rasmdan 10 ta "yangi" rasm hosil bo‘ladi.

**Zarurati:**
1.  **Overfittingni kamaytirish:** Model bir xil rasmni qayta-qayta ko‘rsa, uni yodlab oladi. Agar rasm har safar biroz o‘zgargan holda kelsa, model uni yodlay olmaydi, aksincha, mushukning asosiy belgilarini (quloq, ko‘z) ajratib olishga majbur bo‘ladi.
2.  **Invariantlikni oshirish:** Model rasm burilgan bo‘lsa ham, qorong‘iroq bo‘lsa ham ob’ektni taniy olishi kerak. Augmentatsiya modelni turli sharoitlarga "immunitetini" oshiradi.
3.  **Ma’lumot yetishmovchiligi:** Agar qo‘limizda real ma’lumotlar kam bo‘lsa, bu usul datasetni sun’iy ravishda ko‘paytirishning eng samarali yo‘lidir.

---

### 32. Shovqin kiritish (noise injection) orqali muntazamlashtirishning ishlash prinsipi va maqsadini tushuntirib bering.

**Javob:**
**Ishlash prinsipi:**
Bu usulda o‘qitish jarayonida ma’lumotlarga yoki modelning ichki qismlariga ataylab tasodifiy shovqin (random noise) qo‘shiladi.
Shovqin quyidagi joylarga qo‘shilishi mumkin:
*   **Kirishga (Input):** Rasmga "qor" (gaussian noise) yog‘dirish.
*   **Vaznlarga (Weights):** Neyron tarmoq vaznlariga kichik o‘zgarishlar kiritish.
*   **Chiqishga (Labels):** Ba’zan 100% ishonch bilan berilgan javoblarni biroz yumshatish (Label Smoothing).

**Maqsadi:**
Bu usul modelning **mustahkamligini (robustness)** oshiradi.
Agar model kirishdagi kichik shovqin sababli "bu mushuk" degan fikrini "bu it" ga o‘zgartirib yuborsa, demak u ishonchsiz modeldir. Shovqin bilan o‘qitilgan model esa: "Rasm xira bo‘lsa ham, nuqtalar bo‘lsa ham, bu baribir mushuk" deb xulosa chiqarishni o‘rganadi. Bu modelning haqiqiy hayotdagi (ideal bo‘lmagan) ma’lumotlarda ishlash qobiliyatini keskin yaxshilaydi.

---

### 33. Oversampling (ortiqcha tanlash) usuli nomutanosib (imbalanced) sinflarga ega ma’lumotlar bilan ishlashda qanday yordam berishini izohlang.

**Javob:**
**Muammo:**
Tasavvur qiling, bizda kasallikni aniqlash uchun ma’lumotlar bor: 990 ta sog‘lom odam va faqat 10 ta kasal odam. Agar model hamma joyda "Sog‘lom" deb javob bersa ham, u 99% aniqlikka erishadi, lekin kasallarni topolmaydi. Bu **nomutanosiblik** muammosidir.

**Oversampling yordami:**
Oversampling (masalan, SMOTE algoritmi yoki oddiy nusxalash) kam sonli sinf (kasallar) vakillarini sun’iy ravishda ko‘paytiradi.
*   Biz 10 ta kasal odam ma’lumotini 100 marta nusxalab, 1000 taga yetkazamiz.
*   Endi modelda 990 ta sog‘lom va 1000 ta kasal namuna bor (balans 50/50).

**Natija:**
Model o‘qish jarayonida kasallarni e’tiborsiz qoldira olmaydi. U majburan kasallik belgilarini o‘rganishga kirishadi, chunki endi xato qilsa, jarima (loss) katta bo‘ladi. Bu kam uchraydigan, lekin muhim sinflarni aniqlash aniqligini (Recall) oshiradi.

---

### 34. Nega juda katta (ko‘p parametrli) modellar overfitting ga kuchliroq moyil bo‘ladi? Izohlab bering.

**Javob:**
**Xotira sig‘imi:**
Modelning parametrlari soni — bu uning "xotira hajmi" kabidir.
*   Kichik model (kam parametr) xotirasi past bo‘lgani uchun har bir detalni eslab qololmaydi. U faqat eng umumiy qoidalarni o‘rganishga majbur.
*   Katta model (ko‘p parametr) esa "fotografik xotira"ga ega bo‘lishi mumkin. U o‘qitish to‘plamidagi har bir rasmni, har bir nuqtasigacha individual tarzda yodlab olishi mumkin.

**Occam's Razor (Okkam ustarasi):**
Mantiq qonuniga ko‘ra, eng oddiy yechim ko‘pincha eng to‘g‘ri yechim bo‘ladi. Katta modellar esa o‘ta murakkab va egri-bugri funksiyalarni qurishga qodir. Ular ma’lumotdagi haqiqiy signalni emas, balki tasodifiy shovqinni ham qonuniyat deb o‘ylab, unga moslashib oladi. Natijada, train setda xatolik 0% bo‘ladi, lekin yangi ma’lumotda model butunlay adashadi.

Shuning uchun katta modellarda regulyarizatsiya (Dropout, L2) juda qattiq qo‘llanilishi shart.

---

### 35. Train, validation va test xatoliklari o‘rtasidagi farqlarni tushuntirib bering.

**Javob:**
Ma’lumotlar uch qismga bo‘linadi va har birining o‘z xatolik ko‘rsatkichi bor:

1.  **Train Error (O‘qitish xatoligi):**
    *   Model ayni damda o‘qiyotgan ma’lumotlardagi xatolik.
    *   Bu xatolik vaqt o‘tishi bilan doim kamayib boradi. U modelning "o‘rganish qobiliyatini" ko‘rsatadi.
2.  **Validation Error (Tekshirish xatoligi):**
    *   O‘qitish jarayonida model ko‘rmaydigan alohida to‘plam.
    *   Bu xatolik **giperparametrlarni sozlash** (Learning rate, qatlamlar sonini tanlash) va **Overfittingni aniqlash** uchun ishlatiladi. Agar Train error tushsa-yu, Validation error oshsa — demak, muammo bor.
3.  **Test Error (Sinov xatoligi):**
    *   Model to‘liq tayyor bo‘lgandan keyingina ishlatiladigan, "qulflangan" ma’lumotlar to‘plami.
    *   Bu modelning real hayotda qanday ishlashini ko‘rsatuvchi yagona xolis bahodir. Validation to‘plamga qarab modelni sozlaganimiz uchun, Validation natijasi biroz "optimistik" bo‘lishi mumkin. Test natijasi esa haqiqatdir.

---

### 36. Neyron tarmog‘ida faollashtirish funksiyasining umumiy roli nimadan iboratligini izohlang.

**Javob:**
**Asosiy rol: Chiziqli bo‘lmaganlikni (Non-linearity) kiritish.**

Agar neyron tarmoqda faollashtirish funksiyasi bo‘lmasa (yoki faqat chiziqli funksiya $f(x)=x$ bo‘lsa), neyron tarmoq qanchalik chuqur bo‘lmasin, u faqat **bitta chiziqli regressiya** (Oddiy $y = Wx + b$) ga teng bo‘lib qoladi.
Sababi: Chiziqli funksiyalarning yig‘indisi va ko‘paytmasi baribir chiziqli funksiyadir.

Dunyodagi deyarli barcha qiziqarli muammolar (tasvirni tanish, tarjima qilish) chiziqli emas.
Faollashtirish funksiyasi (Sigmoid, ReLU) kirish signalini "egadi", "bükadi" yoki "o‘chiradi". Bu tarmoqqa to‘g‘ri chiziqdan tashqari, egri chiziqlarni, murakkab shakllarni va chegaralarni chizish imkonini beradi. Aynan shu narsa neyron tarmoqni "aqlli" qiladi.

---

### 37. Nega sigmoid faollashtirish funksiyasi chuqur tarmoqlarda ko‘p hollarda samarasiz hisoblanadi? Tushuntiring.

**Javob:**
Sigmoid: $f(x) = \frac{1}{1 + e^{-x}}$. U qiymatni (0, 1) oraliqqa siqadi.

**Muammolari:**
1.  **Vanishing Gradient:** Sigmoidning hosilasi eng ko‘pi bilan 0.25 ga teng. Kirish qiymati juda katta yoki juda kichik bo‘lsa, hosila 0 ga teng bo‘lib qoladi (Saturation - to‘yinganlik). Chuqur tarmoqda bu kichik sonlar ko‘paytirilganda gradient yo‘qolib ketadi.
2.  **Zero-centered emas (Nolga simmetrik emas):** Sigmoid faqat musbat qiymat (0 dan 1 gacha) qaytaradi. Bu shuni anglatadiki, keyingi qatlamga kiruvchi ma’lumotlarning hammasi musbat bo‘ladi. Bu gradientlarning hammasi bir vaqtning o‘zida yoki musbat, yoki manfiy bo‘lishiga olib keladi va optimizatsiya jarayonini "zig-zag" shaklida sekinlashtiradi.
3.  **Eksponentani hisoblash og‘ir:** Kompyuter uchun $e^x$ ni hisoblash oddiy ko‘paytirish yoki taqqoslashdan (ReLU) ko‘ra ko‘proq resurs talab qiladi.

---

### 38. ReLU va Leaky ReLU faollashtirish funksiyalarini taqqoslab bering.

**Javob:**
1.  **ReLU (Rectified Linear Unit):**
    *   Formula: $f(x) = \max(0, x)$.
    *   **Afzalligi:** Hisoblash juda tez. Gradient yo‘qolmaydi (musbat tomonda hosila 1).
    *   **Kamchiligi ("Dying ReLU"):** Agar kirish manfiy bo‘lsa, chiqish 0 bo‘ladi va gradient ham 0 bo‘ladi. Agar neyron bir marta "o‘lib qolsa" (doim manfiy qiymat olsa), u qaytib hech qachon o‘zgarmaydi va o‘qishdan to‘xtaydi.

2.  **Leaky ReLU:**
    *   Formula: $f(x) = \max(\alpha x, x)$. (Odatda $\alpha = 0.01$).
    *   **Farqi:** Manfiy qiymatlar uchun nol emas, balki juda kichik son qaytaradi.
    *   **Afzalligi:** Bu "o‘lgan neyron" muammosini hal qiladi. Hatto manfiy sohada ham kichik gradient oqimi mavjud bo‘ladi, shuning uchun neyron qayta "tirilib" ketishi mumkin.
    *   **Xulosa:** Agar ReLU ishlamasa yoki neyronlar o‘lib qolayotgan bo‘lsa, Leaky ReLU yaxshi alternativ hisoblanadi.

---

### 39. ELU (Exponential Linear Unit) funksiyasi nima va uning asosiy afzalliklari nimada, izohlab bering.

**Javob:**
**Tushuncha:**
ELU — bu ReLU ning yana bir takomillashgan varianti.
*   Musbat qiymatlarda: $x$ (xuddi ReLU kabi).
*   Manfiy qiymatlarda: $\alpha(e^x - 1)$. Bu silliq egri chiziq bo‘lib, sekin-asta $-\alpha$ ga yaqinlashadi.

**Afzalliklari:**
1.  **Nolga yaqin o‘rtacha qiymat:** Leaky ReLU dan farqli o‘laroq, ELU manfiy qiymatlarni silliq qabul qiladi, bu esa chiqishlarning o‘rtacha qiymatini (mean) nolga yaqinlashtiradi. Bu o‘qitishni tezlashtiradi (xuddi Batch Norm effekti kabi).
2.  **Shovqinga chidamlilik:** Manfiy sohadagi to‘yinganlik (saturation) katta manfiy shovqinlarga nisbatan modelni barqarorroq qiladi.
3.  **Kamchiligi:** Eksponenta hisoblangani uchun ishlash tezligi (inference) ReLU dan biroz sekinroq bo‘lishi mumkin, lekin o‘qitish tezligi (konvergensiya) yuqori bo‘lgani uchun bu qoplanib ketadi.

---

### 40. Softmax funksiyasi nima uchun va qaysi qatlamda odatda qo‘llanilishini tushuntirib bering.

**Javob:**
**Vazifasi:**
Softmax funksiyasi ixtiyoriy haqiqiy sonlar vektorini (logits) **ehtimolliklar taqsimotiga** aylantirib beradi.
*   U barcha qiymatlarni (0, 1) oraliqqa tushiradi.
*   Barcha qiymatlar yig‘indisi aynan 1 ga teng bo‘lishini ta’minlaydi.
Formula: $p_i = \frac{e^{z_i}}{\sum e^{z_j}}$.

**Qo‘llanilishi:**
Odatda neyron tarmoqning **eng oxirgi chiqish qatlamida**, agar masala **Ko‘p sinfli tasniflash (Multi-class classification)** bo‘lsa ishlatiladi.
Misol: Rasmda it, mushuk yoki qush borligini aniqlash. Tarmoq oxirida [2.0, 1.0, 0.1] kabi sonlar chiqaradi. Softmax buni [0.7, 0.2, 0.1] ga aylantiradi. Biz buni "70% it, 20% mushuk, 10% qush" deb tushunamiz va eng katta ehtimollikni javob sifatida olamiz.
Agarda masala faqat "ha/yo‘q" (binar) bo‘lsa, Softmax o‘rniga Sigmoid ishlatiladi.
