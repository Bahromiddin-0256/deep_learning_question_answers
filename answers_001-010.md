# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (1-10)

### 1. Chuqur o‘qitish (deep learning) ning klassik mashinaviy o‘qitish (machine learning) dan farqini, misollar asosida tushuntirib bering.

**Javob:**
Chuqur o‘qitish (Deep Learning - DL) va klassik mashinaviy o‘qitish (Machine Learning - ML) sun’iy intellektning ikki muhim tarmog‘i bo‘lib, ular o‘rtasidagi asosiy farq ma’lumotlarni qayta ishlash, o‘rganish uslubi va inson aralashuvi darajasida namoyon bo‘ladi.

**Asosiy farqlar:**
1.  **Xususiyatlarni ajratish (Feature Engineering):**
    *   **Klassik ML:** Bu yondashuvda modelga ma’lumot berishdan oldin, mutaxassis (inzhener) tomonidan ma’lumotlarning eng muhim belgi va xususiyatlari qo‘lda ajratib olinishi kerak. Masalan, tasvirni tanishda "burchaklar", "ranglar gistogrammasi" yoki "shakllar" kabi parametrlar oldindan dasturlashtiriladi yoki hisoblanadi. Model faqat shu tayyor xususiyatlar asosida qaror qabul qiladi.
    *   **Deep Learning:** Chuqur o‘qitishda xususiyatlarni qo‘lda ajratish shart emas. Model xom ma’lumotni (masalan, rasm piksellarini) to‘g‘ridan-to‘g‘ri qabul qiladi va o‘qitish jarayonida o‘zi eng muhim belgilarni (priznaklarni) avtomatik ravishda aniqlaydi. Quyi qatlamlar oddiy chiziqlarni, yuqori qatlamlar esa murakkab ob’ektlarni o‘rganadi.

2.  **Ma’lumotlar hajmi va samaradorlik:**
    *   **Klassik ML:** Kichik va o‘rta hajmdagi ma’lumotlar bilan juda yaxshi ishlaydi. Biroq, ma’lumotlar hajmi ortgani sari uning samaradorligi (aniqligi) ma’lum bir nuqtada to‘xtab qoladi (plato).
    *   **Deep Learning:** Katta hajmdagi ma’lumotlarni (Big Data) talab qiladi. Ma’lumot qancha ko‘p bo‘lsa, model shunchalik aniqroq ishlaydi va uning salohiyati ortib boradi.

3.  **Hisoblash resurslari:**
    *   **Klassik ML:** Oddiy protsessorlarda (CPU) ham tez ishlaydi va o‘qitish uchun kam vaqt talab qiladi.
    *   **Deep Learning:** Katta matritsali amallar va millionlab parametrlarni hisoblash uchun kuchli GPU (Grafik protsessor) yoki TPU (Tenzor protsessor) talab qiladi. O‘qitish soatlab yoki kunlab davom etishi mumkin.

**Misol:**
Tasavvur qiling, biz mushuk va itni ajratuvchi dastur tuzyapmiz.
*   **ML yondashuvi:** Biz dasturga "quloq shakli uchburchak bo‘lsa", "mo‘ylovi uzun bo‘lsa" kabi qoidalarni yoki matematik belgilarni qo‘lda kiritishimiz kerak. Agar mushukning rasmi g‘alati burchakdan tushirilgan bo‘lsa, bizning qoidalarimiz ishlamay qolishi mumkin.
*   **DL yondashuvi:** Biz tizimga minglab mushuk va it rasmlarini shunchaki "bu mushuk", "bu it" deb ko‘rsatamiz. Neyron tarmoq o‘zi piksellar orasidagi bog‘liqlikni tahlil qilib, mushukni itdan ajratib turuvchi noyob naqshlarni (biz insonlar hatto sezmaydigan belgilarni ham) topadi.

---

### 2. Katta hajmdagi ma’lumotlar (Big Data) chuqur o‘qitishning rivojlanishida qanday rol o‘ynashini izohlab bering.

**Javob:**
"Big Data" chuqur o‘qitish inqilobining eng asosiy harakatlantiruvchi kuchidir. Chuqur neyron tarmoqlar (Deep Neural Networks) tabiatan "ochko‘z" algoritmlar hisoblanadi, ya’ni ular yuqori aniqlikka erishish uchun ulkan miqdordagi misollarga muhtoj.

**Roli va ahamiyati:**
1.  **Umumlashtirish qobiliyati (Generalization):** Chuqur modellar millionlab parametrlarga (vaznlarga) ega. Agar ma’lumotlar kam bo‘lsa, model bu ma’lumotlarni shunchaki yodlab oladi (overfitting) va yangi, ko‘rmagan ma’lumotlarda xato ishlaydi. Big Data modelga turli xil vaziyatlarni, burchaklarni, shovqinlarni va variatsiyalarni ko‘rish imkonini beradi. Natijada model yodlab olish o‘rniga, haqiqiy qonuniyatlarni "tushunishga" majbur bo‘ladi.
2.  **Murakkablikni qamrab olish:** Dunyodagi muammolar (masalan, inson nutqini tushunish, avtonom boshqaruv) juda murakkab va chiziqli emas. Katta ma’lumotlar to‘plami ushbu murakkablikning barcha qirralarini o‘z ichiga oladi. Masalan, GPT-4 kabi modellar butun internetdagi matnlar asosida o‘qitilgani uchun deyarli har qanday mavzuda gaplasha oladi.
3.  **Algoritmik ustunlik:** 2010-yillargacha neyron tarmoqlar samarasiz deb hisoblangan, chunki ular kichik ma’lumotlarda oddiy algoritmlardan (masalan, SVM yoki Random Forest) yutqazardi. Faqat ImageNet kabi millionlab rasmlardan iborat bazalar paydo bo‘lgach, chuqur o‘qitish o‘zining haqiqiy kuchini ko‘rsatdi va boshqa barcha usullarni ortda qoldirdi.

---

### 3. Nega chuqur neyron tarmoqlar katta hisoblash resurslarini talab qiladi? Izohlab bering.

**Javob:**
Chuqur neyron tarmoqlarni o‘qitish va ishlatish kompyuter texnikasining eng og‘ir vazifalaridan biridir. Buning bir nechta asosiy sabablari bor:

1.  **Parametrlar sonining ko‘pligi:** Zamonaviy chuqur modellar (masalan, LLMlar yoki zamonaviy ko‘rish modellari) millionlab, hatto milliardlab va trillionlab parametrlarga (vaznlar va siljishlar) ega. Har bir o‘qitish qadamida ushbu milliardlab qiymatlarni yangilab turish kerak.
2.  **Matritsali va Tenzorli amallar:** Neyron tarmoqning har bir qatlami aslida ulkan matritsalarni bir-biriga ko‘paytirishdan iborat. Masalan, 1024 ta neyronli qatlamni yana shunday qatlamga ulash 1 milliondan ortiq ko‘paytirish va qo‘shish amalini talab qiladi. Chuqur tarmoqda bunday qatlamlar yuzlab bo‘lishi mumkin.
3.  **Backpropagation (Orqaga tarqalish):** O‘qitish jarayonida "gradient" hisoblash talab etiladi. Bu degani, xatolik funksiyasidan har bir parametrgacha bo‘lgan hosilalarni zanjir qoidasi bo‘yicha hisoblab chiqish kerak. Bu juda katta hajmdagi suzuvchi nuqtali (floating-point) hisob-kitoblarni talab qiladi.
4.  **Iterativ jarayon:** Model bir marta ko‘rib chiqishda o‘rganmaydi. U ma’lumotlar bazasini yuzlab yoki minglab marta (epoxalar) qayta-qayta ko‘rib chiqishi kerak. Bu esa umumiy hisoblash hajmini ming barobar oshiradi.
5.  **Parallelizatsiya zarurati:** Oddiy protsessorlar (CPU) ketma-ket ishlashga mo‘ljallangan bo‘lib, bu turdagi hisoblashlar uchun sekinlik qiladi. Shu sababli, minglab kichik yadrolarga ega bo‘lgan va parallel ravishda millionlab amallarni bajara oladigan GPU (Videokarta) va maxsus TPU chiplari zarur bo‘ladi.

---

### 4. “Model chuqurligi” (model depth) tushunchasini va uning o‘qitish jarayoniga ta’sirini tushuntirib bering.

**Javob:**
**Tushuncha:** "Model chuqurligi" neyron tarmog‘idagi yashirin qatlamlar (hidden layers) sonini bildiradi. Kirish va chiqish qatlamlari orasida qanchalik ko‘p qatlam joylashgan bo‘lsa, model shunchalik "chuqur" hisoblanadi. Masalan, 2 ta yashirin qatlamli model "sayoz" (shallow), 150 ta qatlamli model esa "juda chuqur" hisoblanadi.

**O‘qitish jarayoniga ta’siri:**
1.  **Ijobiy ta’sir (Abstraksiya):** Chuqurlik modelga ma’lumotlarni ierarxik tarzda o‘rganish imkonini beradi. Har bir yangi qatlam oldingi qatlam ma’lumotlarini birlashtirib, murakkabroq tushunchani hosil qiladi. Chuqur modellar kamroq neyronlar bilan ham juda murakkab funksiyalarni approksimatsiya qila oladi (samaradorlik).
2.  **Salbiy ta’sir (O‘qitish qiyinligi):** Model chuqurlashgani sari, signal (ma’lumot) va gradient (xatolik) tarmoq bo‘ylab uzoq masofaga sayohat qilishi kerak. Bu quyidagi muammolarni keltirib chiqaradi:
    *   **Vanishing Gradient (Gradient yo‘qolishi):** Xatolik signali orqaga qaytishda kichrayib-kichrayib, birinchi qatlamlarga yetib bormay qoladi. Natijada modelning boshi o‘rganmaydi.
    *   **Optimization Landscape:** Chuqur modellarning xatolik fazosi juda g‘adir-budur bo‘lib, unda lokal minimumlar va egar nuqtalar (saddle points) ko‘p bo‘ladi, bu esa optimal yechimni topishni qiyinlashtiradi.
    *   **Resurs talabi:** Chuqurlik ortishi xotira va hisoblash vaqtini oshiradi.

Zamonaviy arxitekturalar (masalan, ResNet) maxsus "skip connections" (sakrab o‘tish) usullari orqali juda chuqur tarmoqlarni ham samarali o‘qitish imkonini berdi.

---

### 5. Chuqur tarmoqlarda xususiyatlarni ifodalash (feature representation) kontseptsiyasini tushuntirib bering.

**Javob:**
Xususiyatlarni ifodalash (Feature Representation yoki Representation Learning) — bu ma’lumotlarni mashina tushunishi oson bo‘lgan formatga o‘tkazish jarayonidir. Chuqur o‘qitishda bu jarayon **ierarxik** (bosqichma-bosqich) tarzda amalga oshiriladi.

**Kontseptsiya mohiyati:**
Oddiy kompyuter uchun rasm — bu shunchaki sonlar (piksellar) to‘plami. Bu sonlar o‘z-o‘zidan hech qanday ma’noga ega emas (masalan, qaysi piksel mushukning ko‘zi ekanligi noaniq). Chuqur tarmoqning maqsadi ushbu "xom" piksellarni yuqori darajadagi tushunchalarga aylantirishdir.

1.  **Quyi darajadagi ifoda:** Tarmoqning dastlabki qatlamlari piksellardagi keskin o‘zgarishlarni aniqlaydi va ularni "gorizontal chiziq", "vertikal chiziq", "burchak" yoki "rang dog‘i" sifatida ifodalaydi.
2.  **O‘rta darajadagi ifoda:** Keyingi qatlamlar ushbu chiziqlarni birlashtirib, oddiy geometrik shakllarni (aylana, kvadrat) yoki teksturalarni (yung, g‘isht devor) ifodalaydi.
3.  **Yuqori darajadagi ifoda:** Eng oxirgi yashirin qatlamlar shakllarni birlashtirib, to‘liq ob’ekt qismlarini (ko‘z, g‘ildirak, burun) va oxir-oqibat butun ob’ektni (mashina, odam yuzi) ifodalovchi vektorlarni hosil qiladi.

Bu jarayon ma’lumotni "chiziqli ajratib bo‘lmaydigan" holatdan "chiziqli ajratib bo‘ladigan" fazoga o‘tkazish deb ham ataladi. Ya’ni, oxirgi qatlamdagi ifoda (vektor) shunday ko‘rinishga keladiki, uni oddiy chiziq bilan ham klassifikatsiya qilish mumkin bo‘ladi.

---

### 6. Nega chuqur modellarda xususiyatlarni (priznaklarni) avtomatik ajratib olish imkoniyati mavjud? Izohlab bering.

**Javob:**
Chuqur modellar sehrli tarzda emas, balki qat’iy matematik optimallashtirish tamoyillari asosida xususiyatlarni ajratib oladi. Buning sabablari quyidagilardir:

1.  **Diferensiallanuvchi struktura:** Neyron tarmoqlar boshidan oxirigacha uzluksiz, hosilasi olinadigan funksiyalardan tashkil topgan. Bu shuni anglatadiki, biz chiqishdagi xatolikni (loss) kirishdagi har bir parametrga bog‘liq holda hisoblay olamiz.
2.  **Maqsad funksiyasi (Loss Function):** Modelga yagona maqsad qo‘yiladi: xatolikni kamaytirish. Model ushbu maqsadga erishish uchun o‘zining ichki parametrlarini (filtrlarini) shunday o‘zgartiradiki, ular ma’lumotdagi eng foydali naqshlarni tutib qolsin. Agar "quloq shakli"ni aniqlash xatoni kamaytirishga yordam bersa, modeldagi ma’lum bir filtrlar aynan quloq shakliga reaksiyaga kirishadigan bo‘lib o‘zgaradi.
3.  **Filtrlarning moslashuvchanligi:** Konvolyutsion tarmoqlarda (CNN) filtrlar (yadrolar) dastlab tasodifiy sonlardan iborat bo‘ladi. O‘qitish davomida Gradient Tushish (Gradient Descent) algoritmi bu filtrlarni rasmning muhim qismlarini "yoritib beradigan" detektorlarga aylantiradi. Inson aralashuvisiz, matematik optimizatsiya jarayoni eng informativ belgilarni saqlab qolishni "tanlaydi".

---

### 7. Chuqur o‘qitishda taqsimlangan bilim ifodasi (distributed representation) tushunchasini tushuntirib bering.

**Javob:**
Taqsimlangan bilim ifodasi — bu neyron tarmoqlarning axborotni saqlash va kodlash usulidir. Bu g‘oya "lokalistik" ifodaga qarama-qarshi turadi.

*   **Lokalistik ifoda (One-hot encoding):** Har bir tushuncha bitta alohida element (neyron) bilan ifodalanadi. Masalan, "olma" uchun 1-neyron, "nok" uchun 2-neyron javob beradi. Agar bizda 1000 xil meva bo‘lsa, 1000 ta neyron kerak va ular bir-biri bilan bog‘liq emas.
*   **Taqsimlangan ifoda (Distributed):** Bitta tushuncha ko‘plab neyronlarning faollashuvi kombinatsiyasi orqali ifodalanadi va bitta neyron ko‘plab tushunchalarni ifodalashda qatnashadi.
    *   Masalan: "Olma" tushunchasi [Neyron A: 0.8, Neyron B: 0.1, Neyron C: 0.9] kabi vektor bilan ifodalanishi mumkin.
    *   "Nok" esa [Neyron A: 0.7, Neyron B: 0.2, Neyron C: 0.8] bo‘lishi mumkin.

**Afzalliklari:**
1.  **Samaradorlik:** $N$ ta neyron bilan $2^N$ ta tushunchani kodlash mumkin (kombinatsiyalar hisobiga). Bu xotirani tejaydi.
2.  **O‘xshashlikni tushunish:** Taqsimlangan ifodada vektorlar orqali semantik yaqinlikni aniqlash mumkin. "Olma" va "Nok"ning vektorlari bir-biriga yaqin bo‘ladi, "Olma" va "Mashina"niki esa uzoq. Bu modelga "generalizatsiya" (umumlashtirish) qilishga yordam beradi: agar model "olma" haqida biror narsa o‘rgansa, bu bilim qisman "nok"ka ham tatbiq etilishi mumkin, chunki ular o‘xshash neyronlarni faollashtiradi.

---

### 8. Yuzaki (sayoz) va chuqur modellarning funksiyalarni yaqinlashtirish (approksimatsiya) imkoniyatlarini taqqoslab bering.

**Javob:**
Matematik jihatdan neyron tarmoqlar "universal approksimatorlar" hisoblanadi. Ya’ni, ular har qanday murakkab funksiyani (bog‘liqlikni) o‘zlarida modellashtira oladilar.

*   **Yuzaki (Sayoz) modellar:** "Universal Approksimatsiya Teoremasi"ga ko‘ra, atigi bitta yashirin qatlamga ega bo‘lgan neyron tarmoq ham har qanday uzluksiz funksiyani istalgan aniqlikda ifodalay oladi. **Lekin**, buning uchun o‘sha yagona yashirin qatlamda **haddan tashqari ko‘p (eksponentsial darajada)** neyronlar bo‘lishi talab etiladi. Bu amaliy jihatdan imkonsiz va samarasizdir, chunki parametrlar soni oshib ketib, modelni o‘qitish qiyinlashadi va u osonlikcha "overfitting"ga uchraydi.
*   **Chuqur modellar:** Chuqur tarmoqlar xuddi shu funksiyani ancha **kamroq parametrlar** (neyronlar) yordamida ifodalay oladi. Sababi, chuqur modellar funksiyani kompozitsiya (funksiya ichida funksiya) sifatida quradi. Bu tabiatdagi ko‘plab jarayonlarga (fraktallar, ierarxik tuzilmalar) mos keladi. Chuqur model muammoni mayda bo‘laklarga bo‘lib, har bir qatlamda yechimning bir qismini hal qiladi.

**Xulosa:** Sayoz modellar nazariy jihatdan hamma narsaga qodir bo‘lsa-da, chuqur modellar buni **samaraliroq** (kam resurs va yaxshiroq umumlashtirish bilan) amalga oshiradi. Chuqurlik — bu "ifodalash kuchi"ning (expressive power) tejamkor yo‘lidir.

---

### 9. Katta hajmdagi ma’lumotlar mavjud bo‘lganda, nega chuqur tarmoqlar overfitting ga kamroq moyil bo‘lishi mumkinligini tushuntirib bering.

**Javob:**
Overfitting (qayta moslashish) — bu model o‘qitish ma’lumotlarini yodlab olib, yangi ma’lumotlarda xato ishlash holatidir. Odatda, model qanchalik katta va murakkab bo‘lsa, overfitting xavfi shuncha yuqori bo‘ladi. Biroq, katta ma’lumotlar (Big Data) bu xavfni kamaytiradi.

1.  **Haqiqiy taqsimotni qamrab olish:** Kichik ma’lumotlar to‘plamida tasodifiy shovqinlar yoki kam uchraydigan holatlar umumiy qonuniyat kabi ko‘rinishi mumkin. Model bu "soxta" qonuniyatlarni o‘rganib oladi. Katta ma’lumotlarda esa misollar shunchalik ko‘pki, tasodifiy shovqinlar o‘zaro yo‘qolib ketadi va faqat barqaror, haqiqiy qonuniyatlar (signal) ustunlik qiladi. Model shovqinni yodlashga ulgurmaydi yoki bu unga foyda keltirmaydi.
2.  **Cheklovchi omil:** Ma’lumotlar hajmi modelning "sig‘imi"ga (capacity) qarshi turuvchi kuchdir. Agar sizda 1 million parametrli model bo‘lsa-yu, atigi 10 mingta rasm bo‘lsa, model har bir rasmni alohida parametrlarga joylab, yodlab oladi. Lekin 1 million parametrga 100 million rasm to‘g‘ri kelsa, model har bir rasmni yodlay olmaydi. U majburan rasmlar orasidagi **umumiy o‘xshashliklarni** qidirishga va shularni saqlashga o‘tadi. Bu esa to‘g‘ridan-to‘g‘ri "generalizatsiya" (umumlashtirish) demakdir.

Shu sababli, chuqur o‘qitishda eng yaxshi regulyarizatsiya (overfittingga qarshi dori) — bu ko‘proq ma’lumot yig‘ishdir.

---

### 10. Ko‘p qatlamli perseptronning klassik arxitekturasini tasvirlab bering va uning chuqur o‘qitishdagi rolini izohlang.

**Javob:**
Ko‘p qatlamli perseptron (Multilayer Perceptron - MLP) — bu eng sodda va fundamental chuqur neyron tarmoq turidir.

**Arxitekturasi:**
MLP kamida uchta asosiy qismdan iborat bo‘ladi:
1.  **Kirish qatlami (Input Layer):** Ma’lumotlarni qabul qiladi (masalan, rasm piksellari vektori). Bu qatlamda hech qanday hisob-kitob bajarilmaydi, u faqat uzatuvchi vazifasini o‘taydi.
2.  **Yashirin qatlamlar (Hidden Layers):** Kirish va chiqish orasida joylashgan bir yoki bir nechta qatlamlar. Aynan shu yerda asosiy hisob-kitoblar va xususiyatlarni ajratib olish jarayoni sodir bo‘ladi. Har bir neyron oldingi qatlamdagi barcha neyronlar bilan bog‘langan (Fully Connected - To‘liq bog‘langan). Har bir bog‘lanish o‘z vazniga ($w$) ega. Neyron qiymatlarni vaznlarga ko‘paytirib yig‘adi va **faollashtirish funksiyasini** (masalan, ReLU yoki Sigmoid) qo‘llaydi. Bu funksiyalar tarmoqqa chiziqli bo‘lmagan murakkablikni kiritadi.
3.  **Chiqish qatlami (Output Layer):** Modelning yakuniy bashoratini beradi. Tasniflash masalasida bu qatlamdagi neyronlar soni sinflar soniga teng bo‘ladi va ko‘pincha Softmax funksiyasi orqali ehtimolliklarni qaytaradi.

**Chuqur o‘qitishdagi roli:**
MLP chuqur o‘qitishning "g‘ishti" hisoblanadi.
*   Garchi zamonaviy tarmoqlarda (CNN, Transformer) maxsus qatlamlar (konvolyutsiya, attention) ishlatilsa-da, deyarli har qanday arxitekturaning oxirida yoki oralarida MLP bloklari (Dense layers) ishlatiladi.
*   Masalan, CNN rasm xususiyatlarini ajratib olgandan so‘ng, oxirgi qarorni qabul qilish (bu it yoki mushuk deyish) uchun aynan MLP qatlamiga uzatadi.
*   Transformerlarda "Feed-Forward Network" deb ataladigan blok aslida MLPdir.
Shunday qilib, MLP — ma’lumotlarni yakuniy qayta ishlash va qaror qabul qilish mexanizmi sifatida barcha murakkab tizimlarning ajralmas qismidir.
