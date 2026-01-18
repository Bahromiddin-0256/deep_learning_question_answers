# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (11-20)

### 11. Chuqur modellarda yo‘qotish funksiyasining (loss function) vazifasini va ahamiyatini izohlab bering.

**Javob:**
Yo‘qotish funksiyasi (Loss Function yoki Cost Function) — bu neyron tarmoq o‘qitish jarayonining eng muhim "kompasi"dir. U modelning qanchalik yomon ishlayotganini sonli ko‘rinishda ifodalaydi.

**Vazifasi:**
Model bashorat qilgan natija (\(\hat{y}\)) va haqiqiy to‘g‘ri javob (\(y\)) o‘rtasidagi farqni hisoblash. Agar model "bu rasm it" desa-yu, aslida rasmda mushuk bo‘lsa, yo‘qotish funksiyasi katta qiymat (xatolik) qaytaradi. Agar model to‘g‘ri topsa, qiymat nolga yaqin bo‘ladi.
*   Regressiya masalalari uchun ko‘pincha **MSE (Mean Squared Error)** ishlatiladi.
*   Tasniflash (klassifikatsiya) masalalari uchun **Cross-Entropy Loss** ishlatiladi.

**Ahamiyati:**
Neyron tarmoq o‘qitilayotganda, bizning maqsadimiz aynan shu funksiya qiymatini **minimallashtirishdir**. Butun optimizatsiya jarayoni (Gradient tushish) ushbu funksiyaga asoslanadi. Model o‘z vaznlarini (parametrlarini) tasodifiy o‘zgartirmaydi, balki yo‘qotish funksiyasini kamaytirish yo‘nalishida o‘zgartiradi. Agar yo‘qotish funksiyasi noto‘g‘ri tanlansa yoki noto‘g‘ri ishlasa, model hech qachon o‘rganmaydi, chunki u "yaxshi" va "yomon" natija nima ekanligini ajrata olmay qoladi.

---

### 12. Gradient tushunchasini ta’riflab bering va nima uchun u orqaga tarqalish (backpropagation) algoritmi uchun muhimligini tushuntiring.

**Javob:**
**Ta’rif:**
Gradient — bu ko‘p o‘zgaruvchili funksiyaning (bizning holatda Loss funksiyasining) o‘zgarish tezligi va yo‘nalishini ko‘rsatuvchi vektordir. Oddiy qilib aytganda, gradient bizga funksiyaning eng tik o‘sish yo‘nalishini ko‘rsatadi. Uning har bir elementi xususiy hosiladan iborat (\(\nabla L = [\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, ...]\)).

**Backpropagation uchun ahamiyati:**
Backpropagation (orqaga tarqalish) algoritmining maqsadi — tarmoqdagi har bir vazn (\(w\)) umumiy xatolikka qanchalik hissa qo‘shganini aniqlashdir.
1.  Model xato qildi. Biz xatoni kamaytirmoqchimiz.
2.  Buning uchun vaznlarni qaysi tomonga o‘zgartirish kerak? (Oshirish kerakmi yoki kamaytirish?)
3.  Gradient aynan shu savolga javob beradi. Agar gradient musbat bo‘lsa, demak vaznni kamaytirish kerak. Agar manfiy bo‘lsa, oshirish kerak.
4.  Backpropagation zanjir qoidasi (Chain Rule) yordamida oxirgi qatlamdan boshlab birinchi qatlamgacha bo‘lgan gradientlarni hisoblab chiqadi. Gradientsiz biz vaznlarni qanday yangilashni bilmagan bo‘lardik va modelni o‘qita olmasdik.

---

### 13. Gradient tushish (gradient descent) usulining ishlash mexanizmini bayon qilib bering.

**Javob:**
Gradient tushish — bu funksiyaning minimum nuqtasini (eng kichik qiymatini) topish uchun ishlatiladigan iterativ optimizatsiya algoritmidir. Uni ko‘pincha "tog‘dan vodiyga tushayotgan ko‘r sayyoh"ga o‘xshatishadi.

**Ishlash mexanizmi:**
1.  **Boshlash:** Model parametrlari (vaznlar) tasodifiy qiymatlar bilan initsializatsiya qilinadi. Biz tog‘ning qayeridadir turibmiz.
2.  **Gradientni hisoblash:** Joriy nuqtada Loss funksiyaning gradienti hisoblanadi. Gradient bizga "yuqoriga eng tik chiqish" yo‘nalishini ko‘rsatadi.
3.  **Qadam tashlash:** Biz pastga tushmoqchi bo‘lganimiz uchun, gradientga **qarama-qarshi** yo‘nalishda qadam tashlaymiz.
4.  **Yangilash formulasi:** \(w_{yangi} = w_{eski} - \eta \cdot \nabla L\).
    *   \(w\) — vazn.
    *   \(\nabla L\) — gradient.
    *   \(\eta\) (eta) — o‘qitish tezligi (learning rate), qadam kattaligini belgilaydi.
5.  **Takrorlash:** Bu jarayon (2-4 qadamlar) xatolik minimumga yetguncha yoki ma’lum sonli epoxalar davomida takrorlanadi.

Natijada, har bir qadamda xatolik kamayib boradi va model "o‘rganadi".

---

### 14. Batch, mini-batch va stoxastik gradient tushish (stochastic gradient descent) usullarining farqlarini tushuntirib bering.

**Javob:**
Bu usullar gradientni hisoblashda qancha ma’lumotdan foydalanishiga qarab farqlanadi.

1.  **Batch Gradient Descent:**
    *   **Ishlash:** Vaznlarni yangilashdan oldin **butun ma’lumotlar to‘plamini** (dataset) ko‘rib chiqadi va o‘rtacha gradientni hisoblaydi.
    *   **Afzalligi:** Gradient aniq va barqaror bo‘ladi. Konvergensiya (yaqinlashish) tekis kechadi.
    *   **Kamchiligi:** Juda sekin. Katta datasetlar xotiraga sig‘masligi mumkin.

2.  **Stochastic Gradient Descent (SGD):**
    *   **Ishlash:** Har **bir dona** misol (sample) uchun gradient hisoblaydi va darhol vaznlarni yangilaydi.
    *   **Afzalligi:** Juda tez ishga tushadi. Lokal minimumlardan sakrab o‘tib ketish ehtimoli bor.
    *   **Kamchiligi:** Gradient juda shovqinli bo‘ladi (u yoqqa, bu yoqqa sakraydi). Optimal nuqta atrofida aylanib yurishi mumkin, aniq tushmasligi mumkin.

3.  **Mini-batch Gradient Descent:**
    *   **Ishlash:** Oltin o‘rtalik. Ma’lumotlarni kichik guruhlarga (batch) bo‘ladi (masalan, 32, 64 yoki 128 ta misol). Har bir guruh uchun o‘rtacha gradient hisoblab, vaznlarni yangilaydi.
    *   **Afzalligi:** Batch usulining barqarorligini va SGDning tezligini birlashtiradi. GPU parallel hisoblash imkoniyatlaridan samarali foydalanadi. Amaliyotda deyarli doim shu usul ishlatiladi.

---

### 15. O‘qitish tezligi (learning rate) parametrining model o‘qitilishiga ta’sirini izohlab bering.

**Javob:**
O‘qitish tezligi (\(\eta\)) — bu gradient tushish paytida qadamning kattaligini belgilovchi giperparametrdir. U model o‘qishining eng muhim sozlamalaridan biridir.

*   **Juda katta Learning Rate:**
    *   Model "ulkan qadamlar" bilan harakatlanadi.
    *   **Xavfi:** Minimum nuqtasidan sakrab o‘tib ketishi mumkin. Xatolik kamayish o‘rniga oshib ketishi (divergensiya) va model "portlab ketishi" (NaN qiymatlar) mumkin.
*   **Juda kichik Learning Rate:**
    *   Model "chumoli qadamlari" bilan harakatlanadi.
    *   **Xavfi:** O‘qitish jarayoni haddan tashqari uzoq vaqt oladi. Bundan tashqari, model global minimumga yetib bormasdan, yo‘ldagi kichik chuqurliklarda (lokal minimum) qolib ketishi ehtimoli yuqori.
*   **Optimal Learning Rate:**
    *   Model tezda minimumni topadi va barqarorlashadi. Ko‘pincha "Learning Rate Decay" (o‘qish davomida tezlikni kamaytirib borish) usuli qo‘llaniladi: boshida katta qadamlar, oxirida aniqroq joylashish uchun kichik qadamlar.

---

### 16. “Exploding gradients” (portlab ketuvchi gradientlar) nima va u qaysi omillar sababli yuzaga kelishi mumkinligini tushuntiring.

**Javob:**
**Muammo:**
O‘qitish jarayonida (ayniqsa, orqaga tarqalish paytida) gradient qiymatlarining haddan tashqari kattalashib ketishi. Vaznlar gradientga qarab yangilangani uchun, katta gradient vaznlarni ham ulkan sonlarga aylantirib yuboradi. Natijada hisob-kitoblarda "Number Overflow" (son sig‘may qolishi) yuz berib, model NaN (Not a Number) qaytara boshlaydi.

**Sabablari:**
1.  **Chuqur tarmoqlar:** Gradient zanjir qoidasi bo‘yicha ko‘paytirilgani uchun, agar har bir qatlamdagi hosila 1 dan katta bo‘lsa (masalan, 2), chuqur tarmoqda bu qiymat eksponentsial o‘sib ketadi (\(2^{100}\) juda katta son).
2.  **Noto‘g‘ri initsializatsiya:** Agar vaznlar boshida juda katta qiymatlar bilan boshlansa.
3.  **RNN (Rekurrent tarmoqlar):** Vaqt bo‘yicha orqaga qaytishda (BPTT) bir xil vazn ko‘p marta ko‘paytirilgani uchun bu muammo RNNlarda juda tez-tez uchraydi.

**Yechim:** Gradient Clipping (gradientni qirqish) — gradient normasi ma’lum chegaradan oshsa, uni majburan kichraytirish.

---

### 17. “Vanishing gradients” (yo‘qolib ketuvchi gradientlar) nima va u nimasi bilan xavfli ekanini izohlab bering.

**Javob:**
**Muammo:**
Bu "Exploding gradients"ning teskarisi va ancha keng tarqalgan muammo. Orqaga tarqalish jarayonida gradient qiymatlari qatlamdan-qatlamga o‘tgan sari kichrayib, nolga yaqinlashib qoladi.

**Xavfi:**
Tarmoqning kirish qismiga (boshiga) yaqin joylashgan qatlamlar uchun hisoblangan gradient deyarli 0 ga teng bo‘lib qoladi.
*   Vaznlar yangilash formulasi: \(w = w - \eta \cdot \nabla L\).
*   Agar \(\nabla L \approx 0\) bo‘lsa, \(w\) o‘zgarmaydi.
*   Natijada, chuqur tarmoqning boshlang‘ich qatlamlari (eng muhim xususiyatlarni ajratuvchi qism) o‘rganishdan to‘xtaydi. Model xuddi sayoz model kabi ishlay boshlaydi yoki umuman o‘qimaydi.

**Asosiy sababchi:** Sigmoid yoki Tanh kabi faollashtirish funksiyalari. Ularning hosilasi har doim 1 dan kichik (Sigmoidda max 0.25). Ko‘p qatlamli tarmoqda 0.25 ni qayta-qayta ko‘paytirish gradientni yo‘q qilib yuboradi.

---

### 18. Nega ReLU faollashtirish funksiyasi yo‘qolib ketuvchi gradientlar muammosini ma’lum darajada kamaytirishga yordam beradi? Tushuntirib bering.

**Javob:**
ReLU (Rectified Linear Unit) funksiyasi: \(f(x) = \max(0, x)\).
Uning hosilasi juda oddiy:
*   Agar \(x > 0\) bo‘lsa, hosila \(1\) ga teng.
*   Agar \(x \le 0\) bo‘lsa, hosila \(0\) ga teng.

**Yordami:**
Sigmoid funksiyasida hosila har doim 1 dan kichik bo‘lib, ko‘paytirilganda signalni so‘ndirardi. ReLUda esa musbat qiymatlar uchun hosila aynan **1 ga teng**.
*   Matematik jihatdan: \(1 \times 1 	imes 1 	imes \dots \times 1 = 1\).
*   Bu shuni anglatadiki, gradient chuqur qatlamlar orqali hech qanday yo‘qotishsiz to‘g‘ridan-to‘g‘ri orqaga o‘ta oladi. Bu xususiyat juda chuqur tarmoqlarni (masalan, yuzlab qatlamli CNNlarni) samarali o‘qitish imkonini yaratdi va Deep Learning inqilobining asosiy sababchilaridan biri bo‘ldi.

---

### 19. SGD, Momentum va Nesterov optimizatorlarini taqqoslab, ularning farqli jihatlarini izohlab bering.

**Javob:**
Uchala usul ham gradient tushishining variatsiyalaridir.

1.  **SGD (Standard):**
    *   Faqat **hozirgi** qadamdagi gradientga qaraydi.
    *   **Muammo:** Agar vodiy yuzasi notekis bo‘lsa, SGD u yoqdan-bu yoqqa qattiq tebranadi va manzilga sekin boradi.
2.  **Momentum (Impuls):**
    *   Fizikadagi inersiya qonuniga asoslanadi. U oldingi qadamlar yo‘nalishini "eslab qoladi" va o‘sha tomonga harakatni davom ettiradi.
    *   Formula: \(v_t = \gamma v_{t-1} + \eta \nabla L\). Vazn: \(w = w - v_t\).
    *   **Farqi:** Agar gradient vaqtinchalik noto‘g‘ri tomonga o‘zgarsa ham, momentum uni to‘g‘ri yo‘lda ushlab turadi. Tebranishlarni kamaytiradi va tezlikni oshiradi.
3.  **Nesterov Accelerated Gradient (NAG):**
    *   Momentumning "aqlliroq" versiyasi.
    *   **Farqi:** Oddiy Momentum avval joriy joyda gradient hisoblaydi, keyin sakraydi. Nesterov esa "Men baribir impuls hisobiga oldinga sakrayman, keling, o‘sha **sakrab tushadigan joyimdagi** gradientni hisoblayman" deydi. Bu kelajakni oldindan ko‘rishga o‘xshaydi va modelga to‘siqlarga yaqinlashganda oldinroq "tormozlash" imkonini beradi. Bu konvergensiyani yanada barqaror qiladi.

---

### 20. Adam optimizatorining SGD ga nisbatan asosiy afzalliklarini tushuntirib bering.

**Javob:**
Adam (Adaptive Moment Estimation) — hozirgi kunda eng ommabop optimizator.

**SGD ga nisbatan afzalliklari:**
1.  **Adaptiv Learning Rate:**
    *   SGD barcha parametrlar (vaznlar) uchun bitta umumiy o‘qitish tezligini (\(\eta\)) ishlatadi. Bu muammoli, chunki ba’zi parametrlar tez, ba’zilari sekin o‘zgarishi kerak bo‘lishi mumkin.
    *   Adam har bir parametr uchun **alohida** learning rate hisoblaydi. Tez-tez uchraydigan (gradienti katta) parametrlar uchun qadamni kichraytiradi, kam uchraydiganlar uchun esa kattalashtiradi.
2.  **Momentum va RMSProp kombinatsiyasi:**
    *   Adam o‘zida Momentum (birinchi tartibli moment - o‘rtacha yo‘nalish) va RMSProp (ikkinchi tartibli moment - gradient dispersiyasi) g‘oyalarini birlashtiradi. Bu unga ham tezlikni (impulsni) saqlash, ham tebranishlarni so‘ndirish imkonini beradi.
3.  **Bias Correction (Siljishni tuzatish):**
    *   O‘qitish boshida o‘rtacha qiymatlar nolga yaqin bo‘lib qolishi mumkin. Adam maxsus formulalar orqali buni to‘g‘rilaydi va boshdanoq samarali ishlaydi.
4.  **Foydalanish qulayligi:**
    *   Adam odatda standart giperparametrlar bilan ham juda yaxshi ishlaydi, uni sozlash (tuning) SGDga qaraganda osonroq.

Xulosa qilib aytganda, Adam "avtomatik uzatmalar qutisi"ga o‘xshaydi, u yo‘l sharoitiga qarab tezlikni o‘zi moslashtiradi, SGD esa "mexanik uzatma" bo‘lib, hamma narsani qo‘lda boshqarishni talab qiladi.