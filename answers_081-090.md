# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (81-90)

### 81. LSTM va GRU ni arxitektura va amaliy qo‘llanilishi bo‘yicha taqqoslab bering.

**Javob:**
**Arxitektura:**
*   **LSTM (3 ta darvoza, 2 ta holat):** Forget, Input, Output darvozalari hamda Cell State va Hidden State mavjud. Bu unga xotirani juda nozik boshqarish imkonini beradi.
*   **GRU (2 ta darvoza, 1 ta holat):** Update va Reset darvozalari, faqat Hidden State mavjud. U matematik jihatdan soddaroq.

**Amaliy qo‘llanilishi:**
*   **LSTM:** Agar sizda ma’lumotlar juda ko‘p (Big Data) bo‘lsa va ketma-ketliklar juda uzun va murakkab bo‘lsa (masalan, uzun matnlarni tarjima qilish yoki genetik kodni tahlil qilish), LSTM ning "kuchliroq xotirasi" (Cell State) afzalroq bo‘lishi mumkin.
*   **GRU:** Agar hisoblash resurslari cheklangan bo‘lsa yoki ma’lumotlar hajmi kichikroq bo‘lsa, GRU tanlanadi. U tezroq o‘qiydi (konvergensiya) va kamroq parametrga ega bo‘lgani uchun overfittingga kamroq moyil.
*   **Umumiy xulosa:** Ko‘p holatlarda ikkisining natijasi deyarli bir xil, shuning uchun odatda GRUdan boshlab ko‘riladi, agar yetarli bo‘lmasa, LSTMga o‘tiladi.

---

### 82. Qaysi holatlarda RNN ni CNN ga nisbatan afzal qo‘llash mumkinligini izohlab bering.

**Javob:**
Garchi CNNlar ham ba’zan matn bilan ishlay olsa-da (1D Convolution), RNNlar quyidagi holatlarda tengsizdir:

1.  **O‘zgaruvchan uzunlik:** CNN kirish ma’lumoti qat’iy o‘lchamda bo‘lishini (yoki padding qilishni) talab qiladi. RNN esa istalgan uzunlikdagi ketma-ketlikni (1 ta so‘zdan 1000 ta so‘zgacha) tabiiy ravishda qabul qilaveradi.
2.  **Ketma-ketlik tartibi hal qiluvchi bo‘lganda:** Agar ma’lumotning ma’nosi uning tartibiga qattiq bog‘liq bo‘lsa (masalan, "Men uni urdim" va "U meni urdi" — so‘zlar bir xil, tartib boshqa), RNN bu tartibni "time step"lar orqali yaxshiroq saqlaydi. CNN esa lokallikni (yonma-yonlikni) ko‘radi, lekin uzoq masofali tartibni yo‘qotib qo‘yishi mumkin.
3.  **Holat saqlash:** Agar tizim "o‘tmishni eslab turishi" kerak bo‘lsa (masalan, chatbot suhbat davomini eslab turishi kerak), RNNning hidden state mexanizmi buning uchun ayni muddao.

---

### 83. Teacher forcing yondashuvi nima va u seq2seq modellarida qanday qo‘llanilishini tushuntiring.

**Javob:**
**Muammo:**
Seq2Seq modelni (masalan, tarjimonni) o‘qitayotganda, Dekoder (tarjima qiluvchi qism) har bir qadamda o‘zi oldingi qadamda chiqargan so‘zga tayanadi.
Boshida model ahmoq bo‘ladi va noto‘g‘ri so‘z chiqaradi. Keyingi qadamda u shu xato so‘zga asoslanib yanada battarroq xato qiladi. Jarayon "buzilgan telefon"ga aylanib ketadi va model o‘rganishi juda sekinlashadi.

**Yechim (Teacher Forcing):**
O‘qituvchi (Teacher) o‘quvchini majburlaydi (Forcing).
O‘qitish paytida biz Dekoderga uning **o‘zi chiqargan (ehtimol xato) so‘zni emas**, balki **haqiqiy to‘g‘ri javobni (Ground Truth)** keyingi qadam uchun kirish sifatida beramiz.
*   Model: "Men olma..." (xato, aslida "Men maktabga" bo‘lishi kerak edi).
*   Teacher Forcing: "Mayli, sen 'olma' deding, lekin keyingi so‘zni topish uchun faraz qilaylik, sen to‘g‘ri 'maktabga' deb aytgansan. Qani, davom et-chi?".

Bu o‘qitishni keskin tezlashtiradi va barqarorlashtiradi.

---

### 84. “Sequence-to-sequence” (ketma-ketlikdan ketma-ketlikka) model tushunchasini tushuntirib bering.

**Javob:**
Seq2Seq — bu bir turdagi ketma-ketlikni ikkinchi turdagi ketma-ketlikka aylantirib beruvchi arxitekturadir. U ikkita asosiy qismdan iborat:

1.  **Encoder (Kodlovchi):** Kirish ma’lumotini (masalan, o‘zbekcha gap) o‘qiydi va uni siqilgan **"Kontekst vektori"** (Context Vector) ga aylantiradi. Bu vektor gapning butun ma’nosini o‘zida jamlashi kerak.
2.  **Decoder (Dekodlovchi):** Shu kontekst vektorini oladi va undan bosqichma-bosqich yangi ketma-ketlikni (masalan, inglizcha tarjimasini) ishlab chiqaradi.

**Qo‘llanilishi:**
*   Mashina tarjimasi (Google Translate).
*   Chatbotlar (Savol -> Javob).
*   Matnni qisqartirish (Maqola -> Sarlavha).

---

### 85. Seq2seq modellarida e’tibor (attention) mexanizmining ishlash g‘oyasini izohlab bering.

**Javob:**
**Muammo:**
Oddiy Seq2Seq modelda Encoder butun uzun gapni bitta kichkina "Kontekst vektori"ga joylashga majbur. Agar gap juda uzun bo‘lsa, vektorga ma’lumot sig‘maydi va boshidagi so‘zlar unutilib ketadi ("Bottleneck" muammosi).

**Attention g‘oyasi:**
"Nega hammasini bitta vektorga tiqishimiz kerak?"
Attention mexanizmi Dekoderga tarjimaning har bir qadamida Encoderning **barcha holatlariga (hamma so‘zlariga) to‘g‘ridan-to‘g‘ri qarash** imkonini beradi.
*   Dekoder inglizcha "apple" so‘zini yozayotganda, butun o‘zbekcha gapga qaraydi va e’tiborini (attention) aynan "olma" so‘ziga (masalan 90% kuch bilan) qaratadi.
*   Keyingi so‘zda esa e’tiborini boshqa joyga ko‘chiradi.
Bu "qidiruv chirog‘i" (searchlight) kabi ishlaydi — model kerakli paytda kerakli ma’lumotni asl manbadan oladi.

---

### 86. Nega attention yondashuvi ko‘plab vazifalarda RNN larni almashtirgan deb qaraladi? Tushuntiring.

**Javob:**
Attention (xususan Transformer) RNNlarning ikkita eng katta kamchiligini yo‘q qildi:

1.  **Parallellik:** RNN ishlashi uchun oldingi so‘z hisoblanib bo‘linishini kutish shart (ketma-ketlik). Bu GPU imkoniyatlaridan to‘liq foydalanishga to‘sqinlik qiladi. Attention mexanizmi esa butun gapni bir vaqtning o‘zida (parallel) qayta ishlay oladi. Bu o‘qitish tezligini yuzlab barobar oshirdi.
2.  **Uzoq masofali bog‘liqlik:** RNNda 100-so‘z 1-so‘zni "eslashi" uchun signal 100 ta qatlamdan o‘tishi kerak (va yo‘qolib ketishi mumkin). Attentionda esa 100-so‘z va 1-so‘z o‘rtasidagi masofa **atigi 1 qadam**. Ular bir-biri bilan to‘g‘ridan-to‘g‘ri bog‘langan. Bu modelga juda uzun matnlarni ham mukammal tushunish imkonini beradi.

---

### 87. Self-attention (o‘z-o‘ziga e’tibor) mexanizmi nima ekanini tushuntirib bering.

**Javob:**
Self-Attention — bu bir ketma-ketlik ichidagi elementlarning **o‘zaro bog‘liqligini** aniqlash usulidir.
Biz boshqa tildagi gapga emas, balki gapning o‘ziga "e’tibor" qaratamiz.

**Misol:**
Gap: "Bank daryo bo‘yida joylashgan edi, chunki u suvga yaqin."
Biz (insonlar) bu yerdagi "u" so‘zi "Bank"ni anglatishini tushunamiz.
Self-Attention mexanizmi "u" so‘zini tahlil qilayotganda, gapdagi boshqa barcha so‘zlarga qaraydi va "Bank" so‘zi bilan eng kuchli bog‘liqlikni (yuqori e’tibor vaznini) topadi. Agar gap "Bank inqirozga uchradi, chunki u..." bo‘lsa, e’tibor "inqiroz" so‘ziga qaratilardi.
Bu modelga so‘zlarning **kontekstual ma’nosini** (polisemiyani) tushunishga yordam beradi.

---

### 88. Attention mexanizmida Query, Key va Value tushunchalarining rolini izohlab bering.

**Javob:**
Bu tushunchalar ma’lumotlar bazasidan qidirish (Retrieval) tizimidan olingan. Har bir so‘z (vektor) uchta rolga kiradi:

1.  **Query (So‘rov - Q):** "Men kimman va men nimani qidiryapman?" (Joriy so‘z).
2.  **Key (Kalit - K):** "Men qanday ma’lumotni taklif qila olaman?" (Boshqa barcha so‘zlarning "yorlig‘i").
3.  **Value (Qiymat - V):** "Mening asl mazmunim nima?" (So‘zning o‘zi).

**Jarayon:**
*   Joriy so‘zning **Query** vektori boshqa barcha so‘zlarning **Key** vektorlari bilan solishtiriladi (ko‘paytiriladi).
*   Qanchalik mos kelsa (o‘xshash bo‘lsa), shunchalik yuqori **Score (E’tibor bali)** hosil bo‘ladi.
*   Bu ballar asosida **Value** vektorlari og‘irlikli qo‘shiladi.
Natijada, joriy so‘z o‘ziga eng mos keladigan (bog‘liq) so‘zlarning ma’lumotini o‘ziga singdirib oladi.

---

### 89. Multi-head attention nima va nima uchun bir nechta “bosh” (head) qo‘llanilishini tushuntiring.

**Javob:**
**Tushuncha:**
Bitta Attention mexanizmini bir vaqtning o‘zida, parallel ravishda bir necha marta (masalan, 8 marta) qo‘llash. Har bir "bosh" (Head) o‘zining mustaqil Q, K, V matritsalariga ega.

**Sababi:**
So‘zlar o‘rtasidagi bog‘liqlik turli xil bo‘lishi mumkin:
*   1-bosh: **Grammatik** bog‘liqlikni o‘rganishi mumkin (Ega va Kesim).
*   2-bosh: **Semantik** (ma’no) bog‘liqlikni o‘rganishi mumkin ("u" -> "Bank").
*   3-bosh: **Vaqt/Joy** munosabatlarini o‘rganishi mumkin.

Agar bitta bosh bo‘lsa, u o‘rtacha bir narsani o‘rganadi. Ko‘p boshlar esa modelga matnni **turli rakurslardan** ko‘rish va boyroq ma’lumot olish imkonini beradi. Keyin barcha boshlarning natijalari birlashtiriladi.

---

### 90. Transformer modelida pozitsion kodlash (positional encoding) nima va nima uchun zarurligini izohlang.

**Javob:**
**Zarurati:**
RNN va CNNlarda ma’lumotning tartibi (kim birinchi, kim ikkinchi) arxitekturaning o‘ziga singdirilgan.
Lekin Transformerda (Attentionda) bunday tartib yo‘q. U barcha so‘zlarga bir vaqtda qaraydi. U uchun "Men sen" va "Sen men" degan gaplar to‘plami bir xil (xuddi qopdagi so‘zlar kabi).

**Yechim (Positional Encoding):**
Model so‘zlarning joylashuvini bilishi uchun, har bir so‘zning vektoriga uning **pozitsiyasini bildiruvchi maxsus vektor** qo‘shiladi.
Bu vektorlar sinus va kosinus funksiyalari orqali hosil qilinadi.
*   1-so‘zga: $P_1$ vektori qo‘shiladi.
*   2-so‘zga: $P_2$ vektori qo‘shiladi.
Natijada, "Men" so‘zi gapning boshida kelganda boshqacha, oxirida kelganda boshqacha vektorga ega bo‘ladi. Bu Transformerga tartibni tushunish imkonini beradi.