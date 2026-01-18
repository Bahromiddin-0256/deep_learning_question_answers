# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (111-120)

### 111. WGAN (Wasserstein GAN) g‘oyasini tushuntirib bering.

**Javob:**
**Muammo:**
Oddiy GANlarda "Jensen-Shannon" (JS) divergensiyasi ishlatiladi. Muammo shundaki, agar haqiqiy va soxta rasmlar taqsimoti bir-biriga umuman tegmasa (overlap bo‘lmasa), JS divergensiyasi doimiy qiymat (log 2) beradi va gradient nolga teng bo‘ladi. Generator o‘rganmaydi ("Vanishing Gradient").

**WGAN G‘oyasi:**
WGAN mualliflari (Arjovsky et al., 2017) Loss funksiyasini o‘zgartirishni taklif qilishdi. Ular "Ehtimollik" (soxtami/haqiqiymi?) o‘rniga **"Masofa" (qanchalik uzoq?)** tushunchasini kiritishdi.
Diskriminator endi "Kritik" (Critic) deb ataladi va u 0 yoki 1 degan javob bermaydi, balki shunchaki bitta son (score) chiqaradi. Bu son haqiqiy rasm uchun katta, soxta rasm uchun kichik bo‘lishi kerak.
Bu yondashuv o‘qitishni juda barqaror qiladi va "Mode Collapse"ni deyarli yo‘qotadi.

---

### 112. Wasserstein masofasi (Wasserstein distance) nima ekanini izohlab bering.

**Javob:**
Bu masofa **"Earth Mover's Distance" (Tuproq tashuvchi masofasi)** deb ham ataladi.

**Analogi:**
Tasavvur qiling, bizda ikkita tuproq uyumi bor:
1.  Generator yasagan "soxta" taqsimot (shakli buzuq uyum).
2.  Haqiqiy "ideal" taqsimot (chiroyli uyum).

Wasserstein masofasi — bu birinchi uyumdagi tuproqni ikkinchi uyum shakliga keltirish uchun bajarilishi kerak bo‘lgan **minimal ish** miqdoridir (Og‘irlik $\times$ Masofa).
Oddiy masofalardan farqli o‘laroq, hatto ikki uyum bir-biridan juda uzoqda bo‘lsa ham, Wasserstein masofasi aniq qiymatga ega va bu qiymat Generatorga qaysi tomonga harakatlanish kerakligini (gradientni) aniq ko‘rsatib beradi.

---

### 113. Gradient penalty (gradientni jazolash) g‘oyasini va maqsadini tushuntirib bering.

**Javob:**
**Muammo:**
WGAN ishlashi uchun Kritik (Discriminator) funksiyasi **"1-Lipschitz"** shartiga bo‘ysunishi kerak. Bu degani, funksiyaning o‘zgarish tezligi (gradienti) hech qayerda 1 dan oshib ketmasligi kerak. Boshida buni ta’minlash uchun vaznlar shunchaki qirqib tashlanardi (Weight Clipping), lekin bu model sifatini buzardi.

**Yechim (WGAN-GP):**
Gradient Penalty — bu "yumshoqroq" cheklov.
Biz Loss funksiyasiga maxsus jarima (penalty) qo‘shamiz:
$Penalty = \lambda \cdot (||\nabla D(x)|| - 1)^2$.
Ma’nosi: Agar Kritikning gradienti 1 dan farq qilsa (oshib ketsa yoki tushib ketsa), model qattiq jazolanadi.
Bu usul Kritikni barqaror ushlab turadi va yuqori sifatli rasmlar generatsiyasiga olib keladi.

---

### 114. StyleGAN oddiy GAN ga nisbatan nimalarni yaxshilaganini izohlab bering.

**Javob:**
StyleGAN (NVIDIA) — bu hozirgi kunda eng realistik inson yuzlarini yarata oladigan modeldir.

**Yaxshilanishlar:**
Oddiy GANda Generator kirish shovqinini ($z$) birdaniga rasmga aylantirishga harakat qiladi va jarayon "qora quti" bo‘lib qoladi.
StyleGANda esa jarayon bosqichlarga bo‘lingan va boshqariluvchan:
1.  **Mapping Network:** Kirish shovqini avval oraliq vektorga ($w$) aylantiriladi.
2.  **Style Injection:** Bu vektor har bir konvolyutsiya qatlamiga "stil" (AdaIN) sifatida kiritiladi.
    *   Boshlang‘ich qatlamlar: Yuz shakli, pozasi (Coarse styles).
    *   O‘rta qatlamlar: Ko‘z shakli, soch turmagi (Middle styles).
    *   Oxirgi qatlamlar: Teri rangi, soch tolasining ingichka detallari (Fine styles).
Bu bizga, masalan, odamning pozasini o‘zgartirmasdan, faqat soch rangini o‘zgartirish imkonini beradi.

---

### 115. Word embedding (so‘zlarni zich vektor ko‘rinishida ifodalash) g‘oyasini tushuntirib bering.

**Javob:**
**Eski usul (One-Hot Encoding):**
"Olma" = [0, 0, 1, 0...], "Nok" = [0, 0, 0, 1...]. Vektorlar juda uzun (lug‘at hajmi qadar) va siyrak (faqat bitta 1 bor). Eng yomoni, ular orasida hech qanday ma’no bog‘liqligi yo‘q ("Olma" va "Nok" matematik jihatdan butunlay begona).

**Embedding G‘oyasi (Word2Vec, GloVe):**
So‘zlarni kichikroq o‘lchamli (masalan, 300) zich vektorlarga aylantirish. Bu vektorlar katta matnlarni o‘qish jarayonida o‘rganiladi.
**Natija:** Ma’nosi yaqin so‘zlar fazoda yonma-yon joylashadi.
*   $Vektor("Qirol") - Vektor("Erkak") + Vektor("Ayol") \approx Vektor("Qirolicha")$.
Model so‘zlarning ma’nosini raqamlar orqali "tushuna" boshlaydi.

---

### 116. NLP da tokenizer (tokenizator) nima va uning asosiy vazifasini tushuntirib bering.

**Javob:**
Kompyuter matnni (harflarni) tushunmaydi, u faqat sonlarni tushunadi.
**Tokenizer** — bu matnni kompyuter tushunadigan mayda bo‘laklarga (tokenlarga) ajratib, ularni sonlarga (ID) almashtirib beruvchi vositadir.

**Vazifalari:**
1.  **Bo‘laklash:** Matnni so‘zlarga ("kitob"), so‘z bo‘laklariga ("ki-tob") yoki harflarga ajratish.
2.  **Lug‘at tuzish:** Barcha mumkin bo‘lgan tokenlar ro‘yxatini shakllantirish.
3.  **Kodlash (Encoding):** "Men maktabga bordim" $\to$ [104, 2055, 301].
Tokenizator modelning "og‘zi" hisoblanadi — ma’lumot modelga faqat shu orqali kiradi.

---

### 117. Byte-Pair Encoding (BPE) segmentatsiyasi nima uchun qo‘llanilishini izohlab bering.

**Javob:**
Bu zamonaviy modellar (GPT, BERT) ishlatadigan eng ommabop tokenizatsiya usulidir. U "So‘z" va "Harf" tokenizatsiyasining o‘rtasidagi oltin o‘rtalikdir.

**Muammo:**
*   So‘z bo‘yicha: Lug‘at juda katta bo‘lib ketadi (o‘zbek tilida qo‘shimchalar hisobiga millionlab so‘z shakli bor).
*   Harf bo‘yicha: Matn juda uzun bo‘lib ketadi va ma’no yo‘qoladi.

**BPE Yechimi:**
1.  Boshida hammani harflarga ajratadi.
2.  Yonma-yon eng ko‘p uchraydigan harflarni birlashtiradi (masalan "l" va "a" $\to$ "la").
3.  Jarayon takrorlanadi ("la" va "r" $\to$ "lar").
**Natija:** Ko‘p uchraydigan so‘zlar bitta butun token sifatida qoladi ("O‘zbekiston"), kam uchraydigan so‘zlar esa bo‘laklarga bo‘linadi ("avto-trans-port"). Bu lug‘atni ixcham saqlaydi va notanish so‘zlarni ham o‘qish imkonini beradi.

---

### 118. NLP da OOV (out-of-vocabulary) muammosi nimadan iborat ekanini tushuntiring.

**Javob:**
**Muammo:**
Model o‘qitilayotganda ko‘rmagan (lug‘atida yo‘q) so‘zga duch kelsa nima qiladi?
Eski modellarda bu so‘z shunchaki **`<UNK>` (Unknown)** tokeniga almashtirilardi.
Masalan: "Men *kvant* fizikasini o‘qidim" $\to$ "Men `<UNK>` fizikasini o‘qidim".
Ma’lumot yo‘qoladi va model gapning ma’nosini tushunmay qoladi.

**BPE yechimi:**
BPE (Subword tokenization) bu muammoni deyarli yo‘q qildi. Agar "kvant" so‘zi lug‘atda yo‘q bo‘lsa, u "k-van-t" yoki "kv-ant" kabi o‘zi taniydigan mayda bo‘laklarga bo‘linadi. Model baribir bu bo‘laklardan ma’noni yig‘ib oladi.

---

### 119. Masked language modeling (maskalangan til modellashtirish) g‘oyasini tushuntirib bering.

**Javob:**
Bu **BERT** modelini o‘qitishda ishlatiladigan asosiy usuldir.

**G‘oya:**
Biz modelga matnni beramiz, lekin undagi ba’zi so‘zlarni (masalan, 15% ini) yashirib (maskalab) qo‘yamiz.
*   Kirish: "Amir Temur [MASK] yilda tug‘ilgan."
*   Vazifa: "[MASK]" o‘rnida qaysi so‘z bo‘lishi kerakligini topish ("1336").

**Maqsad:**
Model bu vazifani bajarish uchun nafaqat chapdagi ("Amir Temur"), balki o‘ngdagi ("yilda tug‘ilgan") so‘zlarga ham qarashga majbur bo‘ladi. Bu modelda **chuqur, ikki tomonlama kontekstual tushunchani** shakllantiradi. Oddiy modellar (GPT) faqat chapdan o‘ngga o‘qigani uchun bunday kuchli kontekstni ko‘ra olmaydi.

---

### 120. Mustahkamlangan o‘qitish (Reinforcement Learning) ning nazoratli o‘qitish (supervised learning) dan farqlarini tushuntirib bering.

**Javob:**
**Supervised Learning (Nazoratli):**
*   **O‘qituvchi bor:** Modelga har bir savol uchun "To‘g‘ri javob" (Label) tayyorlab berilgan.
*   Misol: "Bu rasm - mushuk", "Bu rasm - it".
*   Maqsad: Javobni nusxalash.

**Reinforcement Learning (Mustahkamlangan):**
*   **O‘qituvchi yo‘q:** To‘g‘ri javob oldindan berilmaydi.
*   **Tajriba:** Agent (model) muhitda harakat qiladi va xatolardan o‘rganadi.
*   **Signal:** To‘g‘ri javob o‘rniga **Mukofot (Reward)** yoki **Jazo (Penalty)** beriladi.
*   Misol: Shaxmat o‘ynayotgan kompyuterga hech kim har bir yurishda nima qilishni aytmaydi. Faqat o‘yin oxirida "Yutding (+1)" yoki "Yutqazding (-1)" deyiladi. Agent o‘zi qaysi yurishlar g‘alabaga olib kelganini tahlil qilib topishi kerak.
