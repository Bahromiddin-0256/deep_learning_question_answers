# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (111-120)

### 111. WGAN (Wasserstein GAN) g‘oyasini tushuntirib bering.

**Javob:**
**Muammo:**
Oddiy GANlarda Loss funksiyasi sifatida "Jensen-Shannon" (JS) divergensiyasi ishlatiladi. Muammo shundaki, agar haqiqiy va soxta rasmlar taqsimoti bir-biriga umuman tegmasa (overlap bo‘lmasa), JS divergensiyasi doimiy qiymat (log 2) beradi va gradient nolga teng bo‘ladi. Generator o‘rganmaydi ("Vanishing Gradient").

**WGAN G‘oyasi:**
WGAN mualliflari (Arjovsky et al., 2017) Loss funksiyasini tubdan o‘zgartirishni taklif qilishdi. Ular "Ehtimollik" (soxtami/haqiqiymi?) o‘rniga **"Masofa" (qanchalik uzoq?)** tushunchasini kiritishdi.
*   Diskriminator endi "Kritik" (Critic) deb ataladi.
*   U [0, 1] ehtimollik qaytarmaydi, balki chegaralanmagan bitta son (score) chiqaradi.
*   Bu son haqiqiy rasm uchun iloji boricha katta, soxta rasm uchun kichik bo‘lishi kerak.
Bu yondashuv o‘qitishni juda barqaror qiladi, chunki har doim mazmunli gradient mavjud bo‘ladi va "Mode Collapse"ni deyarli yo‘qotadi.

---

### 112. Wasserstein masofasi (Wasserstein distance) nima ekanini izohlab bering.

**Javob:**
Bu masofa **"Earth Mover's Distance" (Tuproq tashuvchi masofasi)** deb ham ataladi.

**Analogi:**
Tasavvur qiling, bizda ikkita tuproq uyumi (taqsimot) bor:
1.  Generator yasagan "soxta" taqsimot (shakli buzuq uyum).
2.  Haqiqiy "ideal" taqsimot (chiroyli uyum).

Wasserstein masofasi — bu birinchi uyumdagi tuproqni ikkinchi uyum shakliga keltirish uchun bajarilishi kerak bo‘lgan **minimal ish** miqdoridir (Og‘irlik $\times$ Masofa).
**Afzalligi:**
Oddiy masofalardan (KL, JS) farqli o‘laroq, hatto ikki uyum bir-biridan juda uzoqda bo‘lsa ham, Wasserstein masofasi aniq qiymatga ega va bu qiymat (uning gradienti) Generatorga tuproqni qaysi tomonga surish kerakligini aniq ko‘rsatib beradi.

---

### 113. Gradient penalty (gradientni jazolash) g‘oyasini va maqsadini tushuntirib bering.

**Javob:**
**Muammo:**
WGAN nazariy jihatdan ishlashi uchun Kritik (Discriminator) funksiyasi **"1-Lipschitz"** shartiga bo‘ysunishi kerak. Bu degani, funksiyaning o‘zgarish tezligi (gradienti) hech qayerda 1 dan oshib ketmasligi kerak. Boshida buni ta’minlash uchun vaznlar shunchaki qirqib tashlanardi (Weight Clipping: [-0.01, 0.01]), lekin bu model sifatini buzardi.

**Yechim (WGAN-GP):**
Gradient Penalty — bu "yumshoqroq" va samaraliroq cheklov.
Biz Loss funksiyasiga maxsus jarima (penalty) qo‘shamiz:
$$ Penalty = \lambda \cdot (||\nabla D(\hat{x})|| - 1)^2 $$
**Maqsadi:**
Agar Kritikning gradienti normasi 1 dan farq qilsa (juda katta yoki juda kichik bo‘lsa), model qattiq jazolanadi. Bu usul Kritikni barqaror ushlab turadi, vaznlarni sun’iy cheklamaydi va yuqori sifatli rasmlar generatsiyasiga olib keladi.

---

### 114. StyleGAN oddiy GAN ga nisbatan nimalarni yaxshilaganini izohlab bering.

**Javob:**
StyleGAN (NVIDIA) — bu hozirgi kunda eng realistik inson yuzlarini yarata oladigan modeldir. U GAN arxitekturasini tubdan o‘zgartirdi.

**Yaxshilanishlar:**
Oddiy GANda Generator kirish shovqinini ($z$) birdaniga rasmga aylantirishga harakat qiladi va jarayon "qora quti" bo‘lib qoladi.
StyleGANda esa jarayon bosqichlarga bo‘lingan va **boshqariluvchan (controllable)**:
1.  **Mapping Network:** Kirish shovqini ($z$) avval oraliq vektorga ($w$) aylantiriladi.
2.  **Style Injection:** Bu $w$ vektor har bir konvolyutsiya qatlamiga "stil" (AdaIN - Adaptive Instance Normalization) sifatida kiritiladi.
    *   **Boshlang‘ich qatlamlar:** Yuz shakli, pozasi (Coarse styles).
    *   **O‘rta qatlamlar:** Ko‘z shakli, soch turmagi (Middle styles).
    *   **Oxirgi qatlamlar:** Teri rangi, soch tolasining ingichka detallari (Fine styles).
Bu bizga "Stil aralashtirish" (Style Mixing) imkonini beradi: masalan, odamning pozasini o‘zgartirmasdan, faqat soch rangini o‘zgartirish mumkin.

---

### 115. Word embedding (so‘zlarni zich vektor ko‘rinishida ifodalash) g‘oyasini tushuntirib bering.

**Javob:**
**Eski usul (One-Hot Encoding):**
Har bir so‘z bitta ulkan vektor bilan ifodalanadi: "Olma" = [0, 0, 1, 0...], "Nok" = [0, 0, 0, 1...].
Kamchiliklari: Vektorlar juda uzun (lug‘at hajmi qadar) va siyrak. Eng yomoni, ular orasida hech qanday **semantik bog‘liqlik** yo‘q ("Olma" va "Nok" matematik jihatdan butunlay begona, ortogonal vektorlar).

**Embedding G‘oyasi (Word2Vec, GloVe):**
So‘zlarni kichikroq o‘lchamli (masalan, 300) **zich vektorlarga** aylantirish. Bu vektorlar katta matnlarni o‘qish jarayonida o‘rganiladi.
**Natija:**
Ma’nosi yaqin so‘zlar fazoda yonma-yon joylashadi.
*   Biz so‘zlar ustida arifmetik amallar bajara olamiz:
    $Vektor("Qirol") - Vektor("Erkak") + Vektor("Ayol") \approx Vektor("Qirolicha")$.
Model so‘zlarning ma’nosini raqamlar orqali "tushuna" boshlaydi.

---

### 116. NLP da tokenizer (tokenizator) nima va uning asosiy vazifasini tushuntirib bering.

**Javob:**
Neyron tarmoqlar matnni (harflarni) tushunmaydi, ular faqat sonlarni tushunadi.
**Tokenizer** — bu inson tilidagi matnni kompyuter tushunadigan mayda bo‘laklarga (tokenlarga) ajratib, ularni sonlarga (ID) almashtirib beruvchi ko‘prikdir.

**Asosiy vazifalari:**
1.  **Bo‘laklash (Segmentation):** Matnni qoidalarga ko‘ra bo‘lish.
    *   So‘z bo‘yicha: "Men maktabga bordim" $\to$ ["Men", "maktabga", "bordim"].
    *   Subword (so‘z bo‘lagi) bo‘yicha: $\to$ ["Men", "maktab", "##ga", "bor", "##dim"].
2.  **Lug‘at tuzish (Vocabulary Building):** Barcha mumkin bo‘lgan tokenlar ro‘yxatini shakllantirish va ularga unikal raqam (ID) berish.
3.  **Kodlash (Encoding):** Matnni raqamli vektorga aylantirish: [104, 2055, 301].
Zamonaviy modellar (BERT, GPT) **WordPiece** yoki **BPE** tokenizatorlaridan foydalanadi.

---

### 117. Byte-Pair Encoding (BPE) segmentatsiyasi nima uchun qo‘llanilishini izohlab bering.

**Javob:**
BPE (Byte-Pair Encoding) — bu zamonaviy NLP modellarida (GPT, RoBERTa) ishlatiladigan eng samarali tokenizatsiya usulidir. U "So‘z" va "Harf" tokenizatsiyasining o‘rtasidagi oltin o‘rtalikdir.

**Muammo:**
*   **So‘z bo‘yicha:** Lug‘at juda katta bo‘lib ketadi (o‘zbek tilida qo‘shimchalar hisobiga millionlab so‘z shakli bor).
*   **Harf bo‘yicha:** Matn juda uzun bo‘lib ketadi va ma’no yo‘qoladi.

**BPE Yechimi:**
1.  Boshida hammani harflarga ajratadi.
2.  Yonma-yon eng ko‘p uchraydigan harflarni birlashtiradi (masalan "l" va "a" $\to$ "la").
3.  Jarayon takrorlanadi ("la" va "r" $\to$ "lar").
**Natija:**
*   Ko‘p uchraydigan so‘zlar bitta butun token sifatida qoladi ("O‘zbekiston").
*   Kam uchraydigan yoki murakkab so‘zlar esa bo‘laklarga bo‘linadi ("avto-trans-port").
Bu lug‘atni ixcham saqlaydi va notanish so‘zlarni ham o‘qish imkonini beradi.

---

### 118. NLP da OOV (out-of-vocabulary) muammosi nimadan iborat ekanini tushuntiring.

**Javob:**
**Muammo:**
Model o‘qitilayotganda ko‘rmagan va lug‘atida yo‘q bo‘lgan so‘zga duch kelsa nima qiladi?
Eski modellarda (Word2Vec) bu so‘z shunchaki **`<UNK>` (Unknown)** tokeniga almashtirilardi.
*   Kirish: "Men *kvant* fizikasini o‘qidim".
*   Agar "kvant" so‘zi lug‘atda bo‘lmasa: "Men `<UNK>` fizikasini o‘qidim".
Ma’lumot yo‘qoladi va model gapning ma’nosini tushunmay qoladi. Bu ayniqsa morfologik boy tillarda (o‘zbek tili) katta muammo.

**Yechim (Subword Tokenization):**
BPE yoki WordPiece bu muammoni deyarli yo‘q qildi. Agar "kvant" so‘zi lug‘atda yo‘q bo‘lsa, u "k-van-t" yoki "kv-ant" kabi o‘zi taniydigan mayda bo‘laklarga bo‘linadi. Model baribir bu bo‘laklardan ma’noni yig‘ib oladi va `<UNK>` ishlatilmaydi.

---

### 119. Masked language modeling (maskalangan til modellashtirish) g‘oyasini tushuntirib bering.

**Javob:**
Bu **BERT** modelini o‘qitishda ishlatiladigan asosiy inqilobiy usuldir (Cloze Task).

**G‘oya:**
Biz modelga matnni beramiz, lekin undagi ba’zi so‘zlarni (masalan, 15% ini) ataylab yashirib (maskalab) qo‘yamiz.
*   Kirish: "Amir Temur [MASK] yilda tug‘ilgan."
*   Vazifa: "[MASK]" o‘rnida qaysi so‘z bo‘lishi kerakligini topish ("1336").

**Farqi va Ustunligi:**
Oddiy til modellari (GPT) faqat chapdan o‘ngga qarab o‘qiydi (keyingi so‘zni topish uchun). BERT esa [MASK] ni topish uchun nafaqat chapdagi ("Amir Temur"), balki o‘ngdagi ("yilda tug‘ilgan") so‘zlarga ham qarashga majbur bo‘ladi.
Bu modelda **chuqur, ikki tomonlama (bidirectional) kontekstual tushunchani** shakllantiradi. Model so‘zning ma’nosini uning atrofidagi butun muhitdan kelib chiqib tushunadi.

---

### 120. Mustahkamlangan o‘qitish (Reinforcement Learning) ning nazoratli o‘qitish (supervised learning) dan farqlarini tushuntirib bering.

**Javob:**
**Supervised Learning (Nazoratli):**
*   **O‘qituvchi bor:** Modelga har bir savol uchun "To‘g‘ri javob" (Label) tayyorlab berilgan.
*   Misol: "Bu rasm - mushuk", "Bu rasm - it".
*   **Maqsad:** Javobni iloji boricha aniq nusxalash va xatoni minimallashtirish.

**Reinforcement Learning (Mustahkamlangan):**
*   **O‘qituvchi yo‘q:** To‘g‘ri javob oldindan berilmaydi.
*   **Tajriba:** Agent (model) noma’lum muhitda harakat qiladi va o‘z xatolaridan o‘rganadi.
*   **Signal:** To‘g‘ri javob o‘rniga **Mukofot (Reward)** yoki **Jazo (Penalty)** beriladi.
*   **Misol:** Shaxmat o‘ynayotgan kompyuterga hech kim har bir yurishda nima qilishni aytmaydi. Faqat o‘yin oxirida "Yutding (+1)" yoki "Yutqazding (-1)" deyiladi. Agent o‘zi qaysi yurishlar g‘alabaga olib kelganini tahlil qilib topishi kerak (Delayed Reward).
