# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (91-100)

### 91. Transformer arxitekturasida kodlovchi (encoder) va dekoder (decoder) ning vazifalarini tushuntirib bering.

**Javob:**
Asl Transformer (Vaswani et al., 2017) mashina tarjimasi uchun yaratilgan bo‘lib, u Encoder-Decoder tuzilishiga ega.

1.  **Encoder (Kodlovchi):**
    *   **Vazifasi:** Kirish matnini (masalan, o‘zbekcha gapni) qabul qilish va uni "tushunish".
    *   **Ishlash:** U so‘zlarning o‘zaro bog‘liqligini (Self-Attention) tahlil qiladi, har bir so‘z uchun kontekstga boyitilgan vektorni hosil qiladi. Encoderning chiqishi — bu kirish ma’lumotining to‘liq va mavhum matematik tasviridir. U hech qanday so‘z ishlab chiqarmaydi, faqat tahlil qiladi.
2.  **Decoder (Dekodlovchi):**
    *   **Vazifasi:** Encoder bergan tahlil va o‘zi oldin yozgan so‘zlarga asoslanib, yangi matn (masalan, inglizcha tarjima) yaratish.
    *   **Ishlash:** U ikkita Attentionga ega: biri o‘zining yozgan so‘zlariga qaraydi (Self-Attention), ikkinchisi Encoderning chiqishiga qaraydi (Cross-Attention). Bu unga tarjima jarayonida asl matnning kerakli joylariga "mo‘ralab" olish imkonini beradi.

---

### 92. Asl Transformer arxitekturasining umumiy tuzilishini qisqacha tushuntirib bering.

**Javob:**
Transformer arxitekturasi to‘liq **Attention** mexanizmiga asoslangan (RNN yoki CNN yo‘q).

*   **Encoder bloki (6 qavat):** Har bir qavat 2 qismdan iborat:
    1.  Multi-Head Self-Attention (So‘zlar orasidagi bog‘liqlikni topish).
    2.  Feed-Forward Neural Network (Ma’lumotni qayta ishlash).
    *   Har bir qismdan keyin "Residual Connection" (qo‘shish) va "Layer Normalization" bor.
*   **Decoder bloki (6 qavat):** Har bir qavat 3 qismdan iborat:
    1.  Masked Multi-Head Self-Attention (Kelajakdagi so‘zlarni ko‘rmasdan, faqat o‘tmishga qarash).
    2.  Multi-Head Attention (Encoderning chiqishiga bog‘lanish).
    3.  Feed-Forward Neural Network.
    *   Yana Residual Connection va Layer Norm.

Oxirida **Linear** va **Softmax** qatlami bo‘lib, u navbatdagi so‘zning ehtimolligini chiqarib beradi.

---

### 93. GPT va BERT modellarining asosiy farqlarini izohlab bering.

**Javob:**
Ikkalasi ham Transformerga asoslangan, lekin arxitektura va maqsadlari tubdan farq qiladi.

1.  **BERT (Bidirectional Encoder Representations from Transformers):**
    *   **Qismi:** Faqat **Encoder** (Kodlovchi) dan iborat.
    *   **Yo‘nalishi:** Ikki tomonlama (**Bidirectional**). U gapni boshidan oxirigacha va oxiridan boshigacha bir vaqtda ko‘radi. "Bank" so‘zini tushunish uchun uning o‘ng va chap tarafidagi so‘zlarni hisobga oladi.
    *   **Maqsadi:** Matnni **tushunish** (tasniflash, savol-javob, nomlarni aniqlash). U gapira olmaydi (generatsiya qilmaydi), lekin o‘qiganini juda yaxshi tushunadi.
2.  **GPT (Generative Pre-trained Transformer):**
    *   **Qismi:** Faqat **Decoder** (Dekodlovchi) dan iborat.
    *   **Yo‘nalishi:** Bir tomonlama (**Unidirectional**). U faqat chapdan o‘ngga qarab o‘qiydi. Keyingi so‘zni bashorat qilishda kelajakni ko‘ra olmaydi.
    *   **Maqsadi:** Matn **yaratish** (generatsiya). Hikoya yozish, kod yozish, suhbatlashish.

---

### 94. Faqat encoder, faqat decoder va encoder–decoder arxitekturalarini taqqoslab bering.

**Javob:**
1.  **Encoder-only (BERT, RoBERTa):**
    *   **Kuchli tomoni:** Matn kontekstini mukammal tushunadi (hamma tomondan).
    *   **Vazifasi:** Sentiment tahlil, spamni aniqlash, matndan ma’lumot qidirish.
    *   **Kamchiligi:** Matn yozish (generatsiya) uchun noqulay va sekin.
2.  **Decoder-only (GPT-3, GPT-4, Llama):**
    *   **Kuchli tomoni:** Tabiiy va ravon matn yozish. "Keyingi so‘z nima?" degan savolga javob berishda tengsiz.
    *   **Vazifasi:** Chatbotlar, ijodiy yozish, kod to‘ldirish.
    *   **Kamchiligi:** Matnni tushunishda (masalan, savol-javobda) BERTdan biroz kuchsizroq bo‘lishi mumkin (chunki kelajakdagi kontekstni ko‘rmaydi), lekin masshtab hisobiga (juda katta modellar) bu kamchilik deyarli yo‘qolgan.
3.  **Encoder-Decoder (T5, BART, Original Transformer):**
    *   **Kuchli tomoni:** Matnni o‘qib, uni boshqa matnga aylantirish.
    *   **Vazifasi:** Tarjima (Translation), Matnni qisqartirish (Summarization).
    *   **Xulosa:** Tushunish va Yozish qobiliyatlarini birlashtiradi.

---

### 95. Katta til modellarida “kontekst oynasi” (context window) cheklovi muammosini tushuntirib bering.

**Javob:**
**Muammo:**
Transformer modellarida "xotira" cheklangan. Ular bir vaqtning o‘zida ma’lum miqdordagi (masalan, 4096 ta) tokenni (so‘z bo‘lagini) ko‘ra oladi, xolos.
Agar siz modelga 100 sahifalik kitobni bersangiz, u faqat oxirgi 5-10 sahifani "eslaydi", boshi esa oynadan chiqib ketadi va unutiladi.

**Sababi:**
Attention mexanizmining hisoblash murakkabligi $O(N^2)$ ga teng ($N$ - tokenlar soni).
    *   Agar matn uzunligi 2 barobar oshsa, hisoblash va xotira sarfi 4 barobar oshadi.
    *   Agar matn 10 barobar oshsa, sarf 100 barobar oshadi.
Shuning uchun kontekstni cheksiz uzaytirib bo‘lmaydi — kompyuter xotirasi (VRAM) yetmay qoladi. Hozirda maxsus usullar (FlashAttention, RoPE) orqali bu oyna 128k yoki 1M tokengacha kengaytirildi, lekin baribir cheklov mavjud.

---

### 96. Autoencoder (avtoenkoder) modelining asosiy g‘oyasini tushuntirib bering.

**Javob:**
Autoencoder — bu "Nazoratsiz o‘qitish" (Unsupervised Learning) turiga kiruvchi neyron tarmoq.
**G‘oya:**
Modelning maqsadi kirishga berilgan ma’lumotning (masalan, rasmning) aynan nusxasini chiqishda hosil qilishdir. $Output ≈ Input$.
Lekin bunda bitta hiyla bor: Tarmoqning o‘rtasi ("beli") ataylab juda tor qilib qo‘yiladi.

**Maqsad:**
Model rasmni shunchaki "ko‘chirib" qo‘ya olmaydi (o‘rtadan sig‘maydi). U rasmni siqishga, eng muhim ma’lumotlarni ajratib olib, keraksiz detallarni tashlab yuborishga majbur bo‘ladi.
Bu xuddi zip-arxivatorga o‘xshaydi, lekin u ma’lumotlarning "mazmunini" siqadi.

---

### 97. Encoder va decoder tushunchalarini izohlab bering (Autoencoder kontekstida).

**Javob:**
Autoencoder ikki qismdan iborat simmetrik tarmoqdir:

1.  **Encoder (Siquvchi):**
    *   Kirish ma’lumotini ($x$) oladi va uni qatlamdan-qatlamga kichraytirib, "Latent kod" ($z$) ga aylantiradi.
    *   Vazifasi: Ma’lumotni siqish va xususiyatlarni ajratib olish.
2.  **Decoder (Tiklovchi):**
    *   Siqilgan kodni ($z$) oladi va uni qatlamdan-qatlamga kengaytirib, asl ma’lumotni ($x'$) qayta tiklashga harakat qiladi.
    *   Vazifasi: Siqilgan koddan to‘liq rasmni rekonstruksiya qilish.

O‘qitish paytida biz $x$ va $x'$ orasidagi farqni (Loss) kamaytiramiz.

---

### 98. Bottleneck (tor bo‘g‘in, siqilgan yashirin qatlam) nima va nima uchun kerakligini tushuntirib bering.

**Javob:**
**Tushuncha:**
Bottleneck — bu Autoencoderning eng o‘rtasidagi, eng kam neyronga ega qatlami (Latent Space). Masalan, kirishda 784 ta neyron (28x28 rasm) bo‘lsa, Bottleneckda faqat 32 ta neyron bo‘lishi mumkin.

**Kerakligi:**
Agar Bottleneck bo‘lmasa (o‘rtasi keng bo‘lsa), model shunchaki "kirishni chiqishga uzatishni" (Identity function: $y=x$) o‘rganib oladi. Bunda hech qanday foydali narsa o‘rganilmaydi.
Tor bo‘g‘in modelni **tanlashga majbur qiladi**: "Sen hamma narsani saqlay olmaysan. Faqat eng muhim narsalarni (masalan, rasmda '8' raqami borligini va u tik turganini) saqlab qol, fon rangi yoki mayda shovqinlarni tashlab yubor".
Bu ma’lumotlarning eng samarali (kompakt) ifodasini topishga olib keladi.

---

### 99. O‘lchamni kamaytirish (dimensionality reduction) vazifasida autoencoder dan qanday foydalanish mumkinligini izohlab bering.

**Javob:**
Odatda o‘lchamni kamaytirish uchun PCA (Principal Component Analysis) ishlatiladi, lekin PCA faqat chiziqli bog‘liqliklarni ko‘radi.
Autoencoder esa — bu **chiziqli bo‘lmagan PCA** dir.

**Foydalanish:**
1.  Autoencoderni to‘liq o‘qitamiz (kirishni tiklashga).
2.  Keyin Decoder qismini kesib tashlaymiz.
3.  Faqat Encoder qismini ishlatamiz: u bizga yuqori o‘lchamli ma’lumotni (masalan, HD rasm) kichik o‘lchamli vektorga (bottleneck code) aylantirib beradi.

Bu kichik vektorlar ma’lumotlarni vizualizatsiya qilish (2D/3D ga tushirish), saqlashni tejash yoki boshqa modellar uchun kirish ma’lumoti sifatida (preprocessing) ishlatilishi mumkin.

---

### 100. Denoising autoencoder qanday ishlashini va uning maqsadini tushuntirib bering.

**Javob:**
**Muammo:**
Oddiy Autoencoder ba’zan baribir "yodlab olishga" (Identity) o‘tib ketishi mumkin.

**Ishlash prinsipi:**
Denoising Autoencoder (DAE) o‘qitish jarayonini qiyinlashtiradi:
1.  Kirish rasmiga ataylab **shovqin qo‘shiladi** (masalan, ba’zi piksellar o‘chiriladi yoki dog‘ tushiriladi) $	o 	ilde{x}$.
2.  Modelga shu buzilgan rasm beriladi.
3.  Lekin talab qilinadi: "Chiqishda menga buzilgan rasmni emas, **asl toza rasmni ($x$)** qaytarib ber!"

**Maqsadi:**
Model shovqinni "shovqin" ekanini tushunishi va uni tozalashni o‘rganishi kerak. Bu modelni shunchaki piksellarni nusxalashdan emas, balki rasmdagi ob’ektlarning **tuzilishi va qonuniyatlarini** chuqur tushunishga majbur qiladi (masalan, "yarimta chiziq bo‘lmaydi, uni to‘ldirish kerak"). Natijada model juda mustahkam (robust) xususiyatlarni o‘rganadi.