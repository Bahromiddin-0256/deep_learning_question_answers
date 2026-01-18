# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (101-110)

### 101. Variational autoencoder (VAE) nima ekanini tushuntirib bering.

**Javob:**
Variational Autoencoder (VAE) — bu oddiy Autoencoderning "Generativ" (yaratuvchi) variantidir.
Oddiy Autoencoder ma’lumotni shunchaki siqishni bilsa, VAE **yangi ma’lumotlarni yaratishni** (masalan, yo‘q odamlarning yuzini chizishni) biladi.

U ehtimollar nazariyasiga (Probability Theory) asoslangan. VAE kirish ma’lumotini bitta qat’iy nuqtaga (vektorga) emas, balki **ehtimollik taqsimotiga** (Normal distribution) aylantiradi. Ya’ni, u rasmni "kodlamaydi", balki rasmning "retseptini" (o‘rtacha qiymati va tarqoqligini) topadi.

---

### 102. VAE va oddiy autoencoder o‘rtasidagi asosiy farqlarni izohlab bering.

**Javob:**
Asosiy farq **Latent Space (Yashirin fazo)** ning tuzilishida.

1.  **Oddiy Autoencoder:**
    *   Yashirin fazo **uzlukli (discrete)**.
    *   Masalan, bitta nuqta "kulayotgan odam", ikkinchi nuqta "yig‘layotgan odam" bo‘lishi mumkin. Lekin bu ikki nuqta orasidagi bo‘shliqda nima borligi noma’lum. Agar biz o‘rtadagi nuqtani olib Decoderga bersak, u ma’nosiz shovqin (abrakadabra) chiqaradi.
    *   Generatsiya uchun yaroqsiz.

2.  **Variational Autoencoder (VAE):**
    *   Yashirin fazo **uzluksiz (continuous)** va **silliq**.
    *   U "kulayotgan" va "yig‘layotgan" nuqtalar orasidagi barcha nuqtalarni ham to‘ldirib chiqadi (interpolatsiya).
    *   Agar biz oraliq nuqtani olsak, Decoder bizga "yarim kulayotgan, yarim yig‘layotgan" yuzni silliq o‘zgarish bilan chiqarib beradi.
    *   Yangi narsa yaratish uchun juda qulay.

---

### 103. Latent space (yashirin fazo) tushunchasini izohlab bering.

**Javob:**
**Tushuncha:**
Latent space — bu ma’lumotlarning siqilgan, mavhum xususiyatlari saqlanadigan ko‘p o‘lchamli fazo.
Tasavvur qiling, bizda millionlab turli odam yuzlari (yuqori o‘lchamli piksellar) bor. Lekin barcha yuzlarni tasvirlash uchun bir nechta asosiy parametr yetarli: jinsi, yoshi, soch rangi, kayfiyati.
Latent space — aynan mana shu parametrlarning koordinata tizimidir.

**Xususiyati:**
*   Bu fazoda o‘xshash narsalar yonma-yon turadi (barcha ayollar bir tomonda, erkaklar boshqa tomonda).
*   Biz bu fazoda "sayr qilib", parametrlarni o‘zgartirish orqali (masalan, "yosh" o‘qini kattalashtirish orqali) natijaviy rasmni boshqarishimiz mumkin.

---

### 104. Generativ modellar tarkibida VAE qaysi maqsadlarda qo‘llanilishini tushuntiring.

**Javob:**
VAE quyidagi vazifalarda keng qo‘llaniladi:
1.  **Yangi namunalar yaratish:** Mavjud ma’lumotlarga o‘xshash, lekin aynan nusxasi bo‘lmagan yangi rasmlar, musiqalar yoki molekular tuzilmalarni generatsiya qilish.
2.  **Interpolatsiya (Morphing):** Bir rasmdan ikkinchi rasmga silliq o‘tish. Masalan, "Qora mashina" rasmidan sekin-asta "Oq mashina" rasmiga aylantirish.
3.  **Anomaliyalarni aniqlash:** VAE normal ma’lumotlarni yaxshi tiklaydi. Agar unga g‘alati (anomal) rasm berilsa, u buni tiklay olmaydi (xatolik katta bo‘ladi). Shu orqali defektlar yoki kasalliklar aniqlanadi.
4.  **Denoising:** Tasvirdagi shovqinni tozalash.

---

### 105. VAE da KL-divergensiya (Kullback–Leibler divergence) nimani anglatishini va nima uchun qo‘llanilishini izohlab bering.

**Javob:**
VAEning Loss funksiyasi ikki qismdan iborat:
1.  **Reconstruction Loss:** Rasmni to‘g‘ri tiklash xatoligi (MSE).
2.  **KL-Divergence:** Bu "Jazo" (Regularization) qismi.

**Ma’nosi:**
KL-divergensiya bizning latent fazoyimizdagi taqsimotning **Normal taqsimotga (Gaussian Distribution, $\mu=0, \sigma=1$)** qanchalik yaqinligini o‘lchaydi.

**Nima uchun kerak?**
Agar KL-loss bo‘lmasa, Encoder nuqtalarni fazoning turli burchaklariga sochib tashlaydi va ular orasida bo‘shliqlar qoladi (Generatsiya buziladi).
KL-divergensiya Encoderni majburlaydi: "Barcha nuqtalarni markazga (0 atrofida) zich joylashtir va ular silliq tarqalsin". Bu latent fazoning uzluksiz va tartibli bo‘lishini ta’minlaydi, shunda biz istalgan joydan nuqta olib, ma’noli rasm hosil qila olamiz.

---

### 106. GAN (Generative Adversarial Network) modelining asosiy g‘oyasini tushuntirib bering.

**Javob:**
GAN (Ian Goodfellow, 2014) — bu chuqur o‘qitishdagi eng qiziqarli g‘oyalardan biri bo‘lib, u **"Raqobat"** prinsipiga asoslanadi.

**G‘oya:**
Tizimda ikkita neyron tarmoq bor va ular bir-biriga dushman:
1.  **Generator (Qalbaki pul yasovchi):** Uning maqsadi shunday mukammal soxta pul (rasm) yasashki, hech kim uni haqiqiydan ajrata olmasin.
2.  **Discriminator (Politsiya/Ekspert):** Uning maqsadi qo‘lidagi pul haqiqiymi yoki soxta ekanini aniqlash.

O‘yin davomida Generator aldashni, Diskriminator esa fosh qilishni o‘rganib boradi. Natijada Generator shunday darajaga yetadiki, u yaratgan "soxta" rasmlar haqiqiysidan farq qilmay qoladi.

---

### 107. GAN tarkibidagi generatorning vazifasi nimadan iborat ekanini tushuntirib bering.

**Javob:**
**Vazifasi:**
Noldan (yo‘q joydan) yangi ma’lumot yaratish.

**Ishlash tartibi:**
1.  Generator kirish sifatida **Tasodifiy shovqin vektorini (Random Noise, $z$)** oladi.
2.  U bu shovqinni neyron qatlamlari (Transposed Convolution) orqali kattalashtirib, rasmga aylantiradi.
3.  Boshida bu rasm shunchaki "rangli bo‘tqa" bo‘ladi.
4.  Lekin Diskriminator unga: "Bu soxta!" deb signal bergach (Gradient), Generator o‘z parametrlarini o‘zgartiradi: "Keyingi safar mana bu joyini o‘zgartirsam, balki ishonarliroq chiqar".
5.  Maqsad: Diskriminatorni adashtirish (uni "Haqiqiy" deb javob berishga majburlash).

---

### 108. GAN tarkibidagi diskriminatorning vazifasini tushuntirib bering.

**Javob:**
**Vazifasi:**
Binar tasniflagich (Binary Classifier) sifatida ishlash: Kirish ma’lumotini **"Real" (1)** yoki **"Fake" (0)** sinfiga ajratish.

**Ishlash tartibi:**
1.  Diskriminatorga aralash ma’lumotlar beriladi: ba’zilari haqiqiy datasetdan, ba’zilari Generatordan.
2.  Agar u haqiqiy rasmni "Real", soxtani "Fake" deb topsa, u mukofotlanadi (vaznlari to‘g‘ri).
3.  Agar u adashsa (soxtani Real deb yuborsa), u jazolanadi va keyingi safar hushyorroq bo‘lishni o‘rganadi.
4.  Diskriminator qanchalik kuchli bo‘lsa, Generator ham shunchalik kuchli bo‘lishga majbur bo‘ladi (chunki oddiy hiylalar o‘tmay qoladi).

---

### 109. Nega GAN larni o‘qitish murakkab hisoblanadi? Asosiy sabablari-ni izohlab bering.

**Javob:**
GANlarni o‘qitish — bu juda nozik muvozanatni talab qiladigan jarayon.

1.  **Mode Collapse (Modalar kollapsi):** Generator dangasalik qilib, Diskriminatorni aldashning bitta oson yo‘lini topib oladi (masalan, faqat bitta turdagi "oq it" rasmini chizish). U faqat shu rasmni qayta-qayta chiqaraveradi, xilma-xillik yo‘qoladi.
2.  **Muvozanatsizlik (Imbalance):** Agar Diskriminator juda tez kuchayib ketsa, u Generatorning har qanday urinishini 100% fosh qiladi. Generator "qayerga yursam ham xato ekan" deb tushkunlikka tushadi (gradient yo‘qoladi) va o‘qishdan to‘xtaydi. Teskarisi ham yomon.
3.  **Konvergensiya yo‘qligi:** Oddiy modellarda Loss kamayib boradi va to‘xtaydi. GANda esa ikki tarmoq bir-birini quvib yuradi, Loss doimiy sakrab turishi mumkin, bu esa "qachon to‘xtash kerak?" degan savolni qiyinlashtiradi.

---

### 110. DCGAN oddiy GAN dan nimasi bilan farq qilishini izohlab bering.

**Javob:**
DCGAN (Deep Convolutional GAN, 2015) — bu GAN arxitekturasini barqarorlashtirish uchun kiritilgan birinchi muvaffaqiyatli standartdir. Oddiy GAN (MLP asosida) faqat kichik va oddiy rasmlarda ishlardi.

**Farqlari va Yangiliklari:**
1.  **To‘liq CNN:** Ikkala tarmoq ham to‘liq Konvolyutsion qatlamlardan quriladi (Fully Connected qatlamlar olib tashlanadi).
2.  **Pooling o‘rniga Stride:** O‘lchamni o‘zgartirish uchun Max Pooling emas, balki **Strided Convolution** (Diskriminatorda) va **Transposed Convolution** (Generatorda) ishlatiladi. Bu modelga o‘lchamni silliq o‘zgartirishni o‘rganishga imkon beradi.
3.  **Batch Normalization:** Har bir qatlamda Batch Norm ishlatiladi. Bu Mode Collapse ni oldini olishga va o‘qitishni barqaror qilishga katta yordam beradi.
4.  **Aktivatsiyalar:** Generatorda **ReLU** (chiqishda Tanh), Diskriminatorda **Leaky ReLU** ishlatilishi qat’iy belgilangan.
