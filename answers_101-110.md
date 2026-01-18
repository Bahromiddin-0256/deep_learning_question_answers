# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (101-110)

### 101. Variational autoencoder (VAE) nima ekanini tushuntirib bering.

**Javob:**
Variational Autoencoder (VAE) — bu oddiy Autoencoderning **Generativ (Yaratuvchi)** variantidir.
Oddiy Autoencoder ma’lumotni shunchaki siqishni (kodlashni) bilsa, VAE **yangi ma’lumotlarni yaratishni** (masalan, yo‘q odamlarning yuzini chizishni) biladi.

**Ishlash prinsipi:**
VAE ehtimollar nazariyasiga (Bayesian inference) asoslanadi.
*   **Kodlash:** VAE kirish rasmini ($x$) bitta qat’iy nuqtaga ($z$) emas, balki **ehtimollik taqsimotiga** (Normal distribution) aylantiradi. U har bir rasm uchun ikkita parametrni hisoblaydi: O‘rtacha qiymat ($\mu$) va Tarqoqlik ($\sigma$).
*   **Namuna olish (Sampling):** Keyin latent fazoda shu taqsimotdan tasodifiy nuqta ($z \sim N(\mu, \sigma)$) tanlanadi (Reparameterization trick orqali: $z = \mu + \sigma \cdot \epsilon$).
*   **Dekodlash:** Shu tasodifiy nuqta rasmga aylantiriladi.
Bu yondashuv modelga ma’lumotlarning "mazmunini" chuqur tushunishga va kutilmagan, lekin mantiqiy yangi variantlarni yaratishga imkon beradi.

---

### 102. VAE va oddiy autoencoder o‘rtasidagi asosiy farqlarni izohlab bering.

**Javob:**
Asosiy farq **Latent Space (Yashirin fazo)** ning tuzilishi va xususiyatlarida yotadi.

1.  **Oddiy Autoencoder:**
    *   **Maqsad:** Faqat kirishni to‘g‘ri tiklash ($x \approx x'$).
    *   **Fazo:** Yashirin fazo **uzlukli (discrete)** va tartibsiz bo‘lishi mumkin. Masalan, bitta nuqta "kulayotgan odam", ikkinchi nuqta "yig‘layotgan odam". Lekin bu ikki nuqta orasidagi bo‘shliqda nima borligi noma’lum. Agar biz o‘rtadagi nuqtani olib Decoderga bersak, u ma’nosiz shovqin (abrakadabra) chiqaradi.
    *   **Natija:** Generatsiya uchun yaroqsiz.

2.  **Variational Autoencoder (VAE):**
    *   **Maqsad:** Ma’lumotlar taqsimotini o‘rganish.
    *   **Fazo:** Yashirin fazo **uzluksiz (continuous)** va **silliq**.
    *   **Afzalligi:** U "kulayotgan" va "yig‘layotgan" nuqtalar orasidagi barcha nuqtalarni ham to‘ldirib chiqadi (interpolatsiya). Agar biz oraliq nuqtani olsak, Decoder bizga "yarim kulayotgan, yarim yig‘layotgan" yuzni silliq o‘zgarish bilan chiqarib beradi.

---

### 103. Latent space (yashirin fazo) tushunchasini izohlab bering.

**Javob:**
**Tushuncha:**
Latent space — bu ma’lumotlarning siqilgan, mavhum xususiyatlari saqlanadigan ko‘p o‘lchamli matematik fazo (koordinatalar tizimi).
Tasavvur qiling, bizda millionlab turli odam yuzlari (yuqori o‘lchamli piksellar: 1024x1024) bor. Lekin barcha yuzlarni tasvirlash uchun bir nechta asosiy parametr yetarli: jinsi, yoshi, soch rangi, kayfiyati, ko‘zoynagi borligi.
Latent space — aynan mana shu "yashirin" (latent) parametrlarning makonidir.

**Xususiyatlari:**
1.  **O‘xshashlik:** Bu fazoda o‘xshash narsalar yonma-yon turadi (barcha ayollar bir tomonda, erkaklar boshqa tomonda).
2.  **Vektor arifmetikasi:** Biz bu fazoda matematik amallarni bajarishimiz mumkin. Masalan:
    $Vektor("Qirol") - Vektor("Erkak") + Vektor("Ayol") \approx Vektor("Qirolicha")$.
    Yoki: $Vektor("Ko'zoynaksiz odam") + Vektor("Ko'zoynak") = Vektor("Ko'zoynakli odam")$.
Bu bizga tasvirlarni manipulyatsiya qilish (masalan, rasmga sun’iy ko‘zoynak taqish) imkonini beradi.

---

### 104. Generativ modellar tarkibida VAE qaysi maqsadlarda qo‘llanilishini tushuntiring.

**Javob:**
VAE quyidagi amaliy va ijodiy vazifalarda keng qo‘llaniladi:

1.  **Yangi namunalar yaratish:** Mavjud ma’lumotlarga o‘xshash, lekin aynan nusxasi bo‘lmagan yangi rasmlar (masalan, anime qahramonlari), musiqalar yoki yangi kimyoviy dori formulalarini generatsiya qilish.
2.  **Interpolatsiya (Morphing):** Bir rasmdan ikkinchi rasmga silliq o‘tish. Masalan, "Qora mashina" rasmidan sekin-asta "Oq mashina" rasmiga aylantirish (video effektlar uchun).
3.  **Anomaliyalarni aniqlash (Anomaly Detection):** VAE normal ma’lumotlarni yaxshi tiklaydi. Agar unga g‘alati (anomal) rasm (masalan, defektli detal) berilsa, u buni tiklay olmaydi (Reconstruction Loss juda katta bo‘ladi). Shu orqali ishlab chiqarishda nuqsonlar aniqlanadi.
4.  **Denoising va Inpainting:** Tasvirdagi shovqinni tozalash yoki rasmning yirtilgan/yo‘qolgan qismini "xayolan" tiklash.

---

### 105. VAE da KL-divergensiya (Kullback–Leibler divergence) nimani anglatishini va nima uchun qo‘llanilishini izohlab bering.

**Javob:**
VAEning Loss funksiyasi ikki qismdan iborat:
$$ Loss = Loss_{Reconstruction} + \beta \cdot Loss_{KL} $$
1.  **Reconstruction Loss:** Rasmni to‘g‘ri tiklash xatoligi (MSE yoki Cross-Entropy).
2.  **KL-Divergence:** Bu "Jazo" (Regularization) qismi.

**Ma’nosi:**
KL-divergensiya ($D_{KL}(P || Q)$) bizning latent fazoyimizdagi taqsimotning ($P$) standart **Normal taqsimotga (Gaussian, $\mu=0, \sigma=1$)** ($Q$) qanchalik yaqinligini o‘lchaydi.

**Nima uchun kerak?**
Agar KL-loss bo‘lmasa, Encoder nuqtalarni fazoning turli burchaklariga tartibsiz sochib tashlaydi va ular orasida katta bo‘shliqlar qoladi (Generatsiya buziladi).
KL-divergensiya Encoderni majburlaydi: "Barcha nuqtalarni markazga (0 atrofida) zich joylashtir va ular silliq tarqalsin". Bu latent fazoning uzluksiz, zich va tartibli bo‘lishini ta’minlaydi, shunda biz istalgan joydan (ayniqsa markazdan) nuqta olib, ma’noli rasm hosil qila olamiz.

---

### 106. GAN (Generative Adversarial Network) modelining asosiy g‘oyasini tushuntirib bering.

**Javob:**
GAN (Ian Goodfellow, 2014) — bu chuqur o‘qitishdagi inqilobiy g‘oya bo‘lib, u **"O‘yin nazariyasi" (Game Theory)** va **"Raqobat"** prinsipiga asoslanadi.

**G‘oya (Politsiya va O‘g‘ri o‘yini):**
Tizimda ikkita neyron tarmoq bor va ular bir-biriga dushman (adversarial):
1.  **Generator (Qalbaki pul yasovchi):** Uning maqsadi shunday mukammal soxta pul (rasm) yasashki, hech kim uni haqiqiydan ajrata olmasin. U doimiy ravishda o‘z mahoratini oshirib boradi.
2.  **Discriminator (Politsiya/Ekspert):** Uning maqsadi qo‘lidagi pul haqiqiymi yoki soxta ekanini aniqlash. U ham doimiy ravishda hushyorligini oshirib boradi.

**Nash Muvozanati (Nash Equilibrium):**
O‘yin davomida ikkalasi ham kuchayib boradi. Ideal holatda, Generator shunday darajaga yetadiki, u yaratgan rasmlar haqiqiysidan umuman farq qilmaydi va Diskriminator tanga tashlagandek (50/50 ehtimol bilan) adashib qoladi.

---

### 107. GAN tarkibidagi generatorning vazifasi nimadan iborat ekanini tushuntirib bering.

**Javob:**
**Vazifasi:**
Noldan (yo‘q joydan) yangi, realistik ma’lumot yaratish. U ma’lumotlar taqsimotini ($P_{data}$) taqlid qilishni o‘rganadi.

**Ishlash tartibi:**
1.  **Kirish:** Generatorga "ilhom manbai" sifatida **Tasodifiy shovqin vektori (Random Noise, $z$)** beriladi (odatda Normal taqsimotdan olingan 100 o‘lchamli vektor).
2.  **Jarayon:** U bu kichik vektorni neyron qatlamlari (Transposed Convolution / Upsampling) orqali kengaytirib, to‘liq o‘lchamli rasmga aylantiradi.
3.  **O‘qitish:** Boshida bu rasm shunchaki "rangli bo‘tqa" bo‘ladi.
4.  **Signal:** Diskriminator unga: "Bu soxta!" deb signal bergach (Gradient orqali), Generator o‘z parametrlarini o‘zgartiradi: "Keyingi safar mana bu piksellarni o‘zgartirsam, balki Diskriminatorni alday olaman".
5.  **Maqsad:** Diskriminator xatosini maksimallashtirish (uni "Haqiqiy" deb javob berishga majburlash).

---

### 108. GAN tarkibidagi diskriminatorning vazifasini tushuntirib bering.

**Javob:**
**Vazifasi:**
Binar tasniflagich (Binary Classifier) sifatida ishlash: Kirish ma’lumotini **"Real" (Haqiqiy - 1)** yoki **"Fake" (Soxta - 0)** sinfiga ajratish.

**Ishlash tartibi:**
1.  Diskriminatorga aralash ma’lumotlar beriladi:
    *   Yarmi: Haqiqiy datasetdan olingan rasmlar ($x$).
    *   Yarmi: Generatordan chiqqan soxta rasmlar ($\hat{x} = G(z)$).
2.  **Loss Funksiyasi:** Binary Cross Entropy. U haqiqiy rasmni 1 ga, soxtani 0 ga yaqinlashtirishga harakat qiladi.
3.  **O‘qitish:** Agar u adashsa (soxtani Real deb yuborsa yoki aksincha), u jazolanadi va keyingi safar nozik farqlarni (masalan, ko‘z qorachig‘ining noto‘g‘ri shaklini) payqashni o‘rganadi.
4.  Diskriminator qanchalik kuchli bo‘lsa, Generator ham shunchalik kuchli bo‘lishga majbur bo‘ladi (chunki oddiy hiylalar o‘tmay qoladi).

---

### 109. Nega GAN larni o‘qitish murakkab hisoblanadi? Asosiy sabablari-ni izohlab bering.

**Javob:**
GANlarni o‘qitish — bu juda nozik muvozanatni talab qiladigan, beqaror jarayondir.

1.  **Mode Collapse (Modalar kollapsi):** Generator dangasalik qilib, Diskriminatorni aldashning bitta oson yo‘lini topib oladi. Masalan, u faqat bitta turdagi "oq it" rasmini chizishni o‘rganadi va har doim shuni chiqaradi. Diskriminator aldanaveradi, lekin biz xilma-xillikni yo‘qotamiz (barcha rasmlar bir xil bo‘lib qoladi).
2.  **Vanishing Gradient (Gradient yo‘qolishi):** Agar Diskriminator juda tez kuchayib ketsa va 100% aniqlik bilan ishlasa, Generator "qayerga yursam ham xato ekan" deb tushkunlikka tushadi. Matematik jihatdan gradient nolga teng bo‘lib qoladi va Generator o‘qishdan to‘xtaydi.
3.  **Konvergensiya yo‘qligi (Oscillation):** Oddiy modellarda Loss kamayib boradi va minimumda to‘xtaydi. GANda esa ikki tarmoq bir-birini quvib yuradi, Loss doimiy sakrab turishi mumkin (Limit Cycle). Model hech qachon barqaror holatga kelmasligi mumkin.

---

### 110. DCGAN oddiy GAN dan nimasi bilan farq qilishini izohlab bering.

**Javob:**
DCGAN (Deep Convolutional GAN, 2015) — bu GAN arxitekturasini barqarorlashtirish uchun kiritilgan birinchi muvaffaqiyatli standart ("Best Practices") to‘plamidir. Oddiy GAN (MLP asosida) faqat kichik va oddiy rasmlarda ishlardi.

**Farqlari va Asosiy Qoidalari:**
1.  **To‘liq CNN:** Ikkala tarmoq ham to‘liq Konvolyutsion qatlamlardan quriladi. Fully Connected (Dense) qatlamlar butunlay olib tashlanadi.
2.  **Pooling o‘rniga Stride:**
    *   O‘lchamni kichraytirish uchun Max Pooling o‘rniga **Strided Convolution** ishlatiladi (Diskriminatorda).
    *   O‘lchamni kattalashtirish uchun esa **Transposed Convolution** (Deconvolution) ishlatiladi (Generatorda).
    *   Bu modelga o‘lchamni silliq o‘zgartirishni o‘rganishga imkon beradi.
3.  **Batch Normalization:** Har bir qatlamda (chiqishdan tashqari) Batch Norm ishlatiladi. Bu Mode Collapse ni oldini olishga va o‘qitishni barqaror qilishga katta yordam beradi.
4.  **Aktivatsiyalar:**
    *   Generatorda: Hamma joyda **ReLU**, faqat oxirida **Tanh** (rasmni [-1, 1] ga tushirish uchun).
    *   Diskriminatorda: Hamma joyda **Leaky ReLU** (gradient o‘lmasligi uchun).
