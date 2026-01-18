# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (121-130)

### 121. Mustahkamlangan o‘qitishda agent va muhit (environment) tushunchalarini izohlang.

**Javob:**
Mustahkamlangan o‘qitish (RL) ikkita asosiy ob’ektning o‘zaro ta’siriga asoslanadi:

1.  **Agent (Ishchi/O‘yinchi):**
    *   Bu qaror qabul qiluvchi "miya" (neyron tarmoq).
    *   Uning vazifasi vaziyatni kuzatish va eng foydali **Harakatni (Action)** tanlashdir.
    *   Misol: Super Mario o‘yinidagi Mario (yoki uni boshqaradigan AI).
2.  **Environment (Muhit/Olam):**
    *   Agent yashaydigan va harakat qiladigan dunyo.
    *   Muhit Agentdan harakatni qabul qiladi va unga javoban **Yangi holatni (State)** va **Mukofotni (Reward)** qaytaradi.
    *   Misol: O‘yinning o‘zi, dushmanlar, to‘siqlar, fizika qonunlari.

Jarayon tsikl (loop) ko‘rinishida bo‘ladi: Agent harakat qiladi $\to$ Muhit o‘zgaradi va baho beradi $\to$ Agent yangi holatni ko‘rib, keyingi harakatni o‘ylaydi.

---

### 122. Reward (mukofot) tushunchasi nima va u qanday shakllanishini tushuntirib bering.

**Javob:**
**Tushuncha:**
Reward (Mukofot, $R$) — bu Agentning qilgan harakati qanchalik yaxshi yoki yomon ekanligini bildiruvchi yagona sonli signaldir. RLda Agentning yagona maqsadi — vaqt davomida yig‘iladigan **jami mukofotni maksimallashtirishdir**.

**Shakllanishi:**
Mukofot funksiyasini inson (muhandis) oldindan belgilab beradi:
*   **Musbat mukofot (+):** Rag‘batlantirish. Masalan, o‘yinda tanga yig‘sa (+10 ball) yoki marraga yetib borsa (+100 ball).
*   **Manfiy mukofot (-):** Jazolash. Masalan, dushmanga tegib ketas (-50 ball), vaqtni behuda sarflasa (-1 ball har soniyada) yoki devorga urilsa.
*   **Nol (0):** Neytral holat.
Agent qaysi harakatlar unga ko‘proq "plyus" olib kelishini sinov-xato (trial and error) orqali o‘rganadi.

---

### 123. Policy (siyosat, strategiya) nima va agent xatti-harakatlarida qanday rol o‘ynashini izohlab bering.

**Javob:**
**Tushuncha:**
Policy ($\pi$) — bu Agentning "xulq-atvor qoidalari to‘plami". U Agentga ma’lum bir vaziyatda ($s$) qanday harakat ($a$) qilish kerakligini aytib beradi. Bu funksiya: $a = \pi(s)$.

**Turlari:**
1.  **Deterministik:** Aniq qoida. "Agar dushmanni ko‘rsang $\to$ Sakra".
2.  **Stoxastik (Ehtimolli):** Ehtimollik taqsimoti. "Agar dushmanni ko‘rsang $\to$ 80% ehtimol bilan sakra, 20% ehtimol bilan egil".

**Roli:**
Policy — bu RLning yakuniy mahsulidir. Biz o‘qitish davomida aynan eng optimal Policy ni (ya’ni har doim g‘alabaga yetaklaydigan strategiyani) topishga harakat qilamiz. Neyron tarmoq aslida shu Policy funksiyasini approksimatsiya qiladi.

---

### 124. Q-learning algoritmining asosiy g‘oyasini tushuntirib bering.

**Javob:**
Q-learning — bu "Qiymatga asoslangan" (Value-based) algoritmdir.
U Policy ni to‘g‘ridan-to‘g‘ri qidirmaydi, balki har bir harakatning "qadrini" (Quality - shuning uchun Q) baholashni o‘rganadi.

**G‘oya:**
**Q-funksiya ($Q(s, a)$):** "Agar men hozir $s$ holatda bo‘lsam va $a$ harakatni qilsam, kelajakda jami qancha mukofot olishim mumkin?" degan savolga javob beradi.
*   Masalan: $Q(\text{oldinda chuqur}, \text{oldinga yurish}) = -100$ (O‘lasan).
*   $Q(\text{oldinda chuqur}, \text{sakrash}) = +50$ (Tirik qolasan va tanga olasan).

Agent har doim eng katta $Q$ qiymatga ega bo‘lgan harakatni tanlaydi (Greedy strategy).
O‘qitish davomida Agent ushbu Q-qiymatlarni Bellman tenglamasi orqali yangilab, aniqlashtirib boradi.

---

### 125. DQN (Deep Q-Network) da neyron tarmoq qanday vazifani bajarishini izohlab bering.

**Javob:**
**Muammo:**
Klassik Q-learningda barcha holatlar va harakatlar uchun Q-qiymatlar **jadvalda (Q-table)** saqlanardi.
Lekin murakkab muhitlarda (masalan, shaxmat yoki video o‘yin) holatlar soni milliardlab bo‘lishi mumkin. Bunday ulkan jadvalni xotiraga sig‘dirib bo‘lmaydi.

**DQN Yechimi:**
Jadval o‘rniga **Chuqur Neyron Tarmoq** ishlatiladi.
*   Kirish: O‘yin holati (masalan, ekran piksellari).
*   Chiqish: Barcha mumkin bo‘lgan harakatlar uchun Q-qiymatlar (masalan: Chap: 0.2, O‘ng: 0.9, Sakrash: 1.5).

Neyron tarmoq Q-funksiyani **approksimatsiya** qiladi (taxminiy hisoblaydi). Biz endi har bir holatni yodlab olishimiz shart emas, tarmoq umumiy qonuniyatlarni o‘rganadi va ko‘rmagan holati uchun ham taxminiy Q-qiymat chiqara oladi.

---

### 126. Experience replay (tajribani qayta ijro qilish) nima va nima uchun qo‘llanilishini tushuntiring.

**Javob:**
**Tushuncha:**
Agent o‘ynash davomida o‘z boshidan kechirgan har bir qadamni (Holat, Harakat, Mukofot, Yangi holat) maxsus "Xotira buferi"ga (Replay Buffer) saqlab boradi.
O‘qitish paytida u so‘nggi ko‘rgan narsasini emas, balki shu buferdan **tasodifiy tanlab olingan** eski voqealarni olib, modelni o‘qitadi.

**Nima uchun kerak?**
1.  **Bog‘liqlikni buzish (Breaking Correlation):** O‘yinda ketma-ket keladigan kadrlar bir-biriga juda o‘xshash. Agar model faqat ketma-ket o‘qisa, u bir tomonga og‘ib ketadi (bias) va "unitib qo‘yish" kasaliga uchraydi. Tasodifiy tanlash bu bog‘liqlikni uzadi va o‘qitishni barqaror qiladi.
2.  **Samaradorlik:** Bitta tajribadan (qimmatli vaziyatdan) bir necha marta foydalanish mumkin.

---

### 127. Target network nima va DQN da u nima maqsadda qo‘llanilishini izohlab bering.

**Javob:**
**Muammo:**
DQNda biz Q-tarmoqni o‘qitayotganda, nishon (target) qiymatni hisoblash uchun ham **o‘sha tarmoqning o‘zidan** foydalanamiz.
Bu xuddi "dumini quvayotgan it"ga o‘xshaydi: har qadamda nishon ham o‘zgarib qochib ketadi. Bu o‘qitishni juda beqaror (tebranuvchan) qiladi.

**Yechim (Target Network):**
Biz ikkita tarmoq yaratamiz:
1.  **Main Network:** Har qadamda yangilanadi (o‘qiydi).
2.  **Target Network:** Bu Main tarmoqning nusxasi, lekin uning vaznlari **muzlatib qo‘yilgan**. U har 1000 yoki 10000 qadamda bir marta Main tarmoqdan ko‘chirib olinadi.

**Maqsad:**
Nishonni hisoblash uchun Target Network ishlatiladi. U tez o‘zgarmagani uchun, nishon barqaror turadi va Main Network unga qarab bemalol yaqinlasha oladi (konvergensiya).

---

### 128. Policy gradient yondashuvi nima ekanini tushuntirib bering.

**Javob:**
Q-learning (DQN) bilvosita yo‘l bilan (avval Q-qiymatni topib, keyin eng kattasini tanlab) ishlaydi.
**Policy Gradient** esa to‘g‘ridan-to‘g‘ri yo‘l tutadi.

**G‘oya:**
Biz Policy ni (ya’ni harakat qilish ehtimolligini) to‘g‘ridan-to‘g‘ri optimallashtiramiz.
*   Agar biror harakat yaxshi natija (ko‘p mukofot) bersa, neyron tarmoqning o‘sha harakatni tanlash **ehtimolligini oshiramiz**.
*   Agar yomon natija bersa, ehtimolligini **kamaytiramiz**.
Matematik jihatdan biz mukofot funksiyasining gradientini hisoblaymiz va Policy parametrlarini shu gradient bo‘yicha o‘zgartiramiz (Gradient Ascent).

**Afzalligi:**
Stoxastik (tasodifiy) siyosatlarni o‘rganishi mumkin va uzluksiz harakatlar fazosida (masalan, robot qo‘lini boshqarishda burchaklarni aniq belgilash) DQNdan ko‘ra samaraliroq.

---

### 129. Actor–Critic modellarining asosiy afzalliklarini izohlab bering.

**Javob:**
Bu usul "Value-based" (DQN) va "Policy-based" (Policy Gradient) yondashuvlarining eng yaxshi tomonlarini birlashtiradi.

Tizimda ikkita tarmoq bor:
1.  **Actor (Aktyor):** Harakatni tanlaydi (Policy-based). U sahnada o‘ynaydi.
2.  **Critic (Tanqidchi):** Aktyorning harakatiga baho beradi (Value-based). U hakamlik qiladi.

**Afzalligi:**
*   Oddiy Policy Gradient juda shovqinli bo‘ladi (chunki mukofot faqat o‘yin oxirida kelishi mumkin, qaysi harakat to‘g‘ri ekanini bilish qiyin).
*   Actor-Criticda esa **Tanqidchi** har bir qadamda Aktyorga darhol "bahosini" (TD-error) aytib turadi ("Yaxshi yurish qilding" yoki "Xato qilding"). Bu o‘qitishni ancha tezlashtiradi va tebranishlarni (variance) kamaytiradi.

---

### 130. Ob’ektlarni aniqlash (object detection) nima ekanini tushuntirib bering.

**Javob:**
Ob’ektlarni aniqlash — bu Kompyuterni Ko‘rish (Computer Vision) sohasidagi murakkab vazifa bo‘lib, u ikkita ishni bir vaqtda bajarishni talab qiladi:

1.  **Tasniflash (Classification):** Rasmda nima borligini aytish (masalan, "Mashina", "Odam").
2.  **Lokalizatsiya (Localization):** O‘sha ob’ekt rasmning **qayerda** joylashganini aniq ko‘rsatib berish. Odatda bu ob’ekt atrofida to‘rtburchak chizish (Bounding Box) orqali amalga oshiriladi (koordinatalar: $x, y, width, height$).

Agar rasmda bir nechta ob’ekt bo‘lsa (masalan, ko‘chadagi tirbandlik), Object Detection modeli ularning **har birini** alohida-alohida topib, ajratib berishi kerak.
Mashhur modellari: YOLO, SSD, Faster R-CNN.