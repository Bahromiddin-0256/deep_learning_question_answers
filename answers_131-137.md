# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (131-137)

### 131. Tasniflash (klassifikatsiya) va segmentatsiya vazifalari o‘rtasidagi farqlarni tushuntirib bering.

**Javob:**
Bu ikki vazifa tasvirni tushunishning turli darajalarini ifodalaydi.

1.  **Tasvirni Tasniflash (Image Classification):**
    *   **Savol:** "Bu rasmda nima bor?"
    *   **Javob:** Bitta umumiy yorliq (Label).
    *   **Xususiyati:** Model ob’ektning joylashgan o‘rnini yoki shaklini bilmaydi, faqat uning borligini aytadi. Butun rasm bitta javobga ega bo‘ladi.
    *   **Misol:** Google Photos da "Itlar" deb qidirganda it bor rasmlarning chiqishi.

2.  **Tasvirni Segmentatsiya qilish (Image Segmentation):**
    *   **Savol:** "Ob’ekt rasmning aynan qaysi piksellarida joylashgan?"
    *   **Javob:** Pikselma-piksel xarita (Mask). Rasm bilan bir xil o‘lchamdagi matritsa bo‘lib, unda har bir pikselga sinf rangi beriladi.
    *   **Xususiyati:** Bu eng batafsil tushunish darajasidir. Biz ob’ektning aniq konturini (chegarasini) va shaklini olamiz.
    *   **Misol:** Zoom fonini o‘zgartirishda dastur sizni ajratib olib (segmentatsiya qilib), fonni xiralashtiradi.

---

### 132. Semantik segmentatsiya (semantic segmentation) nima ekanini izohlab bering.

**Javob:**
Semantik segmentatsiya (Semantic Segmentation) — bu rasmdagi **har bir pikselni** ma’lum bir **sinfga (class)** (masalan, "Odam", "Yo‘l", "Daraxt", "Osmon") tegishli deb belgilash jarayonidir.

**Asosiy qoidasi:**
Bir xil sinfga mansub barcha ob’ektlar **bitta butun** deb qaraladi (birlashtirib yuboriladi).
*   Agar rasmda yonma-yon turgan 5 ta odam bo‘lsa, Semantik Segmentatsiya ularning hammasini "Odam" sinfiga kiritadi va bitta rangga (masalan, qizilga) bo‘yaydi.
*   U "Odam #1" va "Odam #2"ni bir-biridan ajratmaydi. Ular bitta katta "Odamlar massasi" bo‘lib ko‘rinadi.

**Qo‘llanilishi:**
*   **Avtonom mashinalar:** Mashina uchun "bu yerda yo‘l bor", "bu yerda to‘siq bor" ekanini bilish muhim. To‘siq 1 ta mashinami yoki 2 ta mashinami ekanligi ikkinchi darajali (baribir urilmaslik kerak).
*   **Tibbiiyot:** Rentgen rasmida o‘simta to‘qimasini (tumor) sog‘lom to‘qimadan ajratish.

---

### 133. Instance segmentatsiya nima va u semantik segmentatsiyadan nimasi bilan farq qilishini tushuntirib bering.

**Javob:**
Instance (Namuna) Segmentatsiyasi — bu Semantik Segmentatsiya va Ob’ektni Aniqlash (Object Detection) ning kuchli kombinatsiyasidir.

**Farqi:**
U nafaqat piksellarning sinfini aniqlaydi, balki **har bir alohida ob’ektni (instance)** individual tarzda ajratadi.
*   **Semantik:** 5 ta qo‘y = Bitta katta oq dog‘.
*   **Instance:** 5 ta qo‘y = 5 ta alohida (har xil rangdagi) qo‘y. Qo‘y #1, Qo‘y #2, ..., Qo‘y #5.

**Qiyinchiligi:**
Model bir-birini to‘sib turgan (occlusion) ob’ektlarni to‘g‘ri ajratishi kerak. Masalan, bir odam ikkinchi odamning orqasida tursa, Semantik segmentatsiya ularni qo‘shib yuboradi, Instance segmentatsiya esa ularning chegarasini aniq topib, ikki xil shaxs ekanini ko‘rsatadi.
Mashhur arxitektura: **Mask R-CNN**.

---

### 134. U-Net arxitekturasining asosiy g‘oyasini va tuzilishini tushuntirib bering.

**Javob:**
U-Net (Ronneberger et al., 2015) — dastlab biomeditsina tasvirlarini (hujayralarni) segmentatsiya qilish uchun yaratilgan, lekin hozirda segmentatsiya sohasining "oltin standarti"ga aylangan arxitektura.

**Tuzilishi ("U" harfi shaklida):**
1.  **Encoder (Chap qanot - Siqish):** Oddiy CNN kabi ishlaydi. Rasmni qatlamdan-qatlamga kichraytirib (Pooling), muhim xususiyatlarni ("Nima bor?") ajratib oladi. Kontekstni tushunadi.
2.  **Decoder (O‘ng qanot - Kengaytirish):** Siqilgan xususiyatlarni qaytadan kattalashtirib (Up-convolution), asl rasm o‘lchamiga qaytaradi. Maqsad — xususiyatlarning **"Qayerda"** joylashganini aniqlash.
3.  **Skip Connections (Ko‘priklar):** Bu U-Netning sehrli qismidir. Chap tomondagi (Encoder) har bir qatlamning chiqishi to‘g‘ridan-to‘g‘ri o‘ng tomondagi (Decoder) mos qatlamga ulanadi (Concatenate).

**G‘oya:**
Kattalashtirish jarayonida mayda detallar va aniq chegaralar yo‘qolishi mumkin. Skip connectionlar orqali Decoder asl rasmning yuqori aniqlikdagi detallarini Encoderdan ko‘chirib oladi va mukammal segmentatsiya xaritasini chizadi.

---

### 135. YOLO arxitekturasi R-CNN oilasiga mansub modellar bilan taqqoslang, asosiy farqlarini izohlab bering.

**Javob:**
Bular Ob’ektlarni Aniqlash (Object Detection) dagi ikki xil yondashuvdir.

1.  **R-CNN oilasi (Faster R-CNN): Ikki bosqichli (Two-stage).**
    *   **1-bosqich:** Rasmda ob’ekt bo‘lishi mumkin bo‘lgan shubhali joylarni (Region Proposals) topadi (yuzlab to‘rtburchaklar).
    *   **2-bosqich:** Har bir to‘rtburchakni alohida neyron tarmoqdan o‘tkazib, "Bu nima?" deb tekshiradi va to‘rtburchakni to‘g‘rilaydi.
    *   **Natija:** Juda aniq, lekin sekin (chunki har bir taklifni qayta ishlash kerak).

2.  **YOLO (You Only Look Once): Bir bosqichli (One-stage).**
    *   **G‘oya:** Rasmni kataklarga (grid) bo‘ladi (masalan 13x13).
    *   Bitta katta neyron tarmoq rasmni **bir marta** ko‘rib chiqadi va har bir katak uchun birdaniga "Bu yerda ob’ekt bormi?", "Qaysi sinf?", "Koordinatasi nima?" degan savollarga javob beradi.
    *   **Natija:** Juda tez (Real vaqtda, videolarda ishlaydi). Aniqligi R-CNNdan biroz pastroq bo‘lishi mumkin (ayniqsa mayda ob’ektlarda), lekin tezligi evaziga juda ommabop.

---

### 136. Anchor boxes nima va ularning object detection dagi vazifasini tushuntirib bering.

**Javob:**
**Muammo:**
Modelga "Rasmda ob’ekt top va unga to‘rtburchak chiz" deyish qiyin, chunki ob’ektlar har xil shaklda bo‘ladi (odam — uzunchoq, mashina — yotiq, koptok — kvadrat). Model noldan boshlab to‘rtburchak chizishga qiynaladi (konvergensiya sekin bo‘ladi).

**Yechim (Anchor Boxes):**
Biz modelga yordam sifatida **oldindan tayyorlangan shablonlar (andozalar)** to‘plamini beramiz. Bular Anchor Boxlardir.
Har bir nuqta uchun (masalan) 3 xil andoza beriladi:
1.  Kvadrat (1:1).
2.  Uzunchoq (1:2) - tik turgan ob’ektlar (odamlar) uchun.
3.  Yotiq (2:1) - yotiq ob’ektlar (mashinalar) uchun.

**Vazifasi:**
Model endi "to‘rtburchak chizmaydi", balki:
1.  Mavjud andozalardan qaysi biri ob’ektga eng ko‘p o‘xshashini tanlaydi.
2.  O‘sha andozani biroz **tuzatadi** (sal kengaytiradi yoki suradi).
Bu vazifani (tuzatish kiritishni) o‘rganish noldan chizishdan ko‘ra ancha osonroq.

---

### 137. Bounding box regression (chegaralovchi to‘rtburchak parametrlarini regressiya qilish) nima ekanini tushuntirib bering.

**Javob:**
Bounding Box Regression — bu Ob’ektni aniqlash modellarining "sayqallash" (fine-tuning) qismidir.

**Jarayon:**
Model Anchor Boxni (andozani) tanladi, lekin bu andoza ob’ektga mukammal tushmasligi mumkin (masalan, odamning boshi sal chiqib qolgan).
Model to‘rtburchakni to‘g‘rilash uchun 4 ta parametrni (regressiya qiymatlarini) bashorat qiladi:
1.  $\Delta x$: Markazni gorizontal surish.
2.  $\Delta y$: Markazni vertikal surish.
3.  $\Delta w$: Enini cho‘zish/qisish (logarifmik shkalada).
4.  $\Delta h$: Bo‘yini cho‘zish/qisish.

**Formula:**
Haqiqiy quti ($b$) va bashorat qilingan quti ($\hat{b}$) o‘rtasidagi farqni (MSE yoki IoU Loss) minimallashtirish orqali model ushbu tuzatishlarni o‘rganadi.
Natijada, qo‘pol andoza haqiqiy ob’ektni (Ground Truth) zich o‘rab turadigan aniq ramkaga aylanadi.