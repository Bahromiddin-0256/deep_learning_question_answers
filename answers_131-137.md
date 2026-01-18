# CHUQUR O‘QITISH (DEEP LEARNING) – SAVOL-JAVOBLAR (131-137)

### 131. Tasniflash (klassifikatsiya) va segmentatsiya vazifalari o‘rtasidagi farqlarni tushuntirib bering.

**Javob:**
Bu ikki vazifa tasvirni tushunishning turli darajalarini ifodalaydi.

1.  **Tasniflash (Image Classification):**
    *   **Savol:** "Bu rasmda nima bor?"
    *   **Javob:** Bitta umumiy yorliq (Label).
    *   **Misol:** Rasmga qarab "Bu - Mushuk" deyish. Model rasmning qaysi qismida mushuk borligini ko‘rsatmaydi, shunchaki uning borligini tasdiqlaydi. Butun rasm uchun bitta javob olinadi.

2.  **Segmentatsiya (Image Segmentation):**
    *   **Savol:** "Rasmning qaysi piksellari qaysi ob’ektga tegishli?"
    *   **Javob:** Pikselma-piksel xarita (Mask).
    *   **Misol:** Rasmdagi har bir nuqtani bo‘yab chiqish: mushukning burni qizil, qulog‘i qizil, orqasidagi divan esa yashil rangda. Bu yerda biz ob’ektning aniq **konturlarini** va shaklini olamiz. Segmentatsiya ancha nozik va batafsilroq vazifadir.

---

### 132. Semantik segmentatsiya (semantic segmentation) nima ekanini izohlab bering.

**Javob:**
Semantik segmentatsiya — bu rasmdagi har bir pikselni ma’lum bir **sinfga (class)** tegishli deb belgilash jarayonidir.
Bunda bir xil sinfga mansub ob’ektlar **birlashtirib yuboriladi**.

**Misol:**
Ko‘cha harakati rasmi.
*   Model barcha mashinalarni "ko‘k" rangga, barcha odamlarni "qizil" rangga, yo‘lni "kulrang" rangga bo‘yaydi.
*   **Muhim jihati:** Agar rasmda yonma-yon turgan 5 ta mashina bo‘lsa, semantik segmentatsiya ularning hammasini bitta katta "mashinalar to‘plami" (bitta rang) deb qaraydi. U mashina-1 va mashina-2 ni bir-biridan ajratmaydi. U uchun faqat "bu piksel mashinaga tegishli" degan fakt muhim.

---

### 133. Instance segmentatsiya nima va u semantik segmentatsiyadan nimasi bilan farq qilishini tushuntirib bering.

**Javob:**
Instance (Namuna) segmentatsiyasi — bu Semantik segmentatsiya va Ob’ektni aniqlash (Object Detection) ning birlashmasidir.

**Farqi:**
U nafaqat piksellarning sinfini aniqlaydi, balki **har bir alohida ob’ektni (instance)** bir-biridan ajratadi.
*   **Semantik:** 5 ta mashina = Bitta katta ko‘k dog‘.
*   **Instance:** 5 ta mashina = 5 ta alohida (ko‘k, yashil, sariq...) rangdagi ob’ekt.

Bu ancha qiyin vazifa hisoblanadi (masalan, Mask R-CNN modeli bajaradi), chunki model ustma-ust tushgan odamlarni yoki bir-biriga yopishib turgan narsalarni qayerda tugab, qayerda boshlanishini aniq bilishi kerak.

---

### 134. U-Net arxitekturasining asosiy g‘oyasini va tuzilishini tushuntirib bering.

**Javob:**
U-Net (2015) — asosan tibbiy tasvirlarni (masalan, hujayralarni yoki o‘simtalarni) segmentatsiya qilish uchun yaratilgan, lekin hozirda sohada standartga aylangan arxitektura.

**Tuzilishi ("U" harfi shaklida):**
1.  **Encoder (Chap tomon - Contracting path):** Oddiy CNN kabi ishlaydi. Rasmni qatlamdan-qatlamga kichraytirib (Pooling), muhim xususiyatlarni ("Nima bor?") ajratib oladi.
2.  **Decoder (O‘ng tomon - Expanding path):** Kichraygan xususiyatlarni qaytadan kattalashtirib (Up-convolution), asl rasm o‘lchamiga qaytaradi. Maqsad — xususiyatlarning "Qayerda" joylashganini tiklash.
3.  **Skip Connections (Ko‘priklar):** Chap tomondagi (Encoder) har bir qatlamning chiqishi to‘g‘ridan-to‘g‘ri o‘ng tomondagi (Decoder) mos qatlamga ulanadi.

**G‘oya:**
Kattalashtirish (Decoder) jarayonida mayda detallar yo‘qolishi mumkin. Skip connectionlar orqali Decoder asl rasmning detallarini Enkoderdan ko‘chirib oladi. Bu juda yuqori aniqlikdagi chegaralarni (segmentatsiya maskasini) hosil qilish imkonini beradi.

---

### 135. YOLO arxitekturasi R-CNN oilasiga mansub modellar bilan taqqoslang, asosiy farqlarini izohlab bering.

**Javob:**
Bular Ob’ektlarni aniqlashdagi (Object Detection) ikki xil falsafadir.

1.  **R-CNN oilasi (R-CNN, Fast R-CNN, Faster R-CNN):**
    *   **Usul:** Ikki bosqichli (Two-stage).
        1.  Rasmda ob’ekt bo‘lishi mumkin bo‘lgan joylarni (Region Proposals) topish.
        2.  Har bir joyni alohida tekshirib, klassifikatsiya qilish va to‘rtburchakni to‘g‘rilash.
    *   **Natija:** Juda aniq, lekin sekin (real vaqtda ishlash qiyin).

2.  **YOLO (You Only Look Once - Faqat bir marta qarash):**
    *   **Usul:** Bir bosqichli (One-stage).
    *   U rasmni bo‘laklarga (grid) bo‘ladi va bitta neyron tarmoq orqali bir urinishda hamma ob’ektlarni va ularning joylashuvini bashorat qiladi.
    *   **Natija:** Juda tez (45-150 FPS, real vaqtda bemalol ishlaydi). Aniqligi R-CNNdan biroz pastroq bo‘lishi mumkin (ayniqsa mayda ob’ektlarda), lekin tezligi evaziga juda ommabop.

---

### 136. Anchor boxes nima va ularning object detection dagi vazifasini tushuntirib bering.

**Javob:**
**Muammo:**
Ob’ektlar har xil shaklda bo‘ladi: Odam — ingichka va uzun, Mashina — keng va past, Koptok — kvadrat. Modelga shunchaki "to‘rtburchak top" deyish qiyin, chunki u qanday shaklni qidirishni bilmaydi.

**Yechim (Anchor Boxes):**
Biz modelga oldindan tayyorlangan **shablonlar (andozalar)** to‘plamini beramiz. Bular Anchor Boxlardir.
Masalan, har bir nuqta uchun 3 ta andoza:
1.  Kvadrat (1:1).
2.  Uzunchoq (1:2) - odamlar uchun.
3.  Yotiq (2:1) - mashinalar uchun.

**Vazifasi:**
Model noldan to‘rtburchak chizmaydi. U shu andozalardan birini tanlaydi va uni biroz o‘zgartiradi ("Mana bu uzunchoq andozani ol, sal kengaytir va o‘ngga sur"). Bu o‘qitishni osonlashtiradi va turli shakldagi ob’ektlarni aniqlashni yaxshilaydi.

---

### 137. Bounding box regression (chegaralovchi to‘rtburchak parametrlarini regressiya qilish) nima ekanini tushuntirib bering.

**Javob:**
Bu Ob’ektni aniqlash modellarining oxirgi, "sayqallash" qismidir.

**Jarayon:**
Model ob’ektni topdi va taxminiy to‘rtburchak (Anchor Box) ni tanladi. Lekin bu to‘rtburchak ob’ektga mukammal tushmasligi mumkin (masalan, odamning boshi tashqarida qolgan).
Model 4 ta sonni (regressiya qiymatlarini) bashorat qiladi:
1.  $\Delta x$: Markazni qanchaga surish kerak (gorizontal).
2.  $\Delta y$: Markazni qanchaga surish kerak (vertikal).
3.  $\Delta w$: Enini qanchaga cho‘zish/qisish kerak.
4.  $\Delta h$: Bo‘yini qanchaga cho‘zish/qisish kerak.

**Maqsad:**
Ushbu tuzatishlar (offset) yordamida qo‘pol to‘rtburchakni haqiqiy ob’ektni (Ground Truth Box) zich o‘rab turadigan aniq to‘rtburchakka aylantirish. Bu jarayon xuddi "kostyum-shimni mijozning qomatiga moslab tikish"ga o‘xshaydi.
