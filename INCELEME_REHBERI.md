# Proje İnceleme Rehberi (Grup İçin)

**Ders:** BYM303 Algoritma Analizi — Dönem Projesi
**Konu:** 2. Derin Öğrenme (CIFAR-10) — *Yapay Zekâ Algoritmalarında Zaman, Bellek, Enerji ve Performans Karmaşıklığının Deneysel Analizi*
**Teslim:** 18 Aralık 2026 Cuma 23:59 — e-posta ile (rapor + kod ZIP veya Git linki)

| Öğrenci No | Ad Soyad |
|---|---|
| 240609043 | Baran Kemal Kılıç |
| 240609009 | İrem Serra Onmaz |
| 240609050 | Nihat Güney |
| 240609014 | Süleyman Canberk Eğriboz |

---

## 1. Proje tek cümlede ne yapıyor?

CIFAR-10 üzerinde 5 farklı derin öğrenme modelini (MLP, SmallCNN, MediumCNN, MobileNetV2, ResNet-18)
**4 farklı veri boyutunda** (5k, 12.5k, 25k, 50k görüntü) ve **5 farklı seed** ile sıfırdan eğitiyoruz
(**toplam 100 eğitim**). Her eğitim ve tahmin (inference) için ayrı ayrı **süre, RAM, VRAM, enerji (Joule),
karbon (g CO₂)** ve **Accuracy/Precision/Recall/F1/ROC-AUC** ölçüyoruz; sonra bunları her modelin **teorik
karmaşıklığıyla** (parametre sayısı P, görüntü başına işlem sayısı F, aktivasyon sayısı A) karşılaştırıyoruz.

Föydeki tüm zorunlu maddeler karşılanıyor:

| Föy beklentisi | Projede nerede? |
|---|---|
| En az 3 model | 5 model → `src/models.py` |
| Training ve inference ayrı ölçüm | `src/run_experiments.py` → `train_*` ve `infer_*` sütunları |
| Zaman (train / inference) | `train_time_s`, `infer_time_s`, `latency_b1_ms`, `cpu_infer_ms_per_img` |
| Peak RAM, Peak VRAM | `train_peak_ram_mb`, `train_peak_vram_torch_mb`, `..._nvml_mb` |
| Enerji (J/Wh) + karbon | NVML donanım sayacı (`train_gpu_energy_j`) + CodeCarbon (`train_cc_*`) |
| Accuracy, Precision, Recall, F1, ROC-AUC | `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `roc_auc_macro` |
| Parametre sayısı / model büyüklüğü | `results/model_complexity.csv` |
| En az 5 tekrar, mean ± std | 5 seed → `results/summary_by_model_fraction.csv` |
| Veri boyutuna göre eğriler | `figures/fig1`, `fig2`, `fig3` |
| Teorik karmaşıklık vs deney | Makale Bölüm III + V-B, `fig8`, `fig5`, `tab_scaling` |
| Performans–maliyet trade-off | `figures/fig4` (Pareto), `fig9` |
| Donanım / sürüm bilgisi | `results/system_info.json`, makalede Tablo II |
| Konferans formatında İngilizce makale | `paper/main.tex` (IEEE conference şablonu – AISCI 2027) |

---

## 2. Klasör yapısı — hangi dosya ne işe yarar?

```
project/
├── README.md                  İngilizce repo açıklaması (hocaya / GitHub'a)
├── INCELEME_REHBERI.md        Bu dosya
├── requirements.txt           Kütüphane listesi
├── src/
│   ├── models.py              5 modelin mimarisi (PyTorch)
│   ├── complexity.py          Teorik P, MACs, aktivasyon hesabı (forward hook ile)
│   ├── data.py                CIFAR-10 yükleme, stratified alt küme, GPU'da augmentation
│   ├── measure.py             Zaman, RAM, VRAM, NVML enerji, CodeCarbon ölçüm sınıfları
│   ├── run_experiments.py     ANA DENEY: 100 koşuyu çalıştırır, results/raw_runs.csv yazar
│   ├── analyze.py             İstatistik, eğim (b) hesapları, grafikler, LaTeX tabloları
│   └── bench_precision_layout.py   fp16/bf16 ve bellek düzeni hız testi (ek deney)
├── results/
│   ├── raw_runs.csv           ⭐ HAM VERİ: her satır bir eğitim (100 satır)
│   ├── summary_by_model_fraction.csv   model × veri boyutu için ortalama ve std
│   ├── scaling_exponents.csv  T = c·n^b eğimleri (b) ve %95 güven aralıkları
│   ├── model_complexity.csv   teorik parametre / MACs / aktivasyon
│   ├── efficiency_indicators.csv  enerji başına doğruluk vb.
│   ├── analysis_report.json   korelasyonlar, Pareto cephesi
│   ├── welch_tests.csv        modeller arası istatistiksel testler
│   ├── system_info.json       donanım ve yazılım sürümleri, boşta GPU gücü
│   └── precision_layout_benchmark.txt  ek deney çıktısı
├── figures/                   Tüm grafikler (PNG = sunum için, PDF = makale için)
├── paper/
│   ├── main.tex               ⭐ MAKALE (IEEE formatı, İngilizce)
│   ├── refs.bib               Kaynakça (hocanın verdiği 7 makale + 18 temel kaynak)
│   ├── numbers.tex            Makaledeki TÜM sayılar (analyze.py otomatik üretir)
│   ├── tables/*.tex           Tablolar (otomatik üretilir)
│   └── figures/*.pdf          Makaledeki şekiller
└── results_run_log.txt        Deneyin konsol çıktısı (kanıt niteliğinde)
```

---

## 3. Her şeyi nasıl incelersiniz? (önerilen sıra)

### Adım 1 — Grafiklere bakın (10 dk)
`figures/` klasöründeki PNG'leri açın. Her biri makalede bir bulguya karşılık gelir:

| Grafik | Ne gösteriyor? | Ana mesaj |
|---|---|---|
| `fig1_train_time_scaling.png` | Eğitim süresi vs veri boyutu (log-log) | Eğim b ≈ 1 → süre **doğrusal**, teori O(n) doğrulandı |
| `fig2_train_energy_scaling.png` | Eğitim enerjisi vs veri boyutu | Enerji de doğrusal büyüyor |
| `fig3_accuracy_vs_n.png` | Öğrenme eğrileri | Veri arttıkça doğruluk artıyor ama **azalan getiri** |
| `fig4_tradeoff_accuracy_energy.png` | Doğruluk vs enerji, Pareto cephesi | En doğru model ≠ en verimli model |
| `fig5_memory_vram.png` | Teorik vs ölçülen VRAM | Bellek **aktivasyonlarla** belirleniyor, parametreyle değil |
| `fig6_efficiency_power.png` | (a) TFLOP/s, (b) ortalama güç | MobileNetV2 GPU'yu verimsiz kullanıyor; E ≠ sabit × T |
| `fig7_inference.png` | GPU vs CPU, batch vs tekil | Küçük modeller CPU'da da yeterince hızlı |
| `fig8_time_vs_flops.png` | Teorik FLOP vs ölçülen süre | FLOP model içinde açıklıyor, modeller arası açıklamıyor |
| `fig9_profile_heatmap.png` | Tüm boyutların özeti | Sunum için tek bakışta karşılaştırma |

### Adım 2 — Ham veriye bakın (10 dk)
`results/raw_runs.csv` dosyasını **Excel** ile açın (Veri → Metinden/CSV'den).
- Filtre koyun: `model` = ResNet18, `fraction` = 1 → 5 satır (5 seed) göreceksiniz. Değerler birbirine yakınsa ölçüm güvenilir demektir.
- Özet istatistikler için `results/summary_by_model_fraction.csv` (`_mean` ve `_std` sütunları).

### Adım 3 — Kodu okuyun (her kişi bir dosya, 20-30 dk)
Hoca "kullandığınız algoritmanın çalışma prensibini ve teorik maliyetini açıklayabilmeniz" bekliyor. Önerilen paylaşım:

| Kişi | Okuyacağı dosya | Açıklayabilmesi gereken |
|---|---|---|
| Baran | `run_experiments.py`, `measure.py` | Deney akışı, enerji nasıl ölçülüyor (NVML sayaç, CodeCarbon), neden `cuda.synchronize()` |
| İrem | `models.py` | 5 mimarinin yapısı: conv, depthwise conv, residual bağlantı, BatchNorm |
| Nihat | `complexity.py` + Makale Bölüm III | MAC formülleri, O(E·n·F) eğitim, bellek = 3·4P + B·A·2 |
| Süleyman | `analyze.py` + Makale Bölüm V | log-log eğim (b) nasıl bulunur, Pareto cephesi, Welch t-testi |

Her dosyanın başında İngilizce açıklama (docstring) var; önce onu okuyun.

### Adım 4 — Makaleyi okuyun (30 dk)
Derlenmiş hali hazır: **`BYM 303 Proje Raporu/main.pdf`** (8 sayfa, IEEE formatı). Değişiklik yaparsanız iki yol var:

- Bu bilgisayarda (Tectonic indirildi):
  ```powershell
  cd paper
  & "$env:TEMP\claude\0cfb991d0d7b57d0\29a13725-739c-4e99-9d03-e102be338039\scratchpad\tectonic\tectonic.exe" -X compile main.tex
  ```
  (Bu geçici klasör silinirse Tectonic'i https://tectonic-typesetting.github.io adresinden tekrar indirin.)
- Grupça düzenlemek için **Overleaf** (ücretsiz):
1. https://www.overleaf.com → Yeni Proje → *Upload Project*
2. `paper/` klasörünü ZIP'leyip yükleyin (`main.tex`, `refs.bib`, `numbers.tex`, `tables/`, `figures/` birlikte olmalı)
3. Menüden Compiler = **pdfLaTeX** → *Recompile*. IEEEtran sınıfı Overleaf'te hazır.
4. Overleaf projesini 4 kişiyle paylaşıp birlikte düzenleyebilirsiniz.

⚠️ Makalede **doldurmanız gereken tek yer**: `main.tex` başındaki `[University Name]` (üniversite adınız)
ve şablon isterse e-posta adresleri.

### Adım 5 — Deneyi kendiniz tekrar çalıştırmak isterseniz
```powershell
# Sanal ortam bu bilgisayarda hazır:
$py = "$env:USERPROFILE\.venvs\bym303\Scripts\python.exe"
& $py src\run_experiments.py --quick     # 1 dakikalık test (results_quick/ klasörüne yazar)
& $py src\analyze.py                     # grafikleri ve tabloları yeniden üretir
```
Tam deneyi baştan çalıştırmak için `results/raw_runs.csv`'yi başka yere taşıyıp
`& $py src\run_experiments.py` çalıştırın (~1,5 saat, laptop prize takılı olmalı).
Başka bir bilgisayarda: `README.md` içindeki kurulum adımlarını izleyin.

---

## 4. Sunumda / soru-cevapta sorulabilecek sorular ve cevapları

**S: Enerjiyi nasıl ölçtünüz, tahmin mi gerçek mi?**
C: GPU enerjisi **gerçek ölçüm**: NVIDIA sürücüsünün donanım enerji sayacı (NVML `nvmlDeviceGetTotalEnergyConsumption`, milijoule).
CPU ve RAM enerjisi CodeCarbon **tahmini** (Windows'ta Intel RAPL erişilemiyor). CodeCarbon'un GPU değeri NVML ile
neredeyse birebir aynı çıktı — bu, iki yöntemin birbirini doğruladığını gösteriyor.

**S: E = P·T formülü doğru mu?**
C: Formül doğru ama P sabit değil. Küçük modeller GPU'yu tam kullanamıyor, ortalama güçleri düşük; büyük modeller güç limitine yakın çalışıyor.
Yani enerjiyi sadece süreden çıkaramazsınız (fig6b).

**S: Teorik karmaşıklık neden O(n)?**
C: Her epoch'ta n görüntünün her biri için ileri (F) + geri (~2F) geçiş yapılıyor → 3·F·n·E işlem. E ve F sabitken n'e doğrusal.
Log-log grafikte eğim (b) tüm modellerde 0,95–1,04 çıktı, %95 güven aralıklarının hepsi 1'i içeriyor.
Küçük modellerde (MLP, SmallCNN) süre yine doğrusal ama c sabiti FLOP'u değil adım başına sabit maliyeti (kernel başlatma) izliyor:
ikisi de ≈40 µs/görüntü. Enerjide ise MLP (b = 0,91) ve ResNet-18 (b = 0,96) doğrusalın biraz altında, çünkü uzun koşularda ortalama güç düşüyor.

**S: En önemli 3 bulgunuz ne?**
C: (1) Süre Θ(n), bellek n'den bağımsız — teori doğrulandı. (2) MobileNetV2 ResNet-18'den 6 kat az FLOP'a rağmen 1,5 kat yavaş, 3 kat fazla VRAM, 1,5 kat fazla enerji.
(3) MediumCNN, ResNet-18'in doğruluğunun %98'ini (%90,6 vs %92,2) enerjisinin üçte biriyle veriyor; ResNet'in her ek doğruluk puanı 5,5 kJ.

**S: MobileNetV2 daha az işlem yapıyor, neden daha yavaş?**
C: Depthwise konvolüsyonun **aritmetik yoğunluğu** düşük (byte başına az işlem). GPU hesaplamayı değil bellek bant genişliğini bekliyor
(roofline modeli). Ayrıca genişletme katmanları çok fazla aktivasyon üretiyor → hem VRAM hem zaman artıyor.

**S: Bellek neden veri boyutuna bağlı değil?**
C: Veriyi mini-batch'lerle (B = 128) işliyoruz; bellekte aynı anda sadece bir batch'in aktivasyonları var. Bellek = 3·4P + B·A·2 byte → n'den bağımsız.

**S: Neden 5 tekrar?**
C: GPU sıcaklığı, işletim sistemi arka plan işleri ölçümü etkiler. 5 seed ile mean ± std veriyoruz; std küçük olduğu için farklar anlamlı (Welch t-testi).

**S: Adil karşılaştırmayı nasıl sağladınız?**
C: Aynı seed → aynı veri alt kümesi tüm modeller için; aynı donanım, aynı optimizer, aynı epoch, aynı batch; model sırası her blokta karıştırıldı (ısınma etkisi tek modele binmesin);
ölçüm öncesi ısınma koşusu; veri GPU'da tutuldu (veri yükleme gürültüsü yok).

**S: Neden bfloat16?**
C: Ek deneyde (`bench_precision_layout.py`) aynı modelin fp16 + channels_last ile 8 kata kadar yavaş çalıştığını gördük. Her model için en hızlı ayarı (bf16) kullandık.
Bu da "uygulama tercihleri asimptotik karmaşıklık kadar önemli" bulgusunu destekliyor (Akpınar vd. 2026 makalesiyle uyumlu).

---

## 5. Teslim kontrol listesi

- [ ] `main.tex` içindeki `[University Name]` dolduruldu
- [ ] Overleaf'te derlendi, PDF 4 kişi tarafından okundu
- [ ] AISCI 2027 sitesinden güncel sayfa sınırı kontrol edildi (https://jadara.edu.jo/aisci2027/index.php)
- [ ] Kod ZIP'i hazırlandı (`data/` klasörü HARİÇ — 170 MB, gerek yok) veya GitHub'a yüklendi
- [ ] E-posta: rapor PDF + kod ZIP/Git linki, gövdede 4 kişi "Öğrenci No – Ad Soyad" formatında
- [ ] Sunum provası yapıldı
