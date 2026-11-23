# Konuşmacı Notları

## Slayt 1

Açılış (Baran): Projemiz, yapay zekâ modellerini yalnızca doğrulukla değil; zaman, bellek ve enerji maliyetiyle birlikte değerlendiriyor. Derin öğrenme konusunu seçtik ve CIFAR-10 üzerinde beş modeli 100 kez eğitip her eğitimi ölçtük.

## Slayt 2

Baran: Föyün istediği dört boyutu birlikte ölçüyoruz. Asıl sorumuz: E = P × T formülündeki P gerçekten sabit mi? Yoksa model değiştikçe ortalama güç de değişiyor mu? Sonuçlarda bunun sabit olmadığını göstereceğiz.

## Slayt 3

Nihat: Hipotezleri deneyden önce, teorik karmaşıklık formüllerinden çıkardık. F görüntü başına çarp-topla işlemi, n eğitim görüntü sayısı, E epoch, P parametre, A aktivasyon, B batch boyutu.

## Slayt 4

Süleyman: Adil karşılaştırma için aynı seed tüm modellere aynı veriyi veriyor. Laptop ısınması tek bir modeli cezalandırmasın diye model sırası her blokta karıştırıldı. Ölçüm öncesi bir ısınma koşusu yapılıyor ve kaydedilmiyor.

## Slayt 5

İrem: Beş model, karmaşıklığı farklı yönlerde değişsin diye seçildi. Konvolüsyon MAC formülü: H·W·C_out·(C_in/g)·k². Depthwise konvolüsyonda g = C_in olduğu için maliyet C_in kat düşer. Bu tablo, bir sonraki slayttaki teorik tahminlerin girdisidir.

## Slayt 6

Nihat: Zaman formülü O(n) veriyor; deneyde log-log grafikte eğimin 1 çıkıp çıkmadığına bakacağız. Bellek formülünde n yok; bu yüzden veri büyüdükçe VRAM değişmemeli. Roofline: GPU’nun hesap gücü ile bellek bant genişliğinden hangisi önce dolarsa hız onunla sınırlanır.

## Slayt 7

Baran: Veri setinin tamamı GPU belleğinde tutuldu, böylece veri yükleme gürültüsü ölçüme karışmıyor. Inference enerjisi için test kümesi en az 3 saniye boyunca tekrar tekrar işlendi, çünkü enerji sayacının çözünürlüğü kısa işlemler için yetersiz.

## Slayt 8

Süleyman: Log-log grafikte doğrunun eğimi b, T = c·n^b üssünü verir. Eğim 1 ise süre doğrusal. 20’şer koşuya en küçük kareler uydurduk ve güven aralığı hesapladık. Eğitim süresinde H1 tamamen doğrulandı; enerjide ise küçük bir sapma var, bunu ortalama güçle açıklıyoruz.

## Slayt 9

Baran: Sol grafik teorik FLOP’u ölçülen süreye bölüyor, yani GPU’nun gerçekte ne kadar verimli kullanıldığını gösteriyor. Roofline modeline göre depthwise konvolüsyon byte başına az işlem yapar ve bant genişliğini bekler. Sağ grafik E = P × T formülündeki P’nin modele göre değiştiğini kanıtlıyor.

## Slayt 10

Nihat: Formüldeki B·A terimi baskın çıktı. Teorik model MobileNetV2’yi biraz eksik tahmin ediyor çünkü BatchNorm girdileri ve cuDNN çalışma alanları da tutuluyor; ResNet-18’i biraz fazla tahmin ediyor çünkü in-place ReLU bellek paylaşıyor. Yine de sıralama ve büyüklük doğru.

## Slayt 11

İrem: Pareto cephesi, daha az enerjiyle daha yüksek doğruluk sunan başka bir modelin olmadığı modellerdir. Enerji bütçesine göre bu kümeden seçim yapılmalı. Ayrıca 5 bin görüntüde MediumCNN, ResNet-18’den çok daha iyi (%67,7 vs %56,7): büyük modeller veriye aç.

## Slayt 12

Süleyman: Tek görüntüde GPU dolmuyor; her katman sabit bir çekirdek başlatma süresi ekliyor. Bu yüzden 58 katmanlı MobileNetV2, işlem sayısı az olmasına rağmen en yavaş. Bu, eğitimde gördüğümüz sabit maliyetin aynısı.

## Slayt 13

Baran: Isı haritasında koyu renk, o sütundaki en yüksek değere yakın demek. Toplam karbon küçük görünse de n, epoch ve hiperparametre deneme sayısıyla doğrusal büyür. CodeCarbon’un GPU tahmini NVML donanım ölçümüyle neredeyse birebir aynı; iki yöntem birbirini doğruluyor.

## Slayt 14

Nihat ve Süleyman: Sonuç olarak teorik Big-O analizi tek bir modelin ölçeklenmesini çok iyi açıklıyor; ama farklı mimarileri karşılaştırırken sabit çarpanlar ve donanım verimliliği belirleyici oluyor. Bu da Akpınar vd.’nin sıralama algoritmalarında gördüğü “uygulama tercihleri önemlidir” bulgusuyla uyumlu.

## Slayt 15

Görev dağılımı öneridir; grup kendi çalışmasına göre güncelleyebilir. Soru-cevap için INCELEME_REHBERI.md dosyasındaki hazır cevaplara bakın.
