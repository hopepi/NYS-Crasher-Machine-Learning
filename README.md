# NYS-Crasher-Machine-Learning

## Proje Hakkında
Bu depo, **Trafik Kazası (Crash) Verileri** üzerinde Makine Öğrenimi (Machine Learning) teknikleri kullanılarak analiz ve modelleme yapmayı amaçlayan bir projedir. Üniversite dersi (`FET445 Veri Madenciliği`) kapsamında bir grup çalışması hazırlanmıştır.

Temel hedef, kazaları etkileyen faktörleri incelemek, veriyi görselleştirmek ve kazaların sonuçlarını veya belirli özelliklerini tahmin etmek için makine öğrenimi modelleri geliştirmektir.

## İçindekiler
Bu depo, projenin farklı aşamalarını temsil eden Jupyter Notebook dosyalarından oluşmaktadır:

1.  **Veri Keşfi ve Görselleştirme (`*EDA.ipynb`, `*ÖnVeriGörselleştirme.ipynb`):**
    * Veri setinin ilk incelenmesi ve yapısının anlaşılması.
    * Eksik değer analizi.
    * Temel istatistikler ve değişkenler arası ilişkilerin grafiklerle görselleştirilmesi.
2.  **Veri Ön İşleme (`*Encoding.ipynb`):**
    * Kategorik değişkenlerin makine öğrenimi modellerine uygun hale getirilmesi (One-Hot Encoding, Label Encoding vb.).
    * Özellik mühendisliği (Feature Engineering) adımları.
3.  **Model Uygulaması ve Değerlendirme (`*KNNSonrası.ipynb`):**
    * K-En Yakın Komşu (K-Nearest Neighbors - KNN) gibi çeşitli makine öğrenimi algoritmalarının uygulanması.
    * Model performans metrikleri (doğruluk, kesinlik, geri çağırma, F1 skoru vb.) ile modellerin değerlendirilmesi.

## Dosya Yapısı

| Dosya Adı | İçerik Özeti |
| `FET445..._VeriAnalizi.ipynb` | Veri setinin detaylı analizi ve temizlik adımları. |
| `FET445..._EDA.ipynb` | Keşifçi Veri Analizi (Exploratory Data Analysis). |
| `FET445..._ÖnVeriGörselleştirme.ipynb` | Veri görselleştirme sonuçları. |
| `FET445..._Encoding.ipynb` | Kategorik değişkenlerin kodlanması (ön işleme). |
| `FET445..._KNNSonrası.ipynb` | KNN modelinin eğitimi ve sonuçlarının incelenmesi. |
| Diğer `FET445...ipynb` dosyaları | Farklı model denemeleri veya analiz varyasyonları. |

## Gereksinimler
Bu projeyi yerel ortamınızda çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız vardır:

* Python 3.x
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib / Seaborn
* Scikit-learn

Proje Ekibi:
* Semih ALTUN 
* Efe İNGİN 
* Enes ÇAKIR 
* Hüseyin BALIK 
* Muhammed Emir AYDOĞAN 
* Umut ÖZKAN 

Bu kütüphaneleri genellikle tek bir komutla yükleyebilirsiniz:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter


