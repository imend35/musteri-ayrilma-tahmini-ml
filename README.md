# Müşteri Ayrılma Tahmini — Makine Öğrenmesi Akışı

Bu proje, **Türkiye Yapay Zeka Akademisi Makine Öğrenmesi Ara Ödevi** kapsamında hazırlanmıştır.

Projede müşteri ayrılma tahmini problemi üzerinden temel bir sınıflandırma akışı uygulanmıştır. Veri ön işleme işlemleri `Pipeline` ve `ColumnTransformer` kullanılmadan, adım adım manuel olarak gerçekleştirilmiştir.

## Projenin Amacı

Projenin amacı, müşterilerin hizmetten ayrılıp ayrılmayacağını tahmin eden temel bir makine öğrenmesi sistemi geliştirmektir.

Proje kapsamında aşağıdaki işlemler uygulanmıştır:

* Örnek müşteri veri seti oluşturma
* Temel veri inceleme
* Eksik değer kontrolü
* Eksik değerleri doldurma
* Öznitelik üretme
* Train, validation ve test ayrımı
* Kategorik değişkenleri sayısal forma dönüştürme
* Sayısal değişkenleri ölçeklendirme
* Birden fazla sınıflandırma modeli eğitme
* Modelleri validation setinde karşılaştırma
* En iyi modeli test setinde değerlendirme
* Confusion matrix ve sınıflandırma metriklerini oluşturma

## Proje Yapısı

```text
musteri-ayrilma-tahmini-manuel/
│
├── churn_prediction.py
├── requirements.txt
├── README.md
│
├── data/
│   └── musteri_churn.csv
│
└── outputs/
    ├── churn_dagilimi.png
    ├── validation_sonuclari.csv
    └── confusion_matrix.png
```

`data` ve `outputs` klasörleri Python kodu çalıştırıldığında otomatik olarak oluşturulur.

## Kullanılan Teknolojiler

* Python
* NumPy
* pandas
* Matplotlib
* Seaborn
* scikit-learn

## Veri Seti

Projede Python kullanılarak 300 satırlık örnek bir müşteri veri seti oluşturulmuştur.

Veri setinde aşağıdaki temel sütunlar bulunmaktadır:

| Sütun                  | Açıklama                               |
| ---------------------- | -------------------------------------- |
| `yas`                  | Müşterinin yaşı                        |
| `gelir`                | Müşterinin gelir bilgisi               |
| `abonelik_suresi`      | Abonelik süresi, ay                    |
| `destek_talebi_sayisi` | Destek birimine yapılan başvuru sayısı |
| `sehir`                | Müşterinin yaşadığı şehir              |
| `uyelik_tipi`          | Temel, Standart veya Premium üyelik    |
| `churn`                | Müşterinin ayrılma durumu              |

Hedef değişken `churn` sütunudur:

* `0`: Müşteri kalır.
* `1`: Müşteri ayrılır.

Eksik değer işlemlerini gösterebilmek amacıyla `gelir`, `sehir` ve `uyelik_tipi` sütunlarına kontrollü olarak bazı eksik değerler eklenmiştir.

## Üretilen Öznitelikler

Modelin mevcut veriden daha anlamlı bilgiler öğrenebilmesi için aşağıdaki öznitelikler oluşturulmuştur:

| Öznitelik              | Açıklama                                                     |
| ---------------------- | ------------------------------------------------------------ |
| `abonelik_yili`        | Abonelik süresinin yıl karşılığı                             |
| `destek_talebi_var_mi` | Müşterinin en az bir destek talebi olup olmadığı             |
| `gelir_grubu`          | Gelirin Düşük, Orta, Yüksek ve Çok Yüksek olarak gruplanması |

## Veri Ayrımı

Veri seti aşağıdaki oranlarla bölünmüştür:

| Veri Kümesi | Oran | Kullanım Amacı                      |
| ----------- | ---: | ----------------------------------- |
| Train       |  %60 | Modellerin eğitilmesi               |
| Validation  |  %20 | Modellerin karşılaştırılması        |
| Test        |  %20 | Seçilen modelin son değerlendirmesi |

Sınıflandırma problemindeki churn oranlarının veri kümelerinde korunması için `stratify` kullanılmıştır.

## Manuel Veri Ön İşleme

Bu projede `Pipeline` ve `ColumnTransformer` kullanılmamıştır. Ön işleme adımları manuel olarak gerçekleştirilmiştir.

### Sayısal Değişkenler

Sayısal sütunlardaki eksik değerler, yalnızca train verisinden öğrenilen medyan değerlerle doldurulmuştur.

Ardından sayısal değişkenlere `StandardScaler` uygulanmıştır.

### Kategorik Değişkenler

Kategorik sütunlardaki eksik değerler, yalnızca train verisinden öğrenilen en sık değerlerle doldurulmuştur.

Kategorik değişkenler `OneHotEncoder` kullanılarak sayısal forma dönüştürülmüştür.

Encoder için `handle_unknown="ignore"` kullanılmıştır. Böylece validation veya test setinde train setinde bulunmayan bir kategoriyle karşılaşıldığında hata oluşması önlenmiştir.

### Veri Sızıntısının Önlenmesi

Ön işleme nesneleri yalnızca train verisi üzerinde `fit` edilmiştir.

Validation ve test verilerinde yalnızca `transform` işlemi uygulanmıştır. Bu yöntemle validation ve test setlerindeki bilgilerin model eğitim sürecine sızması önlenmiştir.

## Kullanılan Modeller

Projede aşağıdaki sınıflandırma modelleri eğitilmiştir:

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree

Decision Tree modeli ödevde bonus model olarak belirtilmiştir.

## Model Karşılaştırması

Modeller validation seti üzerinde aşağıdaki metriklerle karşılaştırılmıştır:

* Accuracy
* Precision
* Recall
* F1-score

Churn tahmininde ayrılacak müşterilerin doğru şekilde tespit edilmesi önemlidir. Bu nedenle model seçiminde yalnızca accuracy değerine bakılmamıştır.

En iyi model öncelikle `F1-score` değerine göre seçilmiştir. Eşitlik durumunda sırasıyla recall ve accuracy değerleri dikkate alınmıştır.

Validation sonuçları aşağıdaki dosyaya kaydedilmektedir:

```text
outputs/validation_sonuclari.csv
```

## En İyi Modelin Seçilmesi

Validation sonuçlarına göre en yüksek F1-score değerine sahip model otomatik olarak seçilmektedir.

Seçim işleminden sonra:

1. Train ve validation verileri birleştirilir.
2. Eksik değer doldurma, One-Hot Encoding ve ölçekleme nesneleri birleşik eğitim verisi üzerinde yeniden öğrenilir.
3. Seçilen model birleşik eğitim verisi üzerinde yeniden eğitilir.
4. Model daha önce görmediği test verisi üzerinde değerlendirilir.

Bu yaklaşım, model seçiminden sonra mevcut eğitim verisinin daha etkin kullanılmasını sağlar.

## Sonuçların Yorumlanması

Oluşturulan veri setinde churn davranışı büyük ölçüde doğrusal ilişkilerle belirlenmiştir.

Destek talebi sayısı arttıkça churn riski artarken abonelik süresi ve gelir arttıkça churn riski azalmaktadır. Temel üyelik churn riskini artırırken Premium üyelik churn riskini azaltmaktadır.

Bu nedenle Logistic Regression modelinin güçlü bir performans göstermesi beklenmektedir. Logistic Regression, değişkenlerdeki artış ve azalışların churn olasılığı üzerindeki doğrusal etkisini başarılı şekilde öğrenebilir.

KNN modeli gözlemler arasındaki uzaklıklara göre tahmin yapmaktadır. One-Hot Encoding sonrasında veri boyutunun artması ve veri setinin görece küçük olması, KNN modelinin performansını olumsuz etkileyebilir.

Decision Tree modeli doğrusal olmayan ilişkileri ve eşik tabanlı kuralları öğrenebilir. Ancak küçük veri setlerinde bazı eğitim örneklerine fazla uyum sağlayarak validation performansında geride kalabilir.

Kod, en iyi modeli önceden sabitlemez. Validation sonuçlarında hangi modelin F1-score değeri daha yüksekse o model otomatik olarak seçilir.

## Test Değerlendirmesi

Seçilen model test seti üzerinde aşağıdaki metriklerle değerlendirilir:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

Confusion matrix grafiği aşağıdaki konuma kaydedilir:

```text
outputs/confusion_matrix.png
```

Confusion matrix şu değerleri gösterir:

* Doğru tahmin edilen kalan müşteriler
* Yanlışlıkla ayrılacak olarak tahmin edilen kalan müşteriler
* Yanlışlıkla kalacak olarak tahmin edilen ayrılan müşteriler
* Doğru tahmin edilen ayrılan müşteriler

## Kurulum

Repository’yi bilgisayarınıza indirin:

```bash
git clone https://github.com/imend35/musteri-ayrilma-tahmini-ml/tree/main
```

Proje klasörüne geçin:

```bash
cd musteri-ayrilma-tahmini-manuel
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

## Çalıştırma

Projeyi aşağıdaki komutla çalıştırın:

```bash
python churn_prediction.py
```

Windows sistemlerde gerekirse şu komut kullanılabilir:

```bash
py churn_prediction.py
```

## Oluşturulan Çıktılar

Kod çalıştırıldığında aşağıdaki dosyalar otomatik olarak oluşturulur:

| Dosya                              | Açıklama                                 |
| ---------------------------------- | ---------------------------------------- |
| `data/musteri_churn.csv`           | Oluşturulan müşteri veri seti            |
| `outputs/churn_dagilimi.png`       | Churn sınıf dağılımı grafiği             |
| `outputs/validation_sonuclari.csv` | Modellerin validation performansları     |
| `outputs/confusion_matrix.png`     | Seçilen modelin confusion matrix grafiği |

## Çalışmanın Kazandırdıkları

Bu proje ile aşağıdaki temel makine öğrenmesi konuları uygulanmıştır:

* Sınıflandırma problemi oluşturma
* Veri seti hazırlama
* Eksik değer yönetimi
* Öznitelik üretme
* Veri sızıntısını önleme
* One-Hot Encoding
* Sayısal ölçekleme
* Train-validation-test ayrımı
* Birden fazla modeli karşılaştırma
* Uygun değerlendirme metriği seçme
* En iyi modeli test verisinde değerlendirme

## Geliştirme Fikirleri

Proje aşağıdaki çalışmalarla geliştirilebilir:

* Daha büyük ve gerçek bir müşteri veri seti kullanma
* Farklı KNN komşu sayılarını karşılaştırma
* Decision Tree için hiperparametre optimizasyonu yapma
* Cross-validation uygulama
* ROC-AUC metriğini ekleme
* Özellik önemlerini inceleme
* Modeli dosyaya kaydetme
* Streamlit veya Flask arayüzü geliştirme

## Not

Bu çalışma eğitim amacıyla hazırlanmıştır. Kullanılan müşteri verileri Python ile yapay olarak oluşturulmuştur ve gerçek kişilere ait değildir.
