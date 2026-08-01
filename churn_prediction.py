"""
Makine Öğrenmesi Ara Ödevi
Türkiye Yapay Zeka Akademisi

Proje Amacı:
Bu projede, müşteri ayrılma tahmini problemi üzerinden temel bir
makine öğrenmesi sınıflandırma akışı uygulanmaktadır.

Proje kapsamında:
- Örnek müşteri veri seti oluşturulması
- Temel veri inceleme
- Eksik değer kontrolü ve doldurma
- Kategorik değişkenlerin sayısal forma dönüştürülmesi
- Sayısal değişkenlerin ölçeklendirilmesi
- Yeni özniteliklerin oluşturulması
- Verinin train, validation ve test kümelerine ayrılması
- Logistic Regression, KNN ve Decision Tree modellerinin eğitilmesi
- Modellerin validation performanslarının karşılaştırılması
- En iyi modelin test verisi üzerinde değerlendirilmesi
adımları gerçekleştirilmektedir.

Kullanılan Kütüphaneler:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Çalıştırma Adımları:
1. Gerekli kütüphaneleri yüklemek için terminalde şu komutu çalıştırın:

   pip install -r requirements.txt

2. Python dosyasını çalıştırın:

   python churn_prediction.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")

RANDOM_STATE = 42

print("=" * 70)
print("MÜŞTERİ AYRILMA TAHMİNİ - MAKİNE ÖĞRENMESİ ARA ÖDEVİ")
print("=" * 70)

# -------------------------------------------------------------------
# 1. ÖRNEK MÜŞTERİ VERİ SETİNİN OLUŞTURULMASI
# -------------------------------------------------------------------

print("\n1. VERİ SETİ OLUŞTURULUYOR")
print("-" * 70)

# Rastgele oluşturulan verilerin her çalıştırmada aynı olması için
# sabit bir başlangıç değeri kullanıyoruz.
rng = np.random.default_rng(RANDOM_STATE)

# Veri setindeki müşteri sayısı
musteri_sayisi = 300

# Temel müşteri bilgileri
yas = rng.integers(18, 71, size=musteri_sayisi)

gelir = rng.normal(
    loc=45000,
    scale=15000,
    size=musteri_sayisi
)

# Negatif gelir oluşmaması için alt sınır belirliyoruz.
gelir = np.clip(gelir, 12000, 120000).round(2)

abonelik_suresi = rng.integers(
    1,
    121,
    size=musteri_sayisi
)

destek_talebi_sayisi = rng.poisson(
    lam=2,
    size=musteri_sayisi
)

sehir = rng.choice(
    ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"],
    size=musteri_sayisi,
    p=[0.30, 0.20, 0.25, 0.15, 0.10]
)

uyelik_tipi = rng.choice(
    ["Temel", "Standart", "Premium"],
    size=musteri_sayisi,
    p=[0.40, 0.35, 0.25]
)

# -------------------------------------------------------------------
# Churn hedef değişkeninin oluşturulması
# -------------------------------------------------------------------

# Müşterinin ayrılma ihtimalini etkileyebilecek bazı koşullar
# kullanılarak churn riski oluşturulmaktadır.

churn_riski = (
    -1.20
    + (destek_talebi_sayisi * 0.35)
    - (abonelik_suresi * 0.015)
    - (gelir * 0.000008)
    + np.where(uyelik_tipi == "Temel", 0.65, 0)
    + np.where(uyelik_tipi == "Premium", -0.35, 0)
    + np.where(yas < 25, 0.25, 0)
)

# Skoru 0 ile 1 arasında bir olasılığa dönüştürüyoruz.
churn_olasiligi = 1 / (1 + np.exp(-churn_riski))

# Olasılığa göre churn değerini oluşturuyoruz.
churn = rng.binomial(
    n=1,
    p=churn_olasiligi
)

# DataFrame oluşturma
df = pd.DataFrame(
    {
        "yas": yas,
        "gelir": gelir,
        "abonelik_suresi": abonelik_suresi,
        "destek_talebi_sayisi": destek_talebi_sayisi,
        "sehir": sehir,
        "uyelik_tipi": uyelik_tipi,
        "churn": churn,
    }
)

print("Veri seti başarıyla oluşturuldu.")

# -------------------------------------------------------------------
# 2. VERİ SETİNE KONTROLLÜ EKSİK DEĞER EKLENMESİ
# -------------------------------------------------------------------

print("\n2. EKSİK DEĞERLER EKLENİYOR")
print("-" * 70)

# Gelir sütunundaki yaklaşık %5 satırı eksik yapıyoruz.
gelir_eksik_indeksleri = rng.choice(
    df.index,
    size=15,
    replace=False
)

df.loc[gelir_eksik_indeksleri, "gelir"] = np.nan

# Şehir sütunundaki bazı değerleri eksik yapıyoruz.
sehir_eksik_indeksleri = rng.choice(
    df.index,
    size=8,
    replace=False
)

df.loc[sehir_eksik_indeksleri, "sehir"] = np.nan

# Üyelik tipi sütunundaki bazı değerleri eksik yapıyoruz.
uyelik_eksik_indeksleri = rng.choice(
    df.index,
    size=6,
    replace=False
)

df.loc[uyelik_eksik_indeksleri, "uyelik_tipi"] = np.nan

print("Kontrollü eksik değerler veri setine eklendi.")

# -------------------------------------------------------------------
# 3. TEMEL VERİ İNCELEMESİ
# -------------------------------------------------------------------

print("\n3. TEMEL VERİ İNCELEMESİ")
print("-" * 70)

print("\nVeri setinin ilk 5 satırı:")
print(df.head())

print("\nVeri setinin satır ve sütun sayısı:")
print(f"Satır sayısı : {df.shape[0]}")
print(f"Sütun sayısı: {df.shape[1]}")

print("\nVeri setindeki sütunlar:")
print(df.columns.tolist())

print("\nSütunların veri tipleri:")
print(df.dtypes)

print("\nSayısal değişkenlerin özet istatistikleri:")
print(df.describe())

print("\nHedef değişken dağılımı:")
print(df["churn"].value_counts())

print("\nHedef değişken yüzdelik dağılımı:")
print(
    df["churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nEksik değer sayıları:")
print(df.isnull().sum())

# Churn sınıflarının açıklamalı olarak gösterilmesi
churn_dagilimi = df["churn"].value_counts().sort_index()

print("\nChurn sınıflarının açıklaması:")

print(
    f"Kalan müşteri sayısı (churn = 0): "
    f"{churn_dagilimi.get(0, 0)}"
)

print(
    f"Ayrılan müşteri sayısı (churn = 1): "
    f"{churn_dagilimi.get(1, 0)}"
)

# -------------------------------------------------------------------
# 4. HEDEF DEĞİŞKEN DAĞILIM GRAFİĞİ
# -------------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="churn"
)

plt.title("Müşteri Ayrılma Dağılımı")
plt.xlabel("Churn Durumu")
plt.ylabel("Müşteri Sayısı")
plt.xticks(
    ticks=[0, 1],
    labels=["Kaldı", "Ayrıldı"]
)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 5. ÖZNİTELİK ÜRETME
# -------------------------------------------------------------------

print("\n5. ÖZNİTELİK ÜRETME")
print("-" * 70)

# Abonelik süresini ay cinsinden yılda ifade ediyoruz.
df["abonelik_yili"] = (
    df["abonelik_suresi"] / 12
).round(2)

# Müşteri en az bir kez destek talebi oluşturduysa 1,
# hiç oluşturmadıysa 0 değerini veriyoruz.
df["destek_talebi_var_mi"] = np.where(
    df["destek_talebi_sayisi"] > 0,
    1,
    0
)

# Gelir değişkenini kategorik gruplara ayırıyoruz.
df["gelir_grubu"] = pd.cut(
    df["gelir"],
    bins=[
        -np.inf,
        30000,
        50000,
        70000,
        np.inf
    ],
    labels=[
        "Düşük",
        "Orta",
        "Yüksek",
        "Çok Yüksek"
    ]
)

print("Yeni öznitelikler başarıyla oluşturuldu.")

print("\nOluşturulan yeni özniteliklerin ilk 10 satırı:")
print(
    df[
        [
            "abonelik_suresi",
            "abonelik_yili",
            "destek_talebi_sayisi",
            "destek_talebi_var_mi",
            "gelir",
            "gelir_grubu",
        ]
    ].head(10)
)

# -------------------------------------------------------------------
# 6. BAĞIMSIZ VE HEDEF DEĞİŞKENLERİN AYRILMASI
# -------------------------------------------------------------------

print("\n6. BAĞIMSIZ VE HEDEF DEĞİŞKENLER AYRILIYOR")
print("-" * 70)

X = df.drop(columns="churn")
y = df["churn"]

print(f"Bağımsız değişkenlerin boyutu: {X.shape}")
print(f"Hedef değişkenin boyutu     : {y.shape}")

print("\nModelde kullanılacak bağımsız değişkenler:")
print(X.columns.tolist())
# -------------------------------------------------------------------
# 7. TRAIN, VALIDATION VE TEST AYRIMI
# -------------------------------------------------------------------

print("\n7. VERİ TRAIN, VALIDATION VE TEST KÜMELERİNE AYRILIYOR")
print("-" * 70)

# İlk aşamada verinin %20'sini test seti olarak ayırıyoruz.
# Geriye kalan %80'lik bölüm geçici train-validation setidir.
X_gecici, X_test, y_gecici, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# İkinci aşamada kalan %80'lik verinin %25'ini validation olarak ayırıyoruz.
# Toplam veri üzerinden:
# %80 x %25 = %20 validation
# %80 x %75 = %60 train
X_train, X_validation, y_train, y_validation = train_test_split(
    X_gecici,
    y_gecici,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_gecici
)

print(f"Train seti boyutu     : {X_train.shape}")
print(f"Validation seti boyutu: {X_validation.shape}")
print(f"Test seti boyutu      : {X_test.shape}")

# Veri kümelerindeki hedef değişken oranlarını kontrol ediyoruz.
print("\nTrain seti churn dağılımı (%):")
print(
    y_train
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nValidation seti churn dağılımı (%):")
print(
    y_validation
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nTest seti churn dağılımı (%):")
print(
    y_test
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

# -------------------------------------------------------------------
# 8. VERİ KÜMELERİNİN KOPYALANMASI
# -------------------------------------------------------------------


# Veri kümelerinde güvenli şekilde değişiklik yapabilmek için kopyalıyoruz.   
X_train = X_train.copy()
X_validation = X_validation.copy()
X_test = X_test.copy()

# -------------------------------------------------------------------
# 9. SAYISAL VE KATEGORİK DEĞİŞKENLERİN BELİRLENMESİ
# -------------------------------------------------------------------

print("\n8. SAYISAL VE KATEGORİK DEĞİŞKENLER BELİRLENİYOR")
print("-" * 70)

sayisal_sutunlar = [
    "yas",
    "gelir",
    "abonelik_suresi",
    "destek_talebi_sayisi",
    "abonelik_yili",
    "destek_talebi_var_mi",
]

kategorik_sutunlar = [
    "sehir",
    "uyelik_tipi",
    "gelir_grubu",
]

print("Sayısal sütunlar:")
print(sayisal_sutunlar)

print("\nKategorik sütunlar:")
print(kategorik_sutunlar)


# -------------------------------------------------------------------
# 10. SAYISAL EKSİK DEĞERLERİN DOLDURULMASI
# -------------------------------------------------------------------

print("\n10. SAYISAL EKSİK DEĞERLER DOLDURULUYOR")
print("-" * 70)

sayisal_imputer = SimpleImputer(strategy="median")

X_train[sayisal_sutunlar] = sayisal_imputer.fit_transform(
    X_train[sayisal_sutunlar]
)

X_validation[sayisal_sutunlar] = sayisal_imputer.transform(
    X_validation[sayisal_sutunlar]
)

X_test[sayisal_sutunlar] = sayisal_imputer.transform(
    X_test[sayisal_sutunlar]
)

print("Sayısal eksik değerler train setinin medyanlarıyla dolduruldu.")

# -------------------------------------------------------------------
# 11. KATEGORİK EKSİK DEĞERLERİN DOLDURULMASI
# -------------------------------------------------------------------

print("\n11. KATEGORİK EKSİK DEĞERLER DOLDURULUYOR")
print("-" * 70)

kategorik_imputer = SimpleImputer(
    strategy="most_frequent"
)

X_train[kategorik_sutunlar] = kategorik_imputer.fit_transform(
    X_train[kategorik_sutunlar]
)

X_validation[kategorik_sutunlar] = kategorik_imputer.transform(
    X_validation[kategorik_sutunlar]
)

X_test[kategorik_sutunlar] = kategorik_imputer.transform(
    X_test[kategorik_sutunlar]
)

print(
    "Kategorik eksik değerler train setindeki "
    "en sık görülen değerlerle dolduruldu."
)

#Eksik değer kontrolü
print("\nEksik değer kontrolü:")

print(
    f"Train setindeki toplam eksik değer: "
    f"{X_train.isnull().sum().sum()}"
)

print(
    f"Validation setindeki toplam eksik değer: "
    f"{X_validation.isnull().sum().sum()}"
)

print(
    f"Test setindeki toplam eksik değer: "
    f"{X_test.isnull().sum().sum()}"
)
# -------------------------------------------------------------------
# 12. ONE-HOT ENCODING
# -------------------------------------------------------------------

print("\n12. KATEGORİK DEĞİŞKENLERE ONE-HOT ENCODING UYGULANIYOR")
print("-" * 70)

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_encoded_array = encoder.fit_transform(
    X_train[kategorik_sutunlar]
)

X_validation_encoded_array = encoder.transform(
    X_validation[kategorik_sutunlar]
)

X_test_encoded_array = encoder.transform(
    X_test[kategorik_sutunlar]
)

encoded_sutunlar = encoder.get_feature_names_out(
    kategorik_sutunlar
)

print("\nOne-Hot Encoding sonucunda oluşan sütunlar:")
print(encoded_sutunlar)

#Encoding sonuçlarını DataFrame’e çevirme

X_train_encoded = pd.DataFrame(
    X_train_encoded_array,
    columns=encoded_sutunlar,
    index=X_train.index
)

X_validation_encoded = pd.DataFrame(
    X_validation_encoded_array,
    columns=encoded_sutunlar,
    index=X_validation.index
)

X_test_encoded = pd.DataFrame(
    X_test_encoded_array,
    columns=encoded_sutunlar,
    index=X_test.index
)

#Eski kategorik sütunları kaldırma

X_train = X_train.drop(
    columns=kategorik_sutunlar
)

X_validation = X_validation.drop(
    columns=kategorik_sutunlar
)

X_test = X_test.drop(
    columns=kategorik_sutunlar
)

#Encoded sütunları ana veri kümelerine ekleme

X_train = pd.concat(
    [X_train, X_train_encoded],
    axis=1
)

X_validation = pd.concat(
    [X_validation, X_validation_encoded],
    axis=1
)

X_test = pd.concat(
    [X_test, X_test_encoded],
    axis=1
)

print("\nOne-Hot Encoding sonrası veri boyutları:")

print(f"Train seti     : {X_train.shape}")
print(f"Validation seti: {X_validation.shape}")
print(f"Test seti      : {X_test.shape}")

print("\nModelde kullanılacak son sütunlar:")
print(X_train.columns.tolist())

assert X_train.shape[1] == X_validation.shape[1]
assert X_train.shape[1] == X_test.shape[1]

print("\nTüm veri kümelerinin sütun sayıları eşittir.")


# -------------------------------------------------------------------
# 13. SAYISAL DEĞİŞKENLERİN ÖLÇEKLENDİRİLMESİ
# -------------------------------------------------------------------

print("\n13. SAYISAL DEĞİŞKENLER ÖLÇEKLENDİRİLİYOR")
print("-" * 70)

scaler = StandardScaler()

X_train[sayisal_sutunlar] = scaler.fit_transform(
    X_train[sayisal_sutunlar]
)

X_validation[sayisal_sutunlar] = scaler.transform(
    X_validation[sayisal_sutunlar]
)

X_test[sayisal_sutunlar] = scaler.transform(
    X_test[sayisal_sutunlar]
)

print("Sayısal değişkenler StandardScaler ile ölçeklendirildi.")

print("\nÖn işleme sonrası train setinin ilk 5 satırı:")
print(X_train.head())

print("\nTrain setindeki veri tipleri:")
print(X_train.dtypes)

sayisal_olmayan_sutunlar = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

if len(sayisal_olmayan_sutunlar) == 0:
    print("\nTüm bağımsız değişkenler sayısal forma dönüştürüldü.")
else:
    print("\nSayısal olmayan sütunlar bulundu:")
    print(sayisal_olmayan_sutunlar)

# -------------------------------------------------------------------
# 14. LOGISTIC REGRESSION MODELİ
# -------------------------------------------------------------------

print("\n14. LOGISTIC REGRESSION MODELİ EĞİTİLİYOR")
print("-" * 70)

logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000
)

logistic_model.fit(
    X_train,
    y_train
)

logistic_validation_tahmin = logistic_model.predict(
    X_validation
)

print("Logistic Regression modeli başarıyla eğitildi.")

logistic_accuracy = accuracy_score(
    y_validation,
    logistic_validation_tahmin
)

logistic_precision = precision_score(
    y_validation,
    logistic_validation_tahmin,
    zero_division=0
)

logistic_recall = recall_score(
    y_validation,
    logistic_validation_tahmin,
    zero_division=0
)

logistic_f1 = f1_score(
    y_validation,
    logistic_validation_tahmin,
    zero_division=0
)

print("\nLogistic Regression Validation Sonuçları:")

print(f"Accuracy : {logistic_accuracy:.4f}")
print(f"Precision: {logistic_precision:.4f}")
print(f"Recall   : {logistic_recall:.4f}")
print(f"F1-score : {logistic_f1:.4f}")

# -------------------------------------------------------------------
# 15. KNN MODELİ
# -------------------------------------------------------------------

print("\n15. KNN MODELİ EĞİTİLİYOR")
print("-" * 70)

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train,
    y_train
)

knn_validation_tahmin = knn_model.predict(
    X_validation
)

print("KNN modeli başarıyla eğitildi.")

knn_accuracy = accuracy_score(
    y_validation,
    knn_validation_tahmin
)

knn_precision = precision_score(
    y_validation,
    knn_validation_tahmin,
    zero_division=0
)

knn_recall = recall_score(
    y_validation,
    knn_validation_tahmin,
    zero_division=0
)

knn_f1 = f1_score(
    y_validation,
    knn_validation_tahmin,
    zero_division=0
)

print("\nKNN Validation Sonuçları:")

print(f"Accuracy : {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall   : {knn_recall:.4f}")
print(f"F1-score : {knn_f1:.4f}")

# -------------------------------------------------------------------
# 16. DECISION TREE MODELİ
# -------------------------------------------------------------------

print("\n16. DECISION TREE MODELİ EĞİTİLİYOR")
print("-" * 70)

decision_tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=RANDOM_STATE
)

decision_tree_model.fit(
    X_train,
    y_train
)

decision_tree_validation_tahmin = decision_tree_model.predict(
    X_validation
)

print("Decision Tree modeli başarıyla eğitildi.")
decision_tree_accuracy = accuracy_score(
    y_validation,
    decision_tree_validation_tahmin
)

decision_tree_precision = precision_score(
    y_validation,
    decision_tree_validation_tahmin,
    zero_division=0
)

decision_tree_recall = recall_score(
    y_validation,
    decision_tree_validation_tahmin,
    zero_division=0
)

decision_tree_f1 = f1_score(
    y_validation,
    decision_tree_validation_tahmin,
    zero_division=0
)

print("\nDecision Tree Validation Sonuçları:")

print(f"Accuracy : {decision_tree_accuracy:.4f}")
print(f"Precision: {decision_tree_precision:.4f}")
print(f"Recall   : {decision_tree_recall:.4f}")
print(f"F1-score : {decision_tree_f1:.4f}")

# -------------------------------------------------------------------
# 17. VALIDATION SONUÇLARININ KARŞILAŞTIRILMASI
# -------------------------------------------------------------------

print("\n17. MODELLERİN VALIDATION SONUÇLARI KARŞILAŞTIRILIYOR")
print("-" * 70)

validation_sonuclari = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "KNN",
            "Decision Tree",
        ],
        "Accuracy": [
            logistic_accuracy,
            knn_accuracy,
            decision_tree_accuracy,
        ],
        "Precision": [
            logistic_precision,
            knn_precision,
            decision_tree_precision,
        ],
        "Recall": [
            logistic_recall,
            knn_recall,
            decision_tree_recall,
        ],
        "F1-score": [
            logistic_f1,
            knn_f1,
            decision_tree_f1,
        ],
    }
)

validation_sonuclari = validation_sonuclari.sort_values(
    by="F1-score",
    ascending=False
).reset_index(drop=True)

print("\nValidation performans karşılaştırması:")
print(validation_sonuclari.round(4).to_string(index=False))