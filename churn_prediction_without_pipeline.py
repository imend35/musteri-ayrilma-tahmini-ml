"""
Makine Öğrenmesi Ara Ödevi
Türkiye Yapay Zeka Akademisi

Proje Amacı:
Bu projede müşteri ayrılma tahmini problemi üzerinden temel bir
makine öğrenmesi sınıflandırma akışı uygulanmaktadır.

Proje kapsamında:
- Python ile örnek müşteri veri seti oluşturulması
- Temel veri inceleme
- Eksik değer kontrolü
- Öznitelik üretme
- Train, validation ve test ayrımı
- Pipeline kullanmadan manuel veri ön işleme
- Sayısal eksik değerlerin medyan ile doldurulması
- Kategorik eksik değerlerin en sık değer ile doldurulması
- Kategorik değişkenlere One-Hot Encoding uygulanması
- Sayısal değişkenlerin StandardScaler ile ölçeklendirilmesi
- Logistic Regression, KNN ve Decision Tree modellerinin eğitilmesi
- Validation sonuçlarının karşılaştırılması
- En iyi modelin test setinde değerlendirilmesi
- Confusion matrix ve sınıflandırma metriklerinin oluşturulması
adımları gerçekleştirilmektedir.

Bu projede sklearn Pipeline ve ColumnTransformer kullanılmamıştır.
Tüm ön işleme adımları manuel olarak uygulanmıştır.

Kullanılan Kütüphaneler:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Çalıştırma Adımları:
1. Gerekli kütüphaneleri yükleyin:

   pip install -r requirements.txt

2. Python dosyasını çalıştırın:

   python churn_prediction.py
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Proje klasörleri
DATA_KLASORU = Path("data")
OUTPUT_KLASORU = Path("outputs")

DATA_KLASORU.mkdir(exist_ok=True)
OUTPUT_KLASORU.mkdir(exist_ok=True)


def baslik_yazdir(baslik: str, numara: int) -> None:
    """Konsol çıktısı için düzenli bölüm başlığı oluşturur."""

    print("\n" + "=" * 78)
    print(f"{numara}. {baslik}")
    print("=" * 78)


def veri_seti_olustur(
    musteri_sayisi: int = 300,
) -> pd.DataFrame:
    """
    Müşteri ayrılma tahmini için örnek veri seti oluşturur.

    Args:
        musteri_sayisi:
            Oluşturulacak müşteri sayısı.

    Returns:
        Müşteri verilerini içeren pandas DataFrame.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    yas = rng.integers(
        low=18,
        high=71,
        size=musteri_sayisi,
    )

    gelir = rng.normal(
        loc=45000,
        scale=15000,
        size=musteri_sayisi,
    )

    gelir = np.clip(
        gelir,
        12000,
        120000,
    ).round(2)

    abonelik_suresi = rng.integers(
        low=1,
        high=121,
        size=musteri_sayisi,
    )

    destek_talebi_sayisi = rng.poisson(
        lam=2,
        size=musteri_sayisi,
    )

    sehir = rng.choice(
        [
            "İstanbul",
            "Ankara",
            "İzmir",
            "Bursa",
            "Antalya",
        ],
        size=musteri_sayisi,
        p=[
            0.30,
            0.20,
            0.25,
            0.15,
            0.10,
        ],
    )

    uyelik_tipi = rng.choice(
        [
            "Temel",
            "Standart",
            "Premium",
        ],
        size=musteri_sayisi,
        p=[
            0.40,
            0.35,
            0.25,
        ],
    )

    # Churn olasılığını etkileyen örnek ilişkiler
    churn_riski = (
        -0.70
        + destek_talebi_sayisi * 0.35
        - abonelik_suresi * 0.012
        - gelir * 0.000006
        + np.where(
            uyelik_tipi == "Temel",
            0.65,
            0,
        )
        + np.where(
            uyelik_tipi == "Premium",
            -0.35,
            0,
        )
        + np.where(
            yas < 25,
            0.25,
            0,
        )
    )

    # Sigmoid fonksiyonu ile risk skorunu olasılığa dönüştürme
    churn_olasiligi = 1 / (
        1 + np.exp(-churn_riski)
    )

    churn = rng.binomial(
        n=1,
        p=churn_olasiligi,
    )

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

    # Eksik değer işlemlerini gösterebilmek için
    # kontrollü eksik değerler ekleniyor.
    gelir_eksik_indeksleri = rng.choice(
        df.index,
        size=15,
        replace=False,
    )

    df.loc[
        gelir_eksik_indeksleri,
        "gelir",
    ] = np.nan

    sehir_eksik_indeksleri = rng.choice(
        df.index,
        size=8,
        replace=False,
    )

    df.loc[
        sehir_eksik_indeksleri,
        "sehir",
    ] = np.nan

    uyelik_eksik_indeksleri = rng.choice(
        df.index,
        size=6,
        replace=False,
    )

    df.loc[
        uyelik_eksik_indeksleri,
        "uyelik_tipi",
    ] = np.nan

    return df


def ozellik_uret(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mevcut sütunlardan yeni öznitelikler üretir.

    Args:
        df:
            Müşteri veri seti.

    Returns:
        Yeni öznitelikler eklenmiş DataFrame.
    """

    df = df.copy()

    # Abonelik süresinin yıl karşılığı
    df["abonelik_yili"] = (
        df["abonelik_suresi"] / 12
    ).round(2)

    # Müşterinin destek talebi olup olmadığını belirten özellik
    df["destek_talebi_var_mi"] = np.where(
        df["destek_talebi_sayisi"] > 0,
        1,
        0,
    )

    # Gelirin kategorik gruplara ayrılması
    df["gelir_grubu"] = pd.cut(
        df["gelir"],
        bins=[
            -np.inf,
            30000,
            50000,
            70000,
            np.inf,
        ],
        labels=[
            "Düşük",
            "Orta",
            "Yüksek",
            "Çok Yüksek",
        ],
    )

    return df


def metrikleri_hesapla(
    gercek_degerler: pd.Series,
    tahminler: np.ndarray,
) -> dict:
    """
    Sınıflandırma performans metriklerini hesaplar.

    Args:
        gercek_degerler:
            Gerçek hedef değerleri.

        tahminler:
            Modelin tahmin ettiği değerler.

    Returns:
        Accuracy, precision, recall ve F1-score değerleri.
    """

    return {
        "Accuracy": accuracy_score(
            gercek_degerler,
            tahminler,
        ),
        "Precision": precision_score(
            gercek_degerler,
            tahminler,
            zero_division=0,
        ),
        "Recall": recall_score(
            gercek_degerler,
            tahminler,
            zero_division=0,
        ),
        "F1-score": f1_score(
            gercek_degerler,
            tahminler,
            zero_division=0,
        ),
    }


def manuel_on_isleme_fit(
    X_egitim: pd.DataFrame,
    sayisal_sutunlar: list,
    kategorik_sutunlar: list,
) -> tuple:
    """
    Ön işleme nesnelerini eğitim verisi üzerinde öğrenir ve
    eğitim verisini dönüştürür.

    Pipeline kullanılmadan şu işlemler gerçekleştirilir:
    1. Sayısal eksik değerleri medyan ile doldurma
    2. Kategorik eksik değerleri en sık değer ile doldurma
    3. One-Hot Encoding
    4. StandardScaler ile ölçekleme

    Args:
        X_egitim:
            Ön işleme kurallarının öğrenileceği eğitim verisi.

        sayisal_sutunlar:
            Sayısal sütun isimleri.

        kategorik_sutunlar:
            Kategorik sütun isimleri.

    Returns:
        Dönüştürülmüş eğitim verisi ve öğrenilen
        ön işleme nesneleri.
    """

    X_egitim = X_egitim.copy()

    # ---------------------------------------------------------------
    # 1. Sayısal eksik değerleri doldurma
    # ---------------------------------------------------------------

    sayisal_imputer = SimpleImputer(
        strategy="median"
    )

    X_egitim[sayisal_sutunlar] = (
        sayisal_imputer.fit_transform(
            X_egitim[sayisal_sutunlar]
        )
    )

    # ---------------------------------------------------------------
    # 2. Kategorik eksik değerleri doldurma
    # ---------------------------------------------------------------

    kategorik_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    X_egitim[kategorik_sutunlar] = (
        kategorik_imputer.fit_transform(
            X_egitim[kategorik_sutunlar]
        )
    )

    # ---------------------------------------------------------------
    # 3. One-Hot Encoding
    # ---------------------------------------------------------------

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    encoded_array = encoder.fit_transform(
        X_egitim[kategorik_sutunlar]
    )

    encoded_sutunlar = (
        encoder.get_feature_names_out(
            kategorik_sutunlar
        )
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_sutunlar,
        index=X_egitim.index,
    )

    X_egitim = X_egitim.drop(
        columns=kategorik_sutunlar
    )

    X_egitim = pd.concat(
        [
            X_egitim,
            encoded_df,
        ],
        axis=1,
    )

    # ---------------------------------------------------------------
    # 4. Sayısal değişkenleri ölçekleme
    # ---------------------------------------------------------------

    scaler = StandardScaler()

    X_egitim[sayisal_sutunlar] = (
        scaler.fit_transform(
            X_egitim[sayisal_sutunlar]
        )
    )

    on_isleme_nesneleri = {
        "sayisal_imputer": sayisal_imputer,
        "kategorik_imputer": kategorik_imputer,
        "encoder": encoder,
        "scaler": scaler,
        "encoded_sutunlar": encoded_sutunlar,
        "sayisal_sutunlar": sayisal_sutunlar,
        "kategorik_sutunlar": kategorik_sutunlar,
        "son_sutunlar": X_egitim.columns.tolist(),
    }

    return X_egitim, on_isleme_nesneleri


def manuel_on_isleme_transform(
    X_veri: pd.DataFrame,
    on_isleme_nesneleri: dict,
) -> pd.DataFrame:
    """
    Eğitim verisinden öğrenilmiş ön işleme kurallarını
    validation veya test verisine uygular.

    Bu fonksiyonda hiçbir nesne yeniden öğrenilmez.
    Yalnızca transform işlemleri uygulanır.

    Args:
        X_veri:
            Dönüştürülecek validation veya test verisi.

        on_isleme_nesneleri:
            Eğitim verisinden öğrenilmiş ön işleme nesneleri.

    Returns:
        Modele hazır hâle getirilmiş DataFrame.
    """

    X_veri = X_veri.copy()

    sayisal_imputer = (
        on_isleme_nesneleri["sayisal_imputer"]
    )

    kategorik_imputer = (
        on_isleme_nesneleri["kategorik_imputer"]
    )

    encoder = on_isleme_nesneleri["encoder"]
    scaler = on_isleme_nesneleri["scaler"]

    sayisal_sutunlar = (
        on_isleme_nesneleri["sayisal_sutunlar"]
    )

    kategorik_sutunlar = (
        on_isleme_nesneleri["kategorik_sutunlar"]
    )

    encoded_sutunlar = (
        on_isleme_nesneleri["encoded_sutunlar"]
    )

    son_sutunlar = (
        on_isleme_nesneleri["son_sutunlar"]
    )

    # Sayısal eksik değerleri eğitim medyanlarıyla doldurma
    X_veri[sayisal_sutunlar] = (
        sayisal_imputer.transform(
            X_veri[sayisal_sutunlar]
        )
    )

    # Kategorik eksik değerleri eğitim verisinden
    # öğrenilen en sık değerlerle doldurma
    X_veri[kategorik_sutunlar] = (
        kategorik_imputer.transform(
            X_veri[kategorik_sutunlar]
        )
    )

    # Kategorik değişkenleri eğitim verisinde öğrenilen
    # kategorilere göre sayısal forma dönüştürme
    encoded_array = encoder.transform(
        X_veri[kategorik_sutunlar]
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_sutunlar,
        index=X_veri.index,
    )

    X_veri = X_veri.drop(
        columns=kategorik_sutunlar
    )

    X_veri = pd.concat(
        [
            X_veri,
            encoded_df,
        ],
        axis=1,
    )

    # Sayısal değişkenleri eğitim verisinden öğrenilen
    # ortalama ve standart sapma değerleriyle ölçekleme
    X_veri[sayisal_sutunlar] = scaler.transform(
        X_veri[sayisal_sutunlar]
    )

    # Sütun sırasını eğitim verisiyle aynı hâle getirme
    X_veri = X_veri.reindex(
        columns=son_sutunlar,
        fill_value=0,
    )

    return X_veri


def main() -> None:
    """Projenin makine öğrenmesi akışını çalıştırır."""

    print("=" * 78)
    print("MÜŞTERİ AYRILMA TAHMİNİ")
    print("MAKİNE ÖĞRENMESİ ARA ÖDEVİ")
    print("PIPELINE KULLANILMADAN MANUEL UYGULAMA")
    print("=" * 78)

    # ----------------------------------------------------------------
    # 1. VERİ SETİNİN OLUŞTURULMASI
    # ----------------------------------------------------------------

    baslik_yazdir(
        "VERİ SETİNİN OLUŞTURULMASI",
        1,
    )

    df = veri_seti_olustur(
        musteri_sayisi=300
    )

    veri_dosyasi = (
        DATA_KLASORU
        / "musteri_churn.csv"
    )

    df.to_csv(
        veri_dosyasi,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Veri seti oluşturuldu: {veri_dosyasi}"
    )

    # ----------------------------------------------------------------
    # 2. TEMEL VERİ İNCELEMESİ
    # ----------------------------------------------------------------

    baslik_yazdir(
        "TEMEL VERİ İNCELEMESİ",
        2,
    )

    print("\nVeri setinin ilk 5 satırı:")
    print(df.head())

    print("\nVeri setinin boyutu:")
    print(f"Satır sayısı : {df.shape[0]}")
    print(f"Sütun sayısı: {df.shape[1]}")

    print("\nSütunların veri tipleri:")
    print(df.dtypes)

    print("\nSayısal sütunların özet istatistikleri:")
    print(
        df.describe().round(2)
    )

    print("\nHedef değişken dağılımı:")
    print(
        df["churn"]
        .value_counts()
        .sort_index()
    )

    print("\nHedef değişken yüzdelik dağılımı:")
    print(
        df["churn"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nEksik değer sayıları:")
    print(
        df.isnull().sum()
    )

    # ----------------------------------------------------------------
    # 3. HEDEF DEĞİŞKEN GRAFİĞİ
    # ----------------------------------------------------------------

    baslik_yazdir(
        "HEDEF DEĞİŞKEN GRAFİĞİ",
        3,
    )

    plt.figure(
        figsize=(7, 5)
    )

    sns.countplot(
        data=df,
        x="churn",
    )

    plt.title(
        "Müşteri Ayrılma Dağılımı"
    )

    plt.xlabel(
        "Churn Durumu"
    )

    plt.ylabel(
        "Müşteri Sayısı"
    )

    plt.xticks(
        ticks=[0, 1],
        labels=[
            "Kaldı",
            "Ayrıldı",
        ],
    )

    plt.tight_layout()

    churn_grafik_dosyasi = (
        OUTPUT_KLASORU
        / "churn_dagilimi.png"
    )

    plt.savefig(
        churn_grafik_dosyasi,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Churn dağılım grafiği kaydedildi: "
        f"{churn_grafik_dosyasi}"
    )

    # ----------------------------------------------------------------
    # 4. ÖZNİTELİK ÜRETME
    # ----------------------------------------------------------------

    baslik_yazdir(
        "ÖZNİTELİK ÜRETME",
        4,
    )

    df = ozellik_uret(df)

    yeni_ozellikler = [
        "abonelik_yili",
        "destek_talebi_var_mi",
        "gelir_grubu",
    ]

    print("Oluşturulan yeni öznitelikler:")
    print(yeni_ozellikler)

    print("\nYeni özniteliklerin ilk 10 satırı:")
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

    # ----------------------------------------------------------------
    # 5. BAĞIMSIZ VE HEDEF DEĞİŞKENLERİN AYRILMASI
    # ----------------------------------------------------------------

    baslik_yazdir(
        "BAĞIMSIZ VE HEDEF DEĞİŞKENLERİN AYRILMASI",
        5,
    )

    X = df.drop(
        columns="churn"
    )

    y = df["churn"]

    print(
        f"Bağımsız değişkenlerin boyutu: {X.shape}"
    )

    print(
        f"Hedef değişkenin boyutu     : {y.shape}"
    )

    print("\nModelde kullanılacak sütunlar:")
    print(
        X.columns.tolist()
    )

    # ----------------------------------------------------------------
    # 6. TRAIN, VALIDATION VE TEST AYRIMI
    # ----------------------------------------------------------------

    baslik_yazdir(
        "TRAIN, VALIDATION VE TEST AYRIMI",
        6,
    )

    # İlk olarak toplam verinin %20'si test seti olarak ayrılır.
    X_gecici, X_test_raw, y_gecici, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Kalan %80'lik verinin %25'i validation olarak ayrılır.
    # Böylece toplam dağılım:
    # Train      = %60
    # Validation = %20
    # Test       = %20
    X_train_raw, X_validation_raw, y_train, y_validation = (
        train_test_split(
            X_gecici,
            y_gecici,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_gecici,
        )
    )

    print(
        f"Train seti     : {X_train_raw.shape}"
    )

    print(
        f"Validation seti: {X_validation_raw.shape}"
    )

    print(
        f"Test seti      : {X_test_raw.shape}"
    )

    print("\nTrain churn oranları (%):")
    print(
        y_train
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nValidation churn oranları (%):")
    print(
        y_validation
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTest churn oranları (%):")
    print(
        y_test
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # ----------------------------------------------------------------
    # 7. SAYISAL VE KATEGORİK SÜTUNLAR
    # ----------------------------------------------------------------

    baslik_yazdir(
        "SAYISAL VE KATEGORİK SÜTUNLAR",
        7,
    )

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

    # ----------------------------------------------------------------
    # 8. MANUEL VERİ ÖN İŞLEME
    # ----------------------------------------------------------------

    baslik_yazdir(
        "PIPELINE KULLANMADAN MANUEL VERİ ÖN İŞLEME",
        8,
    )

    # fit işlemleri yalnızca train verisi üzerinde yapılır.
    X_train, on_isleme_nesneleri = (
        manuel_on_isleme_fit(
            X_train_raw,
            sayisal_sutunlar,
            kategorik_sutunlar,
        )
    )

    # Validation ve test setlerinde yalnızca transform uygulanır.
    X_validation = manuel_on_isleme_transform(
        X_validation_raw,
        on_isleme_nesneleri,
    )

    X_test = manuel_on_isleme_transform(
        X_test_raw,
        on_isleme_nesneleri,
    )

    print(
        "Sayısal eksik değerler train medyanlarıyla dolduruldu."
    )

    print(
        "Kategorik eksik değerler train setindeki "
        "en sık değerlerle dolduruldu."
    )

    print(
        "Kategorik değişkenlere One-Hot Encoding uygulandı."
    )

    print(
        "Sayısal değişkenler StandardScaler ile ölçeklendirildi."
    )

    print("\nÖn işleme sonrası veri boyutları:")
    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_validation.shape}")
    print(f"Test       : {X_test.shape}")

    print("\nEksik değer kontrolü:")
    print(
        "Train toplam eksik değer: "
        f"{X_train.isnull().sum().sum()}"
    )

    print(
        "Validation toplam eksik değer: "
        f"{X_validation.isnull().sum().sum()}"
    )

    print(
        "Test toplam eksik değer: "
        f"{X_test.isnull().sum().sum()}"
    )

    assert X_train.shape[1] == X_validation.shape[1]
    assert X_train.shape[1] == X_test.shape[1]

    print(
        "\nTrain, validation ve test setlerinin "
        "sütun sayıları eşittir."
    )

    # ----------------------------------------------------------------
    # 9. MODELLERİN OLUŞTURULMASI
    # ----------------------------------------------------------------

    baslik_yazdir(
        "MODELLERİN OLUŞTURULMASI",
        9,
    )

    modeller = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
    }

    for model_adi in modeller:
        print(
            f"{model_adi} modeli oluşturuldu."
        )

    # ----------------------------------------------------------------
    # 10. MODELLERİN EĞİTİLMESİ
    # ----------------------------------------------------------------

    baslik_yazdir(
        "MODELLERİN EĞİTİLMESİ VE VALIDATION DEĞERLENDİRMESİ",
        10,
    )

    validation_sonuclari = []

    for model_adi, model in modeller.items():
        print(
            f"\n{model_adi} modeli eğitiliyor..."
        )

        model.fit(
            X_train,
            y_train,
        )

        validation_tahminleri = model.predict(
            X_validation
        )

        metrikler = metrikleri_hesapla(
            y_validation,
            validation_tahminleri,
        )

        validation_sonuclari.append(
            {
                "Model": model_adi,
                **metrikler,
            }
        )

        print(
            f"Accuracy : "
            f"{metrikler['Accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrikler['Precision']:.4f}"
        )

        print(
            f"Recall   : "
            f"{metrikler['Recall']:.4f}"
        )

        print(
            f"F1-score : "
            f"{metrikler['F1-score']:.4f}"
        )

    # ----------------------------------------------------------------
    # 11. VALIDATION SONUÇLARININ KARŞILAŞTIRILMASI
    # ----------------------------------------------------------------

    baslik_yazdir(
        "VALIDATION SONUÇLARININ KARŞILAŞTIRILMASI",
        11,
    )

    validation_df = pd.DataFrame(
        validation_sonuclari
    )

    validation_df = validation_df.sort_values(
        by=[
            "F1-score",
            "Recall",
            "Accuracy",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print(
        validation_df
        .round(4)
        .to_string(index=False)
    )

    validation_dosyasi = (
        OUTPUT_KLASORU
        / "validation_sonuclari.csv"
    )

    validation_df.to_csv(
        validation_dosyasi,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nValidation sonuçları kaydedildi: "
        f"{validation_dosyasi}"
    )

    # ----------------------------------------------------------------
    # 12. EN İYİ MODELİN SEÇİLMESİ
    # ----------------------------------------------------------------

    baslik_yazdir(
        "EN İYİ MODELİN SEÇİLMESİ",
        12,
    )

    en_iyi_model_adi = (
        validation_df.loc[0, "Model"]
    )

    en_iyi_validation_f1 = (
        validation_df.loc[0, "F1-score"]
    )

    en_iyi_validation_recall = (
        validation_df.loc[0, "Recall"]
    )

    en_iyi_validation_accuracy = (
        validation_df.loc[0, "Accuracy"]
    )

    print(
        "Modeller öncelikle F1-score değerine göre "
        "karşılaştırılmıştır."
    )

    print(
        "Eşitlik durumunda recall ve accuracy değerleri "
        "dikkate alınmıştır."
    )

    print(
        f"\nSeçilen model: {en_iyi_model_adi}"
    )

    print(
        f"Validation F1-score : "
        f"{en_iyi_validation_f1:.4f}"
    )

    print(
        f"Validation Recall   : "
        f"{en_iyi_validation_recall:.4f}"
    )

    print(
        f"Validation Accuracy : "
        f"{en_iyi_validation_accuracy:.4f}"
    )

    # ----------------------------------------------------------------
    # 13. TRAIN VE VALIDATION VERİLERİNİ BİRLEŞTİRME
    # ----------------------------------------------------------------

    baslik_yazdir(
        "SEÇİLEN MODELİN YENİDEN EĞİTİLMESİ",
        13,
    )

    # Model seçildikten sonra train ve validation ham verileri
    # birleştirilir.
    X_train_validation_raw = pd.concat(
        [
            X_train_raw,
            X_validation_raw,
        ],
        axis=0,
    )

    y_train_validation = pd.concat(
        [
            y_train,
            y_validation,
        ],
        axis=0,
    )

    # Ön işleme kuralları birleşik eğitim verisi üzerinde
    # yeniden öğrenilir.
    X_train_validation, final_on_isleme_nesneleri = (
        manuel_on_isleme_fit(
            X_train_validation_raw,
            sayisal_sutunlar,
            kategorik_sutunlar,
        )
    )

    # Test verisine birleşik eğitim verisinden öğrenilen
    # ön işleme kuralları uygulanır.
    X_test_final = manuel_on_isleme_transform(
        X_test_raw,
        final_on_isleme_nesneleri,
    )

    if en_iyi_model_adi == "Logistic Regression":
        en_iyi_model = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
        )

    elif en_iyi_model_adi == "KNN":
        en_iyi_model = KNeighborsClassifier(
            n_neighbors=5,
        )

    else:
        en_iyi_model = DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        )

    en_iyi_model.fit(
        X_train_validation,
        y_train_validation,
    )

    print(
        f"{en_iyi_model_adi} modeli train ve validation "
        "verileri birleştirilerek yeniden eğitildi."
    )

    # ----------------------------------------------------------------
    # 14. TEST SETİ DEĞERLENDİRMESİ
    # ----------------------------------------------------------------

    baslik_yazdir(
        "TEST SETİ DEĞERLENDİRMESİ",
        14,
    )

    test_tahminleri = en_iyi_model.predict(
        X_test_final
    )

    test_metrikleri = metrikleri_hesapla(
        y_test,
        test_tahminleri,
    )

    print(
        f"Seçilen model: {en_iyi_model_adi}"
    )

    print("\nTest sonuçları:")

    print(
        f"Accuracy : "
        f"{test_metrikleri['Accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrikleri['Precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{test_metrikleri['Recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{test_metrikleri['F1-score']:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            test_tahminleri,
            target_names=[
                "Kaldı",
                "Ayrıldı",
            ],
            zero_division=0,
        )
    )

    # ----------------------------------------------------------------
    # 15. CONFUSION MATRIX
    # ----------------------------------------------------------------

    baslik_yazdir(
        "CONFUSION MATRIX",
        15,
    )

    confusion_matrix_degeri = confusion_matrix(
        y_test,
        test_tahminleri,
    )

    print(confusion_matrix_degeri)

    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        confusion_matrix_degeri,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=[
            "Kaldı",
            "Ayrıldı",
        ],
        yticklabels=[
            "Kaldı",
            "Ayrıldı",
        ],
    )

    plt.title(
        f"Confusion Matrix - {en_iyi_model_adi}"
    )

    plt.xlabel(
        "Tahmin Edilen Değer"
    )

    plt.ylabel(
        "Gerçek Değer"
    )

    plt.tight_layout()

    confusion_matrix_dosyasi = (
        OUTPUT_KLASORU
        / "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_dosyasi,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Confusion matrix grafiği kaydedildi: "
        f"{confusion_matrix_dosyasi}"
    )

    # ----------------------------------------------------------------
    # 16. KARŞILAŞTIRMALI SONUÇ YORUMU
    # ----------------------------------------------------------------

    baslik_yazdir(
        "KARŞILAŞTIRMALI SONUÇ YORUMU",
        16,
    )

    print(
        "Validation sonuçlarına göre modellerin "
        "performans sıralaması:"
    )

    for sira, satir in validation_df.iterrows():
        print(
            f"{sira + 1}. {satir['Model']} | "
            f"Accuracy: {satir['Accuracy']:.4f} | "
            f"Precision: {satir['Precision']:.4f} | "
            f"Recall: {satir['Recall']:.4f} | "
            f"F1-score: {satir['F1-score']:.4f}"
        )

    print(
        f"\nEn iyi model {en_iyi_model_adi} olarak seçilmiştir."
    )

    print(
        "Model seçiminde yalnızca accuracy değerine bakılmamış; "
        "precision ve recall arasındaki dengeyi gösteren "
        "F1-score temel alınmıştır."
    )

    if en_iyi_model_adi == "Logistic Regression":
        print(
            "\nLogistic Regression modelinin daha başarılı "
            "olmasının olası nedeni, oluşturulan churn "
            "ilişkilerinin büyük ölçüde doğrusal olmasıdır. "
            "Destek talebi sayısı arttıkça churn riski artarken, "
            "abonelik süresi ve gelir arttıkça churn riski "
            "azalmaktadır. Logistic Regression bu tür doğrusal "
            "ilişkileri başarılı şekilde modelleyebilir."
        )

        print(
            "KNN modeli gözlemler arasındaki uzaklıklara dayandığı "
            "için küçük veri setlerinde ve çok sayıda One-Hot "
            "Encoding sütunu bulunduğunda daha düşük performans "
            "gösterebilir."
        )

        print(
            "Decision Tree modeli doğrusal olmayan ilişkileri "
            "öğrenebilse de küçük veri setinde bazı kurallara "
            "fazla uyum sağlayarak genelleme performansını "
            "düşürebilir."
        )

    elif en_iyi_model_adi == "KNN":
        print(
            "\nKNN modelinin daha başarılı olmasının olası nedeni, "
            "benzer özelliklere sahip müşterilerin benzer ayrılma "
            "davranışları göstermesidir. Sayısal değişkenlerin "
            "ölçeklendirilmesi de uzaklık tabanlı KNN modelinin "
            "daha sağlıklı çalışmasını sağlamıştır."
        )

        print(
            "Logistic Regression doğrusal bir karar sınırı "
            "oluşturduğu için veri içerisindeki yerel ve doğrusal "
            "olmayan örüntüleri KNN kadar iyi yakalayamamış olabilir."
        )

        print(
            "Decision Tree ise küçük eğitim setinde bazı dallara "
            "fazla uyum sağlayarak validation performansında "
            "geride kalmış olabilir."
        )

    else:
        print(
            "\nDecision Tree modelinin daha başarılı olmasının "
            "olası nedeni, müşteri davranışındaki doğrusal olmayan "
            "ilişkileri ve eşik tabanlı kuralları yakalayabilmesidir."
        )

        print(
            "Örneğin destek talebi sayısının belirli bir seviyenin "
            "üzerine çıkması veya abonelik süresinin belirli bir "
            "seviyenin altında kalması churn riskini farklı "
            "şekillerde etkileyebilir."
        )

        print(
            "Logistic Regression doğrusal ilişkilere odaklandığı, "
            "KNN ise yüksek boyutlu One-Hot Encoding verisinde "
            "uzaklık hesaplarından etkilendiği için daha düşük "
            "performans göstermiş olabilir."
        )

    print(
        f"\nSeçilen modelin test setindeki F1-score değeri "
        f"{test_metrikleri['F1-score']:.4f} olarak "
        "hesaplanmıştır."
    )

    print(
        "Test sonucu, modelin daha önce görmediği müşteriler "
        "üzerindeki genelleme başarısını göstermektedir."
    )

    print(
        "\nProje başarıyla tamamlandı."
    )


if __name__ == "__main__":
    main()