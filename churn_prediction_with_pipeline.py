"""
Makine Öğrenmesi Ara Ödevi
Türkiye Yapay Zeka Akademisi

Proje Amacı:
Bu projede müşteri ayrılma tahmini problemi üzerinden temel bir
makine öğrenmesi sınıflandırma akışı uygulanmaktadır.

Proje kapsamında:
- Örnek müşteri veri seti oluşturulması
- Temel veri inceleme
- Eksik değer kontrolü
- Öznitelik üretme
- Train, validation ve test ayrımı
- Pipeline ve ColumnTransformer kullanılarak veri ön işleme
- Logistic Regression, KNN ve Decision Tree modellerinin eğitilmesi
- Validation sonuçlarının karşılaştırılması
- En iyi modelin test setinde değerlendirilmesi
- Confusion matrix ve sınıflandırma metriklerinin oluşturulması
adımları gerçekleştirilmektedir.

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

   python churn_prediction_with_pipeline.py
"""

from pathlib import Path
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

# Proje klasörleri
DATA_KLASORU = Path("data")
OUTPUT_KLASORU = Path("outputs")

DATA_KLASORU.mkdir(exist_ok=True)
OUTPUT_KLASORU.mkdir(exist_ok=True)


def baslik_yazdir(baslik: str, numara: int) -> None:
    """Konsol çıktılarında düzenli bölüm başlığı oluşturur."""
    print("\n" + "=" * 75)
    print(f"{numara}. {baslik}")
    print("=" * 75)


def veri_seti_olustur(
    musteri_sayisi: int = 300,
) -> pd.DataFrame:
    """
    Müşteri ayrılma tahmini için örnek veri seti oluşturur.

    Args:
        musteri_sayisi: Oluşturulacak müşteri sayısı.

    Returns:
        Oluşturulan müşteri verilerini içeren pandas DataFrame.
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
        + np.where(uyelik_tipi == "Temel", 0.65, 0)
        + np.where(uyelik_tipi == "Premium", -0.35, 0)
        + np.where(yas < 25, 0.25, 0)
    )

    # Sigmoid fonksiyonuyla skorları olasılığa dönüştürme
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

    # Kontrollü eksik değerler ekleniyor.
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


def ozellik_uret(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mevcut sütunlardan yeni ve anlamlı öznitelikler üretir.

    Args:
        df: Müşteri veri seti.

    Returns:
        Yeni öznitelikler eklenmiş DataFrame.
    """

    df = df.copy()

    # Abonelik süresinin yıl karşılığı
    df["abonelik_yili"] = (
        df["abonelik_suresi"] / 12
    ).round(2)

    # Destek talebi olup olmadığını belirten ikili değişken
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
        gercek_degerler: Gerçek hedef değerleri.
        tahminler: Model tahminleri.

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


def main() -> None:
    """Projenin temel makine öğrenmesi akışını çalıştırır."""

    print("=" * 75)
    print("MÜŞTERİ AYRILMA TAHMİNİ")
    print("MAKİNE ÖĞRENMESİ ARA ÖDEVİ")
    print("=" * 75)

    # ------------------------------------------------------------------
    # 1. VERİ SETİNİN OLUŞTURULMASI
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 2. TEMEL VERİ İNCELEMESİ
    # ------------------------------------------------------------------

    baslik_yazdir(
        "TEMEL VERİ İNCELEMESİ",
        2,
    )

    print("\nVeri setinin ilk 5 satırı:")
    print(df.head())

    print("\nVeri setinin boyutu:")
    print(f"Satır sayısı : {df.shape[0]}")
    print(f"Sütun sayısı: {df.shape[1]}")

    print("\nSütun veri tipleri:")
    print(df.dtypes)

    print("\nSayısal sütunların özet istatistikleri:")
    print(df.describe().round(2))

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
    print(df.isnull().sum())

    # ------------------------------------------------------------------
    # 3. HEDEF DEĞİŞKEN GRAFİĞİ
    # ------------------------------------------------------------------

    baslik_yazdir(
        "HEDEF DEĞİŞKEN GRAFİĞİ",
        3,
    )

    plt.figure(figsize=(7, 5))

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
        f"Churn dağılım grafiği kaydedildi: "
        f"{churn_grafik_dosyasi}"
    )

    # ------------------------------------------------------------------
    # 4. ÖZNİTELİK ÜRETME
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 5. X VE y AYRIMI
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 6. TRAIN, VALIDATION VE TEST AYRIMI
    # ------------------------------------------------------------------

    baslik_yazdir(
        "TRAIN, VALIDATION VE TEST AYRIMI",
        6,
    )

    # İlk olarak test seti ayrılıyor.
    X_gecici, X_test, y_gecici, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # Kalan veriden validation seti ayrılıyor.
    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_gecici,
            y_gecici,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_gecici,
        )
    )

    print(
        f"Train seti     : {X_train.shape}"
    )
    print(
        f"Validation seti: {X_validation.shape}"
    )
    print(
        f"Test seti      : {X_test.shape}"
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

    # ------------------------------------------------------------------
    # 7. SAYISAL VE KATEGORİK SÜTUNLAR
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 8. ÖN İŞLEME PIPELINE'LARI
    # ------------------------------------------------------------------

    baslik_yazdir(
        "ÖN İŞLEME PIPELINE'LARININ OLUŞTURULMASI",
        8,
    )

    sayisal_pipeline = Pipeline(
        steps=[
            (
                "eksik_deger_doldurma",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "olceklendirme",
                StandardScaler(),
            ),
        ]
    )

    kategorik_pipeline = Pipeline(
        steps=[
            (
                "eksik_deger_doldurma",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "one_hot_encoding",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    on_isleyici = ColumnTransformer(
        transformers=[
            (
                "sayisal_islemler",
                sayisal_pipeline,
                sayisal_sutunlar,
            ),
            (
                "kategorik_islemler",
                kategorik_pipeline,
                kategorik_sutunlar,
            ),
        ]
    )

    print(
        "Sayısal ve kategorik ön işleme "
        "pipeline'ları oluşturuldu."
    )

    # ------------------------------------------------------------------
    # 9. MODEL PIPELINE'LARI
    # ------------------------------------------------------------------

    baslik_yazdir(
        "MODEL PIPELINE'LARININ OLUŞTURULMASI",
        9,
    )

    modeller = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
    }

    model_pipeline_listesi = {}

    for model_adi, model in modeller.items():
        model_pipeline_listesi[model_adi] = Pipeline(
            steps=[
                (
                    "on_isleme",
                    on_isleyici,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        print(
            f"{model_adi} pipeline'ı oluşturuldu."
        )

    # ------------------------------------------------------------------
    # 10. MODELLERİN EĞİTİLMESİ
    # ------------------------------------------------------------------

    baslik_yazdir(
        "MODELLERİN EĞİTİLMESİ VE VALIDATION DEĞERLENDİRMESİ",
        10,
    )

    validation_sonuclari = []

    for model_adi, model_pipeline in (
        model_pipeline_listesi.items()
    ):
        print(
            f"\n{model_adi} modeli eğitiliyor..."
        )

        model_pipeline.fit(
            X_train,
            y_train,
        )

        validation_tahminleri = (
            model_pipeline.predict(
                X_validation
            )
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

    # ------------------------------------------------------------------
    # 11. VALIDATION SONUÇLARININ KARŞILAŞTIRILMASI
    # ------------------------------------------------------------------

    baslik_yazdir(
        "VALIDATION SONUÇLARININ KARŞILAŞTIRILMASI",
        11,
    )

    validation_df = pd.DataFrame(
        validation_sonuclari
    )

    validation_df = validation_df.sort_values(
        by="F1-score",
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
        f"\nValidation sonuçları kaydedildi: "
        f"{validation_dosyasi}"
    )

    # ------------------------------------------------------------------
    # 12. EN İYİ MODELİN SEÇİLMESİ
    # ------------------------------------------------------------------

    baslik_yazdir(
        "EN İYİ MODELİN SEÇİLMESİ",
        12,
    )

    en_iyi_model_adi = (
        validation_df.loc[0, "Model"]
    )

    en_iyi_model_pipeline = (
        model_pipeline_listesi[
            en_iyi_model_adi
        ]
    )

    print(
        "F1-score değerine göre seçilen model:"
    )
    print(en_iyi_model_adi)

    # ------------------------------------------------------------------
    # 13. TRAIN VE VALIDATION VERİLERİNİN BİRLEŞTİRİLMESİ
    # ------------------------------------------------------------------

    baslik_yazdir(
        "SEÇİLEN MODELİN YENİDEN EĞİTİLMESİ",
        13,
    )

    X_train_validation = pd.concat(
        [
            X_train,
            X_validation,
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

    # Model seçimi yapıldıktan sonra train ve validation verileri
    # birleştirilerek seçilen model yeniden eğitiliyor.
    en_iyi_model_pipeline.fit(
        X_train_validation,
        y_train_validation,
    )

    print(
        f"{en_iyi_model_adi} modeli train ve "
        "validation verileriyle yeniden eğitildi."
    )

    # ------------------------------------------------------------------
    # 14. TEST SETİ DEĞERLENDİRMESİ
    # ------------------------------------------------------------------

    baslik_yazdir(
        "TEST SETİ DEĞERLENDİRMESİ",
        14,
    )

    test_tahminleri = (
        en_iyi_model_pipeline.predict(
            X_test
        )
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

    # ------------------------------------------------------------------
    # 15. CONFUSION MATRIX
    # ------------------------------------------------------------------

    baslik_yazdir(
        "CONFUSION MATRIX",
        15,
    )

    confusion_matrix_degeri = confusion_matrix(
        y_test,
        test_tahminleri,
    )

    print(confusion_matrix_degeri)

    plt.figure(figsize=(7, 5))

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

    # ------------------------------------------------------------------
    # 16. KISA SONUÇ YORUMU
    # ------------------------------------------------------------------

    baslik_yazdir(
        "KISA SONUÇ YORUMU",
        16,
    )

    print(
        f"Validation sonuçlarına göre en iyi model "
        f"{en_iyi_model_adi} olmuştur."
    )

    print(
        f"Modelin test setindeki F1-score değeri "
        f"{test_metrikleri['F1-score']:.4f} olarak "
        "hesaplanmıştır."
    )

    if en_iyi_model_adi == "Logistic Regression":
        print(
            "Logistic Regression modelinin daha iyi sonuç "
            "vermesinin nedeni, oluşturulan veri setindeki "
            "churn ilişkilerinin büyük ölçüde doğrusal "
            "olması olabilir."
        )

    elif en_iyi_model_adi == "KNN":
        print(
            "KNN modelinin daha iyi sonuç vermesinin nedeni, "
            "benzer özelliklere sahip müşterilerin benzer "
            "ayrılma davranışları göstermesi olabilir."
        )

    else:
        print(
            "Decision Tree modelinin daha iyi sonuç vermesinin "
            "nedeni, müşteri davranışındaki doğrusal olmayan "
            "kuralları ve değişkenler arasındaki eşik "
            "ilişkilerini yakalayabilmesi olabilir."
        )

    print(
        "\nProje başarıyla tamamlandı."
    )


if __name__ == "__main__":
    main()