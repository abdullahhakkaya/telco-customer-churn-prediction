import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# Grafiklerde Türkçe karakter desteği için stil ayarı
plt.rcParams["font.family"] = "DejaVu Sans"


def load_and_clean_data(file_path):
    """Veri setini yükler, temel temizlik işlemlerini yapar ve X/y döndürür."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: '{file_path}'. "
            "Lütfen dosyanın doğru klasörde olduğundan emin olun."
        )

    # CSV dosyasını oku
    df = pd.read_csv(file_path)

    # İstenen temel bilgileri ekrana yazdır
    print("\n--- İlk 5 Satır ---")
    print(df.head())

    print("\n--- Veri Seti Boyutu ---")
    print(df.shape)

    print("\n--- Sütun İsimleri ---")
    print(df.columns.tolist())

    print("\n--- Eksik Değer Sayıları ---")
    print(df.isnull().sum())

    # Kimlik sütununu modelleme dışında bırak
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges sütununu sayısal tipe çevir; hatalı değerler NaN olur
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Hedef değişkeni sayısal forma çevir
    if "Churn" not in df.columns:
        raise ValueError("'Churn' sütunu veri setinde bulunamadı.")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Beklenmeyen hedef değerleri varsa kullanıcıyı bilgilendir
    if df["Churn"].isnull().any():
        raise ValueError(
            "'Churn' sütununda Yes/No dışında değerler bulundu. "
            "Lütfen veri setini kontrol edin."
        )

    # Özellikler ve hedefi ayır
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return df, X, y


def prepare_model_pipeline(X, model_name):
    """Veri tiplerine göre ön işleme + model pipeline'ı oluşturur."""
    # Sayısal ve kategorik sütunları otomatik belirle
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Sayısal sütunlar için dönüşüm adımları
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Kategorik sütunlar için dönüşüm adımları
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Tüm dönüşümleri tek yapıda birleştir
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # İstenen modele göre sınıflandırıcıyı seç
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            max_depth=None,
        )
    else:
        raise ValueError(
            f"Geçersiz model adı: '{model_name}'. "
            "Desteklenen modeller: logistic_regression, random_forest"
        )

    # Ön işleme + model adımlarını pipeline'a koy
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test, model_label):
    """Model performansını hesaplar, metrikleri ve eğri verilerini döndürür."""
    # Sınıf tahminleri
    y_pred = model.predict(X_test)

    # Olasılık tahminleri (ROC için)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Temel metrikleri hesapla
    metrics = {
        "Model": model_label,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": auc(*roc_curve(y_test, y_prob)[:2]),
    }

    # Ek raporlar
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Sonuçları ekrana yazdır
    print(f"\n===== {model_label} Sonuçları =====")
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1 Score : {metrics['F1']:.4f}")
    print(f"ROC-AUC  : {metrics['ROC-AUC']:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return metrics, cm, fpr, tpr


def save_confusion_matrix(cm, model_title, output_path):
    """Confusion matrix grafiğini kaydeder."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_title} - Karmaşıklık Matrisi")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_curve(roc_data, output_path):
    """İki modelin ROC eğrisini aynı grafikte kaydeder."""
    plt.figure(figsize=(8, 6))

    for item in roc_data:
        model_name = item["model_name"]
        fpr = item["fpr"]
        tpr = item["tpr"]
        roc_auc = item["roc_auc"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={roc_auc:.4f})")

    # Referans diagonal çizgi
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele Tahmin")

    plt.title("Modellerin ROC Eğrisi Karşılaştırması")
    plt.xlabel("Yanlış Pozitif Oranı")
    plt.ylabel("Doğru Pozitif Oranı")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_feature_importance(model, output_path, top_n=15):
    """Random Forest için en önemli özellikleri görselleştirir."""
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    # One-Hot Encoding sonrası özellik isimlerini al
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Özellik": feature_names,
            "Önem": importances,
        }
    ).sort_values(by="Önem", ascending=False)

    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=top_features,
        x="Önem",
        y="Özellik",
        palette="viridis",
    )
    plt.title(f"Random Forest - En Önemli İlk {top_n} Özellik")
    plt.xlabel("Önem Skoru")
    plt.ylabel("Özellikler")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    """Tüm churn analizi akışını çalıştırır."""
    file_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    try:
        # Veriyi yükle ve temizle
        df, X, y = load_and_clean_data(file_path)

        # Churn dağılım grafiği
        plt.figure(figsize=(6, 5))
        churn_map = {0: "No", 1: "Yes"}
        churn_counts = y.map(churn_map).value_counts()
        sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="Set2")
        plt.title("Churn Dağılımı")
        plt.xlabel("Churn Durumu")
        plt.ylabel("Müşteri Sayısı")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "churn_distribution.png"), dpi=300)
        plt.close()

        # Veriyi eğitim ve test olarak ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        # Logistic Regression modeli
        logistic_pipeline = prepare_model_pipeline(X_train, "logistic_regression")
        logistic_pipeline.fit(X_train, y_train)
        (
            logistic_metrics,
            logistic_cm,
            logistic_fpr,
            logistic_tpr,
        ) = evaluate_model(
            logistic_pipeline,
            X_test,
            y_test,
            "Logistic Regression",
        )

        save_confusion_matrix(
            logistic_cm,
            "Logistic Regression",
            os.path.join(results_dir, "confusion_matrix_logistic_regression.png"),
        )

        # Random Forest modeli
        rf_pipeline = prepare_model_pipeline(X_train, "random_forest")
        rf_pipeline.fit(X_train, y_train)
        rf_metrics, rf_cm, rf_fpr, rf_tpr = evaluate_model(
            rf_pipeline,
            X_test,
            y_test,
            "Random Forest",
        )

        save_confusion_matrix(
            rf_cm,
            "Random Forest",
            os.path.join(results_dir, "confusion_matrix_random_forest.png"),
        )

        # ROC eğrisi karşılaştırması
        roc_data = [
            {
                "model_name": "Logistic Regression",
                "fpr": logistic_fpr,
                "tpr": logistic_tpr,
                "roc_auc": logistic_metrics["ROC-AUC"],
            },
            {
                "model_name": "Random Forest",
                "fpr": rf_fpr,
                "tpr": rf_tpr,
                "roc_auc": rf_metrics["ROC-AUC"],
            },
        ]
        save_roc_curve(
            roc_data,
            os.path.join(results_dir, "roc_curve_comparison.png"),
        )

        # Random Forest özellik önem grafiği
        save_feature_importance(
            rf_pipeline,
            os.path.join(results_dir, "random_forest_feature_importance.png"),
            top_n=15,
        )

        # Model karşılaştırma tablosu
        comparison_df = pd.DataFrame([logistic_metrics, rf_metrics])
        comparison_df = comparison_df[
            ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        ]
        comparison_df.to_csv(
            os.path.join(results_dir, "model_comparison.csv"),
            index=False,
        )

        # Program sonu mesajları
        print("\nAnaliz tamamlandı.")
        print(
            "Model karşılaştırma tablosu "
            "results/model_comparison.csv dosyasına kaydedildi."
        )
        print("Grafikler results klasörüne kaydedildi.")

    except FileNotFoundError as err:
        print(f"Hata: {err}")
        sys.exit(1)
    except ValueError as err:
        print(f"Veri Hatası: {err}")
        sys.exit(1)
    except Exception as err:
        print(f"Beklenmeyen bir hata oluştu: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
