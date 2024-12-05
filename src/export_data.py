import joblib
import pandas as pd

def save_model_and_metrics(model, metrics, values, model_filename='model.pkl', metrics_filename='metrics.csv'):
    # Modeli kaydet
    joblib.dump(model, model_filename)
    print(f"Model '{model_filename}' dosyasına kaydedildi.")

    # Performans metriklerini kaydet
    results = pd.DataFrame({'Metrik': metrics, 'Değer': values})
    results.to_csv(metrics_filename, index=False)
    print(f"Performans metrikleri '{metrics_filename}' dosyasına kaydedildi.")