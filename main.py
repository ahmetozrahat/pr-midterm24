from src.export_data import save_model_and_metrics
from src.evaluate_model import evaluate_model
from src.prepare_data import prepare_data
from src.train_model import train_and_predict
from src.fetch_data import load_data

import matplotlib.pyplot as plt

# Veriyi getir
data = load_data()

# Veriyi hazırla
X_train, X_test, y_train, y_test = prepare_data(data, test_size=0.3, random_state=123)

# Modeli eğit ve tahmin yap
model, y_pred = train_and_predict(X_train, X_test, y_train, random_state=123)

# Modeli değerlendir
sensitivity, specificity = evaluate_model(y_test, y_pred)

# Performans metriklerini görselleştir
metrics = ['Sensitivity', 'Specificity']
values = [sensitivity, specificity]

plt.bar(metrics, values, color=['blue', 'green'])
plt.title("Performans Metrikleri")
plt.ylabel("Değer")
plt.show()

# Modeli ve performans sonuçlarını kaydet
save_model_and_metrics(model, metrics, values, model_filename='random_forest_model.pkl', metrics_filename='performance_metrics.csv')