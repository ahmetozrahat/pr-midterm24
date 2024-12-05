from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(y_test, y_pred):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Duyarlılık ve özgüllük
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    print(f"Duyarlılık (Sensitivity): {sensitivity:.2f}")
    print(f"Özgüllük (Specificity): {specificity:.2f}")

    # Detaylı sınıflandırma raporu
    report = classification_report(y_test, y_pred)
    print("Sınıflandırma Raporu:\n", report)

    # Performans metriklerini bir sözlükte sakla
    results = {
        "Confusion Matrix": cm,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Classification Report": report
    }

    return sensitivity, specificity