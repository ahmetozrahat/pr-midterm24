from sklearn.ensemble import RandomForestClassifier

def train_and_predict(X_train, X_test, y_train, random_state=42):
    # Modeli oluştur
    model = RandomForestClassifier(random_state=random_state)

    # Modeli eğit
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    return model, y_pred