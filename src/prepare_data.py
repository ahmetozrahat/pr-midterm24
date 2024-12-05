from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(data, test_size=0.2, random_state=42):

    # Özellikler (X) ve hedef değişken (y)
    X = data.drop(columns=['HighCrime'])
    y = data['HighCrime']

    # Eğitim ve test setine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Veriyi ölçeklendir
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test