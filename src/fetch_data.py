import pandas as pd
import numpy as np

def load_data():
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
    columns = [
        "state", "county", "community", "communityname", "fold", "population", "householdsize",
        "ViolentCrimesPerPop"
    ]

    # Veriyi yükleme
    data = pd.read_csv(data_url, header=None, names=columns)

    # Gereksiz sütunları kaldırma
    data.drop(columns=['state', 'county', 'community', 'communityname'], inplace=True)

    # Eksik verileri temizleme
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Yeni sütun ekleme
    threshold = data['ViolentCrimesPerPop'].median()
    data['HighCrime'] = (data['ViolentCrimesPerPop'] > threshold).astype(int)

    # Kullanılmayan sütunları kaldırma
    data.drop(columns=['ViolentCrimesPerPop'], inplace=True)

    return data