import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(
    path: string
) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def normalize_data(
    data: pd.DataFrame
) -> pd.DataFrame:
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


