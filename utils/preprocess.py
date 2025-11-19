import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def handle_missing(df, method, col):
    if method == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif method == "median":
        df[col] = df[col].fillna(df[col].median())
    elif method == "drop":
        df = df.dropna(subset=[col])
    return df

def scale_data(df, cols, method="standard"):
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df[cols] = scaler.fit_transform(df[cols])
    return df
