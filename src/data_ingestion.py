import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df["failure_rate"] = df["failures"] / df["uptime_hours"].replace(0, 1)
    return df
