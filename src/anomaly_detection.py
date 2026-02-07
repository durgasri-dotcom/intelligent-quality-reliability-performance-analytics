from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    features = ["failure_rate", "avg_latency_ms"]

    # Ensure required columns exist
    if not all(col in df.columns for col in features):
        raise ValueError("Required columns for anomaly detection are missing")

    model = IsolationForest(contamination=0.25, random_state=42)
    df["anomaly"] = model.fit_predict(df[features])

    return df
