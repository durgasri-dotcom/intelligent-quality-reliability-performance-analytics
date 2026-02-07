import pandas as pd
import numpy as np
import random

# -----------------------------
# CONFIGURATION
# -----------------------------
NUM_DEVICES = 5000
import os

# -----------------------------
# SAFE PATH RESOLUTION
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "system_logs.csv")


# -----------------------------
# RANDOM SEED (reproducible)
# -----------------------------
np.random.seed(42)
random.seed(42)

# -----------------------------
# DATA GENERATION
# -----------------------------
data = []

for i in range(1, NUM_DEVICES + 1):
    device_id = f"D{i}"

    uptime_hours = np.random.randint(500, 3000)
    failures = np.random.poisson(lam=uptime_hours / 800)
    avg_latency_ms = np.random.normal(loc=200, scale=50)
    avg_latency_ms = max(20, round(avg_latency_ms, 2))

    error_rate = round(np.random.uniform(0.001, 0.1), 4)

    data.append([
        device_id,
        uptime_hours,
        failures,
        avg_latency_ms,
        error_rate
    ])

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
df = pd.DataFrame(
    data,
    columns=[
        "device_id",
        "uptime_hours",
        "failures",
        "avg_latency_ms",
        "error_rate"
    ]
)

# -----------------------------
# SAVE CSV
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)

print(f"Generated {NUM_DEVICES} rows and saved to {OUTPUT_PATH}")
