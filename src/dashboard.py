import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Quality, Reliability & Performance Analytics",
    layout="wide"
)

# ==================================================
# BACKGROUND STYLING
# ==================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #eaf4ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# PATH SETUP
# ==================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from data_ingestion import load_data, preprocess
from anomaly_detection import detect_anomalies

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "system_logs.csv")

# ==================================================
# LOAD & PROCESS DATA
# ==================================================
df = load_data(DATA_PATH)
df = preprocess(df)
df = detect_anomalies(df)

# ==================================================
# PCA FOR VISUALIZATION
# ==================================================
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df[["failure_rate", "avg_latency_ms"]])
df["pc1"] = pca_features[:, 0]
df["pc2"] = pca_features[:, 1]

# ==================================================
# TITLE
# ==================================================
st.title("📊 Quality, Reliability & Performance Analytics Platform")

st.caption(
    "A scalable analytics dashboard for monitoring device performance "
    "and detecting reliability anomalies using machine learning."
)

# ==================================================
# EXECUTIVE KPIs
# ==================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Devices Monitored", df["device_id"].nunique())
col2.metric("Average Latency (ms)", round(df["avg_latency_ms"].mean(), 2))
col3.metric("Max Latency (ms)", round(df["avg_latency_ms"].max(), 2))
col4.metric("Anomalies Detected", int((df["anomaly"] == -1).sum()))

st.markdown("---")

# ==================================================
# SYSTEM-WIDE PERFORMANCE (AGGREGATED)
# ==================================================
st.subheader("📈 System-Wide Latency Distribution")

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.hist(df["avg_latency_ms"], bins=40, color="#4c72b0", edgecolor="black")
ax1.set_xlabel("Average Latency (ms)")
ax1.set_ylabel("Number of Devices")
ax1.set_title("Latency Distribution Across Devices")

st.pyplot(fig1)

st.markdown("---")

# ==================================================
# DEVICE-LEVEL DRILLDOWN
# ==================================================
st.subheader("🔍 Device-Level Inspection")

selected_device = st.selectbox(
    "Select a device",
    options=["All"] + sorted(df["device_id"].tolist())
)

if selected_device == "All":
    st.info("Select a specific device to inspect detailed metrics.")
else:
    device_df = df[df["device_id"] == selected_device]
    st.dataframe(device_df, use_container_width=True)

st.markdown("---")

# ==================================================
# ANOMALY DETECTION VISUALIZATION
# ==================================================
st.subheader("🚨 Anomaly Detection (Isolation Forest + PCA)")

normal = df[df["anomaly"] == 1]
anomaly = df[df["anomaly"] == -1]

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(
    normal["pc1"],
    normal["pc2"],
    c="#4c72b0",
    alpha=0.4,
    label="Normal Devices"
)

ax2.scatter(
    anomaly["pc1"],
    anomaly["pc2"],
    c="red",
    s=80,
    label="Anomalous Devices"
)

ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.set_title("ML-Based Anomaly Detection on Device Telemetry")
ax2.legend()

st.pyplot(fig2)
