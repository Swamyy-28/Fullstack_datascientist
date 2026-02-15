# ==============================================================
# MALL CUSTOMER SEGMENTATION DASHBOARD
# Hierarchical Clustering (Agglomerative)
# FULL SAFE VERSION (NO CLUSTER KEY ERROR)
# ==============================================================

# ==============================================================
# IMPORT LIBRARIES
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# ==============================================================
# PAGE CONFIGURATION
# ==============================================================

st.set_page_config(
    page_title="Mall Customer Hierarchical Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# CUSTOM CSS
# ==============================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
}

h1, h2, h3 {
    color: white;
}

.prediction-box {
    background: linear-gradient(135deg, #1c92d2 0%, #f2fcfe 100%);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    font-weight: bold;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
}

.footer-style {
    text-align:center;
    color:white;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# ==============================================================
# LOAD FUNCTIONS
# ==============================================================

@st.cache_data
def load_original_data():
    return pd.read_csv("hierarchical_clustering/Mall_Customers (3).csv")

@st.cache_data
def load_clustered_data():
    return pd.read_csv("hierarchical_clustering/Mall_customer_clustered.csv")

@st.cache_resource
def load_model():
    return joblib.load("hierarchical_clustering/hierarchical_model.pkl")

# ==============================================================
# LOAD DATA
# ==============================================================

df = load_original_data()
clustered_df = load_clustered_data()
model = load_model()

# ==============================================================
# ENSURE CLUSTER COLUMN EXISTS (KEYERROR FIX)
# ==============================================================

clustered_df.columns = clustered_df.columns.str.strip()

if "Cluster" not in clustered_df.columns:
    possible_cols = [col for col in clustered_df.columns if col.lower() in ["cluster", "clusters", "label", "labels"]]
    
    if possible_cols:
        clustered_df.rename(columns={possible_cols[0]: "Cluster"}, inplace=True)
    else:
        # If no cluster column exists → generate using model
        features_temp = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        scaler_temp = StandardScaler()
        scaled_temp = scaler_temp.fit_transform(features_temp)
        clustered_df = df.copy()
        clustered_df["Cluster"] = model.fit_predict(scaled_temp)

# ==============================================================
# TITLE
# ==============================================================

st.markdown("""
<h1 style='text-align:center;'>🛍️ Mall Customer Segmentation Dashboard</h1>
<p style='text-align:center;color:white;'>
Hierarchical Clustering using Agglomerative Method
</p>
""", unsafe_allow_html=True)

# ==============================================================
# FEATURE PREPARATION
# ==============================================================

features = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# ==============================================================
# SIDEBAR OPTIONS
# ==============================================================

st.sidebar.header("⚙️ Controls")

show_3d = st.sidebar.checkbox("Show 3D Visualization", True)
show_pie = st.sidebar.checkbox("Show Pie Chart", True)
show_dendrogram = st.sidebar.checkbox("Show Dendrogram", True)
show_table = st.sidebar.checkbox("Show Data Table", True)

# ==============================================================
# METRICS
# ==============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(df))
col2.metric("Average Age", f"{df['Age'].mean():.2f}")
col3.metric("Average Income", f"{df['Annual Income (k$)'].mean():.2f}")
col4.metric("Average Spending", f"{df['Spending Score (1-100)'].mean():.2f}")

st.markdown("---")

# ==============================================================
# 3D VISUALIZATION
# ==============================================================

if show_3d:
    st.markdown("## 📊 3D Cluster Visualization")

    fig = px.scatter_3d(
        clustered_df,
        x="Age",
        y="Annual Income (k$)",
        z="Spending Score (1-100)",
        color=clustered_df["Cluster"].astype(str),
        title="Hierarchical Cluster Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# PIE CHART
# ==============================================================

if show_pie:
    st.markdown("## 🥧 Cluster Distribution")

    counts = clustered_df["Cluster"].value_counts().sort_index()

    pie = go.Figure(
        data=[go.Pie(
            labels=[f"Cluster {i}" for i in counts.index],
            values=counts.values,
            hole=0.4
        )]
    )

    st.plotly_chart(pie, use_container_width=True)

# ==============================================================
# DENDROGRAM
# ==============================================================

if show_dendrogram:
    st.markdown("## 🌳 Dendrogram")

    linked = linkage(scaled_features, method='ward')

    fig2, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Distance")

    st.pyplot(fig2)

# ==============================================================
# PREDICTION SECTION
# ==============================================================

st.markdown("---")
st.markdown("## 🎯 Predict New Customer Cluster")

colA, colB, colC = st.columns(3)

with colA:
    age = st.slider("Age", 18, 70, 30)

with colB:
    income = st.slider("Annual Income (k$)", 15, 140, 60)

with colC:
    spending = st.slider("Spending Score", 1, 100, 50)

if st.button("🚀 Predict Cluster"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending]
    })

    scaled_input = scaler.transform(input_data)

    combined = np.vstack([scaled_features, scaled_input])

    temp_model = model
    labels = temp_model.fit_predict(combined)

    cluster = int(labels[-1])

    st.markdown(f"""
    <div class="prediction-box">
        Predicted Cluster: {cluster}
    </div>
    """, unsafe_allow_html=True)

    if cluster == 0:
        st.success("High Value Customers → Focus on Premium Services")
    elif cluster == 1:
        st.info("Potential Customers → Offer Promotions")
    elif cluster == 2:
        st.warning("Average Customers → Increase Engagement")
    elif cluster == 3:
        st.success("Loyal Customers → Loyalty Programs")
    elif cluster == 4:
        st.error("Low Engagement → Targeted Marketing Needed")

# ==============================================================
# DATA TABLE
# ==============================================================

if show_table:
    st.markdown("## 📋 Clustered Dataset")
    st.dataframe(clustered_df)

# ==============================================================
# DOWNLOAD BUTTON
# ==============================================================

csv = clustered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Clustered Dataset",
    data=csv,
    file_name="mall_customers_clustered.csv",
    mime="text/csv"
)

# ==============================================================
# ADDITIONAL ANALYSIS
# ==============================================================

st.markdown("---")
st.markdown("## 📈 Cluster Feature Comparison")

cluster_summary = clustered_df.groupby("Cluster").mean()

st.dataframe(cluster_summary)

bar_fig = px.bar(
    cluster_summary,
    barmode="group",
    title="Cluster Feature Averages"
)

st.plotly_chart(bar_fig, use_container_width=True)

# ==============================================================
# FOOTER
# ==============================================================

st.markdown("---")
st.markdown("""
<div class='footer-style'>
🛍️ Mall Customer Hierarchical Clustering Project <br>
Agglomerative Clustering | Streamlit Dashboard <br>
Machine Learning Portfolio
</div>
""", unsafe_allow_html=True)
