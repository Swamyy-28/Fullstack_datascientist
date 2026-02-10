# ============================================================
# Mall Customer Clustering – Streamlit Application
# Unsupervised Learning using K-Means
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR STYLING
# ============================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stMetric {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #667eea;
}

h1 {
    color: #ffffff;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 10px;
}

h2 {
    color: #ffffff;
    margin-top: 30px;
}

.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}

.cluster-info {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 5px solid #667eea;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the trained KMeans model.
    """
    try:
        model = joblib.load("kmeans_model.pkl")
        return model
    except Exception as e:
        st.error(f"Model file not found or error loading model: {e}")
        return None

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    """
    Load original mall customer dataset.
    """
    try:
        df = pd.read_csv("Mall_Customers (3).csv")
        return df
    except Exception as e:
        st.error(f"Dataset file not found or error reading file: {e}")
        return None

@st.cache_data
def load_clustered_data():
    """
    Load clustered customer dataset.
    """
    try:
        clustered_df = pd.read_csv("clustered_mall_customers.csv")
        return clustered_df
    except Exception:
        return None

# ============================================================
# CLUSTER INFORMATION
# ============================================================

CLUSTER_INFO = {
    0: {
        "name": "High Value Customers",
        "description": "Young customers with high spending score and good income",
        "color": "#FF6B6B",
        "characteristics": [
            "Age: Young (25–40)",
            "Income: High",
            "Spending Score: High"
        ]
    },
    1: {
        "name": "Potential Target",
        "description": "Middle-aged customers with moderate to high spending",
        "color": "#4ECDC4",
        "characteristics": [
            "Age: Middle-aged",
            "Income: Moderate",
            "Spending: Moderate to High"
        ]
    },
    2: {
        "name": "Average Customers",
        "description": "Customers with low to moderate spending behavior",
        "color": "#45B7D1",
        "characteristics": [
            "Age: Mixed",
            "Income: Low to Moderate",
            "Spending: Average"
        ]
    },
    3: {
        "name": "Loyal Customers",
        "description": "Older customers with consistent purchase patterns",
        "color": "#FFA07A",
        "characteristics": [
            "Age: Older",
            "Income: Stable",
            "Spending: Consistent"
        ]
    },
    4: {
        "name": "Budget Conscious",
        "description": "High income customers with low spending score",
        "color": "#98D8C8",
        "characteristics": [
            "Income: High",
            "Spending: Low",
            "Value-focused"
        ]
    }
}

# ============================================================
# TITLE & DESCRIPTION
# ============================================================

st.markdown("""
<h1 style="text-align:center;">🛍️ Mall Customer Clustering Prediction</h1>
<p style="color:white; font-size:16px; text-align:center;">
Predict customer segments using K-Means Unsupervised Learning
</p>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESOURCES
# ============================================================

model = load_model()
df = load_data()
clustered_df = load_clustered_data()

# ============================================================
# MAIN APPLICATION
# ============================================================

if model is not None and df is not None:

    col1, col2 = st.columns([1, 1], gap="large")

    # --------------------------------------------------------
    # CUSTOMER INPUT SECTION
    # --------------------------------------------------------

    with col1:
        st.markdown("### 📊 Customer Information")
        st.markdown("---")

        age = st.slider(
            "👤 Age",
            min_value=int(df["Age"].min()),
            max_value=int(df["Age"].max()),
            value=30,
            step=1
        )

        annual_income = st.slider(
            "💰 Annual Income (k$)",
            min_value=int(df["Annual Income (k$)"].min()),
            max_value=int(df["Annual Income (k$)"].max()),
            value=50,
            step=1
        )

        spending_score = st.slider(
            "🎯 Spending Score (1-100)",
            min_value=1,
            max_value=100,
            value=50,
            step=1
        )

        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        c1.metric("Age", f"{age} yrs")
        c2.metric("Income", f"${annual_income}k")
        c3.metric("Spending", f"{spending_score}/100")

    # --------------------------------------------------------
    # DATASET STATISTICS
    # --------------------------------------------------------

    with col2:
        st.markdown("### 📈 Dataset Statistics")
        st.markdown("---")

        st.metric("Total Customers", len(df))
        st.metric("Average Age", f"{df['Age'].mean():.1f}")
        st.metric("Average Income", f"${df['Annual Income (k$)'].mean():.1f}k")
        st.metric("Average Spending", f"{df['Spending Score (1-100)'].mean():.1f}")

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICTION SECTION
    # --------------------------------------------------------

    if st.button("🚀 Predict Cluster", use_container_width=True):

        input_data = pd.DataFrame({
            "Age": [age],
            "Annual Income (k$)": [annual_income],
            "Spending Score (1-100)": [spending_score]
        })

        # Scaling (same logic as original, safe)
        scaler = StandardScaler()
        scaler.fit(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
        scaled_input = scaler.transform(input_data)

        cluster = int(model.predict(scaled_input)[0])

        cluster_details = CLUSTER_INFO.get(
            cluster,
            {"name": "Unknown", "description": "No description available", "characteristics": []}
        )

        # ----------------------------------------------------
        # PREDICTION DISPLAY
        # ----------------------------------------------------

        st.markdown(f"""
        <div class="prediction-box">
            <h2>Predicted Cluster</h2>
            <h1>{cluster}</h1>
            <h3>{cluster_details['name']}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Cluster Description")
        st.write(cluster_details["description"])

        st.markdown("### 🎯 Key Characteristics")
        for char in cluster_details["characteristics"]:
            st.write(f"• {char}")

        # ----------------------------------------------------
        # VISUALIZATIONS
        # ----------------------------------------------------

        if clustered_df is not None:

            st.markdown("### 📊 Visualizations")
            v1, v2 = st.columns(2, gap="large")

            with v1:
                fig = px.scatter_3d(
                    clustered_df,
                    x="Age",
                    y="Annual Income (k$)",
                    z="Spending Score (1-100)",
                    color=clustered_df["Cluster"].astype(str),
                    title="3D Cluster Distribution"
                )

                fig.add_scatter3d(
                    x=[age],
                    y=[annual_income],
                    z=[spending_score],
                    mode="markers",
                    marker=dict(size=12, color="red"),
                    name="Your Input"
                )

                st.plotly_chart(fig, use_container_width=True)

            with v2:
                counts = clustered_df["Cluster"].value_counts().sort_index()
                colors = [CLUSTER_INFO[i]["color"] for i in counts.index]

                pie = go.Figure(
                    data=[go.Pie(
                        labels=[f"Cluster {i}" for i in counts.index],
                        values=counts.values,
                        marker=dict(colors=colors)
                    )]
                )

                pie.update_layout(title="Customer Distribution by Cluster")
                st.plotly_chart(pie, use_container_width=True)

        # ----------------------------------------------------
        # BUSINESS RECOMMENDATIONS
        # ----------------------------------------------------

        recommendations = {
            0: "Focus on premium retention and upselling strategies",
            1: "Target with seasonal offers and loyalty programs",
            2: "Increase engagement using discounts and bundles",
            3: "Strengthen long-term relationships",
            4: "Promote value-based premium products"
        }

        st.info(f"💡 Recommendation: {recommendations.get(cluster, 'N/A')}")

else:
    st.error("Failed to load model or dataset. Please ensure files exist.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:white; padding:20px;">
🛍️ Mall Customer Clustering | K-Means Unsupervised Learning Project
</div>
""", unsafe_allow_html=True)
