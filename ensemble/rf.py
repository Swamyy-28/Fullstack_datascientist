import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Random Forest ML Platform",
    page_icon="🌳",
    layout="wide"
)

# ---------------- SOFT PROFESSIONAL THEME ----------------
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3 {
        color: #0f172a;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.5em 1em;
        font-weight: 600;
    }
    .stSidebar {
        background-color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🌳 Random Forest Machine Learning Platform")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Model Configuration")

task = st.sidebar.radio(
    "Select Task Type",
    ["Classification", "Regression"]
)

n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100)
max_depth = st.sidebar.slider("Max Depth", 2, 20, 5)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.25)

st.sidebar.markdown("---")
st.sidebar.markdown("**Algorithm:** Random Forest")
st.sidebar.markdown("**Developer:** Bunny")

# ---------------- STEP 1: DATA INGESTION ----------------
st.header("📥 Step 1: Data Ingestion")

if "df" not in st.session_state:
    st.session_state.df = None

source = st.radio(
    "Dataset Source",
    ["Upload CSV", "Load Sample Dataset"],
    horizontal=True
)

if source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Dataset Uploaded Successfully")

elif source == "Load Sample Dataset":
    if st.button("Load Iris Dataset"):
        st.session_state.df = sns.load_dataset("iris")
        st.success("Iris Dataset Loaded Successfully")

df = st.session_state.df

# ---------------- STEP 2: DATA OVERVIEW ----------------
if df is not None:
    st.header("📊 Step 2: Dataset Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head())
    with col2:
        st.write("**Shape:**", df.shape)
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), cmap="Blues", ax=ax)
    st.pyplot(fig)

# ---------------- STEP 3: PREPROCESSING ----------------
if df is not None:
    st.header("🧹 Step 3: Data Preprocessing")

    target = st.selectbox("Select Target Column", df.columns)

    df_clean = df.dropna()

    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    X = X.select_dtypes(include=np.number)

    if task == "Classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    st.success("Data cleaned and encoded successfully")

# ---------------- STEP 4: MODEL TRAINING ----------------
if df is not None:
    st.header("🚀 Step 4: Model Training & Evaluation")

    if st.button("Train Random Forest Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if task == "Classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"🎯 Accuracy: {acc*100:.2f}%")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success(f"📈 R² Score: {r2_score(y_test, y_pred):.4f}")
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(f"© {datetime.now().year} Random Forest ML Platform | Streamlit")
