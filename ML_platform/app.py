import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="End-to-End ML Platform",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 End-to-End Machine Learning Platform")
st.caption("Decision Tree • Naive Bayes • Random Forest • KNN")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Model Configuration")

model_name = st.sidebar.selectbox(
    "Select Algorithm",
    ["Decision Tree", "Naive Bayes", "Random Forest", "KNN"]
)

# Model-specific parameters
if model_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

elif model_name == "KNN":
    k = st.sidebar.slider("K (Neighbors)", 1, 15, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Developer:** Bunny")
st.sidebar.markdown("**Deployment:** Streamlit Cloud")

# ---------------- STEP 1: DATA INGESTION ----------------
# ---------------- STEP 1: DATA INGESTION ----------------
st.header("📥 Step 1: Data Ingestion")

if "df" not in st.session_state:
    st.session_state.df = None

source = st.radio(
    "Dataset Source",
    ["Upload CSV", "Load Sample Dataset"]
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


# ---------------- STEP 2: EDA ----------------
if df is not None:
    st.header("📊 Step 2: Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(df.head())

    with col2:
        st.write("Shape:", df.shape)
        st.write("Missing Values:")
        st.write(df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------- STEP 3: DATA CLEANING ----------------
if df is not None:
    st.header("🧹 Step 3: Data Cleaning")

    strategy = st.selectbox(
        "Missing Value Strategy",
        ["Mean", "Median", "Drop Rows"]
    )

    df_clean = df.copy()

    if strategy == "Drop Rows":
        df_clean.dropna(inplace=True)
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategy == "Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

    st.success("Data Cleaned Successfully")
    st.dataframe(df_clean.head())

# ---------------- STEP 4: MODEL TRAINING ----------------
if df is not None:
    st.header("🚀 Step 4: Model Training & Evaluation")

    target = st.selectbox("Select Target Column", df_clean.columns)

    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    X = X.select_dtypes(include=np.number)

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model selection
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=max_depth)

        elif model_name == "Naive Bayes":
            model = GaussianNB()

        elif model_name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth
            )

        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=k)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"🎯 {model_name} Accuracy: {acc * 100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

else:
    st.info("Please upload or load a dataset to proceed.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(f"© {datetime.now().year} End-to-End ML Platform | Streamlit")
