import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE CONFIG (MUST BE FIRST) ----------------
st.set_page_config(page_title="End-to-End SVM", layout="wide")
st.title("End-to-End SVM Platform")

# ---------------- LOGGER FUNCTION ----------------
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# ---------------- SESSION STATE ----------------
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False

# ---------------- FOLDER SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

log("Application Started")
log(f"RAW_DIR = {RAW_DIR}")
log(f"CLEAN_DIR = {CLEAN_DIR}")

# ---------------- SIDEBAR ----------------
st.sidebar.title("SVM Settings")
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

log(f"SVM Settings → kernel={kernel}, C={C}, gamma={gamma}")

# ---------------- STEP 1: DATA INGESTION ----------------
st.header("Step 1: Data Ingestion")
log("Step 1 Started")

option = st.radio("Choose Data Source", ["Download Dataset", "Upload CSV"])
df = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)

        raw_path = os.path.join(RAW_DIR, "iris.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(raw_path)
        st.success("Iris Dataset Downloaded Successfully")
        log(f"Iris dataset saved at {raw_path}")

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        raw_path = os.path.join(RAW_DIR, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = pd.read_csv(raw_path)
        st.success("CSV File Uploaded Successfully")
        log(f"Uploaded dataset saved at {raw_path}")

# ---------------- STEP 2: EDA ----------------
if df is not None:
    st.header("Step 2: Exploratory Data Analysis")
    log("Step 2 Started")

    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Missing Values:", df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    log("EDA Completed")

# ---------------- STEP 3: DATA CLEANING ----------------
if df is not None:
    st.header("Step 3: Data Cleaning")

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

    st.session_state.df_clean = df_clean
    st.success("Data Cleaning Completed")
    log("Data Cleaning Completed")

else:
    st.info("Please complete Step 1 first.")

# ---------------- STEP 4: SAVE CLEANED DATA ----------------
st.header("Step 4: Save Cleaned Dataset")

if st.button("Save Cleaned Dataset"):
    if "df_clean" not in st.session_state:
        st.error("No cleaned data found. Complete Step 3 first.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cleaned_dataset_{timestamp}.csv"
        path = os.path.join(CLEAN_DIR, filename)

        st.session_state.df_clean.to_csv(path, index=False)
        st.success("Cleaned Dataset Saved")
        st.info(f"Saved at: {path}")
        log(f"Cleaned dataset saved at {path}")

# ---------------- STEP 5: LOAD CLEANED DATA ----------------
st.header("Step 5: Load Cleaned Dataset")

clean_files = os.listdir(CLEAN_DIR)

if not clean_files:
    st.warning("No cleaned datasets found.")
    log("No cleaned datasets available")
else:
    selected = st.selectbox("Select Cleaned Dataset", clean_files)
    df_model = pd.read_csv(os.path.join(CLEAN_DIR, selected))
    st.success(f"Loaded Dataset: {selected}")
    st.dataframe(df_model.head())
    log(f"Loaded cleaned dataset: {selected}")

# ---------------- STEP 6: TRAIN SVM ----------------
if 'df_model' in locals():
    st.header("Step 6: Train SVM Model")
    log("Step 6 Started")

    target = st.selectbox("Select Target Column", df_model.columns)
    y = df_model[target]

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)
        log("Target column encoded")

    X = df_model.drop(columns=[target])
    X = X.select_dtypes(include=np.number)

    if X.empty:
        st.error("No numeric features available.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.success(f"SVM Accuracy: {acc * 100:.2f}%")
    log(f"SVM Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.warning("Please load a cleaned dataset in Step 5.")
