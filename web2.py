import streamlit as st
import requests
import cv2
import numpy as np
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
import tempfile



st.set_page_config(
    page_title="C. elegans Image Analysis",
    layout="wide"
)


def local_css(css):
    import streamlit as st
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

local_css("""
/* ===============================
   ANIMATIONS
================================ */
@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes softPulse {
    0% { box-shadow: 0 0 25px rgba(0,255,200,0.4); }
    50% { box-shadow: 0 0 60px rgba(0,180,255,0.9); }
    100% { box-shadow: 0 0 25px rgba(0,255,200,0.4); }
}

/* ===============================
   BACKGROUND — DEEP & LUXURIOUS
================================ */
.stApp {
    background: linear-gradient(
        -45deg,
        #031b16,
        #042f2e,
        #032c44,
        #031b16
    );
    background-size: 400% 400%;
    animation: gradientFlow 14s ease infinite;
    color: #eaffff;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ===============================
   HERO TITLE — SEDUCTIVE POWER
================================ */
h1 {
    font-size: 4rem !important;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(
        90deg,
        #00ffcc,
        #00b4ff,
        #00ffcc
    );
    background-size: 300% 300%;
    animation: gradientFlow 6s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1.8px;
    text-shadow: 0 0 40px rgba(0,255,204,0.9);
}

/* ===============================
   HEADERS
================================ */
h2 {
    font-size: 2.2rem !important;
    color: #00ffcc !important;
    text-shadow: 0 0 20px rgba(0,255,204,0.8);
}

h3 {
    color: #00b4ff !important;
}

/* ===============================
   MAIN CARD — SEXY GLASS
================================ */
.card {
    background: rgba(10,40,45,0.65);
    backdrop-filter: blur(22px);
    border-radius: 34px;
    padding: 48px;
    margin-top: 40px;
    border: 3px solid rgba(0,255,204,0.45);
    animation: softPulse 4.5s ease-in-out infinite;
}

/* ===============================
   SIDEBAR — NEON EDGE
================================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #042f2e, #02171f);
    border-right: 4px solid #00ffcc;
    box-shadow: 0 0 40px rgba(0,255,204,0.7);
}

/* ===============================
   BUTTONS — TEMPTING AF
================================ */
.stButton > button {
    background: linear-gradient(
        135deg,
        #00ffcc,
        #00b4ff
    );
    background-size: 300% 300%;
    animation: gradientFlow 5s ease infinite;
    color: #001b1f;
    border-radius: 30px;
    padding: 16px 44px;
    font-size: 18px;
    font-weight: 900;
    border: none;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: scale(1.15);
    box-shadow:
        0 0 40px rgba(0,255,204,1),
        0 0 80px rgba(0,180,255,1);
}

/* ===============================
   METRICS — POWER CELLS
================================ */
[data-testid="stMetric"] {
    background: rgba(0,0,0,0.55);
    padding: 26px;
    border-radius: 26px;
    border: 3px solid #00ffcc;
    box-shadow:
        0 0 35px rgba(0,255,204,0.9);
}


""")







# Folders for output
os.makedirs("output/images", exist_ok=True)
os.makedirs("output/predictions", exist_ok=True)
os.makedirs("traning_data", exist_ok=True)

# ======================
#     STREAMLIT UI
# ======================


st.title("Egg Analysis Tool 🥚")
st.divider()
st.header("Run the Analysis")

st.markdown("### 📤 Upload Your Image")
uploaded = st.file_uploader("`", type=["jpg", "png", "jpeg", "tif"])

API_URL = "http://localhost:8000/predict"   # My FastAPI address


if uploaded:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        temp_path = tmp.name

    # Send file to FastAPI
    with open(temp_path, "rb") as f:
        res = requests.post(API_URL, files={"file": f})

    if res.status_code != 200:
        st.error("API Error: " + res.text)
        st.stop()

    response = res.json()

    st.success("Analysis Completed!")

    # Read returned paths
    annotated_path = response["annotated_image"]
    csv_path = response["csv"]

    # Load CSV results
    df = pd.read_csv(csv_path)
    df["Confidence(%)"] = df["confidence"] * 100

    # Map numeric labels → names (like before)
    # Your FastAPI CSV uses "label" column with class index
    label_map = {
        0: "Egg",       
        1: "Worm",
        2: "Dirt"
    }

    df["Object_Name"] = df["label"].map(label_map)
    class_counts = Counter(df["Object_Name"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_path = f"output/images/{ts}_original.jpg"
    annotated_copy_path = f"output/images/{ts}_annotated.jpg"
    csv_copy_path = f"output/predictions/{ts}.csv"

    # Save files locally again for your local archive
    img = cv2.imread(temp_path)
    cv2.imwrite(original_path, img)
    cv2.imwrite(annotated_copy_path, cv2.imread(annotated_path))
    df.to_csv(csv_copy_path, index=False)

    object_count = len(df)
    st.write(f"**Objects detected:** {object_count}")

    # TABS
    tab1, tab2 = st.tabs(["Summary", "Graphs"])

    # ---------------- TAB 1 ---------------- #
    with tab1:
        st.subheader("🧾 Object Summary")
        for obj, count in class_counts.items():
            st.write(f"**{obj}: {count} detected**")

        st.subheader("Detection Details")
        st.dataframe(df)

        st.image(annotated_copy_path, channels="BGR")

    # ---------------- TAB 2 ---------------- #
    with tab2:
        col1, col2 = st.columns([1,1])

        # BAR GRAPH
        with col1:
            st.subheader("Class Count Bar Graph 🤓")
            fig, ax = plt.subplots(figsize=(6,4), dpi=120)
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#9B9B9B")
            bars = ax.bar(class_counts.keys(), class_counts.values(), color='maroon')
            ax.set_xlabel("Object Class")
            ax.set_ylabel("Count")
            ax.set_title("Detected Objects per Class")
            ax.grid(alpha=0.3)

            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height(),
                    f"{bar.get_height()}",
                    ha='center', va='bottom', fontsize=10
                )
            st.pyplot(fig)

        # CONFIDENCE HISTOGRAM
        with col2:
            st.subheader("Confidence Histogram😎")
            fig2, ax2 = plt.subplots(figsize=(6,4), dpi=120)
            fig.patch.set_facecolor("#9B9B9B")
            ax2.hist(df["Confidence(%)"], bins=10, edgecolor='black', color='pink',  alpha=0.8)
            ax2.set_xlabel("Confidence (%)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Confidence Distribution")
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

        # PIE CHART
        st.subheader("Class Distribution Pie Chart 🧐")
        fig3, ax3 = plt.subplots()
        ax3.pie(class_counts.values(), labels=class_counts.keys(), autopct="%1.1f%%")
        ax3.set_title("Class Distribution")
        st.pyplot(fig3)

    # DOWNLOAD CSV BUTTON
    st.download_button(
        label="Download CSV 🪱",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"prediction_{ts}.csv",
        mime="text/csv"
    )


# ======================================================
#   TRAINING IMAGE UPLOAD
# ======================================================

st.title("Upload Images for Future Training")
st.header("No detection will be performed on these !")
st.markdown("### 👀 Upload raw images to store for dataset creation")
training_uploads = st.file_uploader(
    "`",
    type=["jpg", "jpeg", "png", "tif", "mp4"],
    accept_multiple_files=True
)

if training_uploads:
    saved_files = []

    for file in training_uploads:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = f"traning_data/training_data{ts}_{file.name.replace(' ', '_')}"
        bytes_data = file.read()

        with open(save_path, "wb") as f:
            f.write(bytes_data)

        saved_files.append(save_path)

    st.success(f"✅ Saved {len(saved_files)} images for training!")

    with st.expander("Show Saved File Paths"):
        for path in saved_files:
            st.write(path)
            
