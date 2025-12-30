import streamlit as st
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_bool
from PIL import Image
import pandas as pd
import math
from collections import deque
from scipy import stats



#CSS PART


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



#===============================================
#Statistical Analysis
#===============================================

st.divider()
st.header("📊 Statistical Analysis (t-tests)")

st.write(
    "Upload an Excel file and perform **t-tests** between two groups. "
    "Excel percentage columns are handled correctly."
)

# ======================
# FILE UPLOAD
# ======================
excel_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"],
    key="stats_upload"
)

# ======================
# HELPER: detect Excel-style %
# ======================
def is_excel_percentage(series: pd.Series) -> bool:
    s = series.dropna()
    if s.empty:
        return False
    return s.max() <= 1.0 and s.min() >= 0.0

if excel_file:

    # ----------------------
    # READ DATA
    # ----------------------
    df = pd.read_excel(excel_file)

    # Freeze column names exactly as read
    column_names = list(df.columns)

    st.subheader("Preview of Data")
    st.dataframe(df.head(), width="stretch")

    # ----------------------
    # GROUP SELECTION
    # ----------------------
    st.subheader("Group Selection")

    col1 = st.selectbox(
        "Group 1 column",
        options=column_names,
        key="group1_col"
    )

    col2 = st.selectbox(
        "Group 2 column",
        options=column_names,
        key="group2_col"
    )

    test_type = st.radio(
        "t-test type",
        options=[
            "Independent (Welch)",
            "Paired"
        ]
    )

    # ----------------------
    # DATA CLEANING
    # ----------------------
    raw1 = pd.to_numeric(df[col1], errors="coerce").dropna()
    raw2 = pd.to_numeric(df[col2], errors="coerce").dropna()

    if len(raw1) < 2 or len(raw2) < 2:
        st.error("Each group must contain at least 2 numeric values.")
        st.stop()

    # ----------------------
    # FIX EXCEL PERCENTAGES
    # ----------------------
    if is_excel_percentage(raw1) and is_excel_percentage(raw2):
        data1 = raw1 * 100.0
        data2 = raw2 * 100.0
        unit = "%"
        unit_note = "Detected Excel-style percentages (0–1 → converted to %)"
    else:
        data1 = raw1
        data2 = raw2
        unit = "units"
        unit_note = "Detected raw numeric values"

    st.info(
        f"Comparing **{col1}** (N={len(data1)}) vs "
        f"**{col2}** (N={len(data2)})\n\n"
        f"Units: **{unit}** — {unit_note}"
    )

    # ----------------------
    # RUN TEST
    # ----------------------
    if st.button("Run t-test"):

        if test_type == "Independent (Welch)":
            t_stat, p_val = stats.ttest_ind(
                data1, data2, equal_var=False
            )
            df_used = len(data1) + len(data2) - 2
        else:
            min_len = min(len(data1), len(data2))
            t_stat, p_val = stats.ttest_rel(
                data1.iloc[:min_len],
                data2.iloc[:min_len]
            )
            df_used = min_len - 1

        # ----------------------
        # SUMMARY TABLE
        # ----------------------
        summary = pd.DataFrame({
            "Group name": [col1, col2],
            f"Mean ({unit})": [data1.mean(), data2.mean()],
            f"Std Dev ({unit})": [data1.std(ddof=1), data2.std(ddof=1)],
            "N": [len(data1), len(data2)]
        })

        st.subheader("📈 Group Statistics")
        st.dataframe(summary, width="stretch")

        # ----------------------
        # TEST RESULTS
        # ----------------------
        st.subheader("📌 Test Results")

        st.metric("t-statistic", f"{t_stat:.4f}")
        st.metric("p-value", f"{p_val:.6e}")
        st.caption(f"Degrees of freedom (approx): {df_used}")

        if p_val < 0.05:
            st.success("Statistically significant (p < 0.05)")
        else:
            st.warning("Not statistically significant (p ≥ 0.05)")

        # ----------------------
        # EXPORT
        # ----------------------
        export = pd.DataFrame({
            "Test type": [test_type],
            "Group 1": [col1],
            "Group 2": [col2],
            "Unit": [unit],
            "N Group 1": [len(data1)],
            "N Group 2": [len(data2)],
            "Mean Group 1": [data1.mean()],
            "Mean Group 2": [data2.mean()],
            "t statistic": [t_stat],
            "p value": [p_val]
        })

        st.download_button(
            "Download statistics (CSV)",
            export.to_csv(index=False).encode("utf-8"),
            file_name="t_test_results.csv",
            mime="text/csv"
        )
