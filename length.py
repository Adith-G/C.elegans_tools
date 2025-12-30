# ======================================================
# Length Analysis
# ======================================================



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


# ======================
# PAGE SETUP
# ======================
st.set_page_config(
    page_title="Length Tool",
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




st.title("Curved Line Length Analyzer🌀")
st.write(
    "Skeletonize thick curved lines and measure **true length** "
    "by **graph traversal from endpoints**. Output in **mm**."
)

# ======================
# SIDEBAR
# ======================
st.sidebar.header("Image Settings")

invert = st.sidebar.checkbox(
    "Lines are black on white background",
    value=True
)

blur = st.sidebar.select_slider(
    "Pre-blur (noise reduction)",
    options=[1, 3, 5, 7, 9],
    value=3
)

min_component = st.sidebar.slider(
    "Remove small components (pixels)",
    0, 500, 30
)

# ======================
# SCALE
# ======================
st.sidebar.header("Scale Calibration")

pixels_per_mm = st.sidebar.number_input(
    "Scale (pixels per mm)",
    min_value=0.0001,
    value=1000.0,
    step=10.0,
    format="%.2f"
)

# ======================
# FILE UPLOAD
# ======================
uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "tif"]
)

# ======================
# GRAPH UTILITIES
# ======================
neighbors_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))
]

def get_neighbors(y, x, mask):
    for dy, dx, w in neighbors_8:
        ny, nx = y + dy, x + dx
        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
            if mask[ny, nx]:
                yield ny, nx, w

def find_endpoints(mask):
    endpoints = []
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        count = sum(1 for _ in get_neighbors(y, x, mask))
        if count == 1:
            endpoints.append((y, x))
    return endpoints

def longest_path_length(mask):
    """
    Graph traversal on skeleton:
    endpoint → endpoint longest path
    """
    endpoints = find_endpoints(mask)

    if len(endpoints) < 2:
        return 0.0

    def bfs(start):
        visited = set()
        queue = deque([(start[0], start[1], 0.0)])
        visited.add(start)
        farthest = (start, 0.0)

        while queue:
            y, x, dist = queue.popleft()
            if dist > farthest[1]:
                farthest = ((y, x), dist)

            for ny, nx, w in get_neighbors(y, x, mask):
                if (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append((ny, nx, dist + w))

        return farthest

    # Two-pass BFS (tree diameter)
    start = endpoints[0]
    far_node, _ = bfs(start)
    _, max_dist = bfs(far_node)

    return max_dist

# ======================
# MAIN
# ======================
if uploaded:

    img = np.array(Image.open(uploaded).convert("L"))

    if blur > 1:
        img = cv2.GaussianBlur(img, (blur, blur), 0)

    st.subheader("Original Image")
    st.image(img, clamp=True)

    _, binary = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if invert:
        binary = cv2.bitwise_not(binary)

    st.subheader("Binary Image")
    st.image(binary, clamp=True)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    skeleton = skeletonize(img_as_bool(binary))
    skeleton_img = (skeleton * 255).astype(np.uint8)

    st.subheader("Skeleton")
    st.image(skeleton_img, clamp=True)

    num_labels, labels = cv2.connectedComponents(skeleton_img)

    results = []

    for label in range(1, num_labels):
        component = (labels == label)
        pixel_count = np.sum(component)

        if pixel_count < min_component:
            continue

        length_pixels = longest_path_length(component)
        length_mm = length_pixels / pixels_per_mm

        results.append({
            "Line ID": label,
            "Skeleton pixels": int(pixel_count),
            "Path length (pixels)": round(length_pixels, 3),
            "Length (mm)": round(length_mm, 4)
        })

    df = pd.DataFrame(results)

    st.subheader("📊 Results")

    if not df.empty:
        st.metric("Total Length (mm)", f"{df['Length (mm)'].sum():.4f}")
        st.info(f"Scale used: **{pixels_per_mm} pixels per mm**")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No valid skeleton components found.")

    if not df.empty:
        st.download_button(
            "Download Results (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            "line_lengths_graph_mm.csv",
            "text/csv"
        )

else:
    st.info("⬆ Upload an image to start analysis.")
