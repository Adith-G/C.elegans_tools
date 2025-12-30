import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(page_title="Worm Body Bend Analyzer", layout="wide")



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




st.title("🪱 Worm Body Bend Analyzer")

st.write(
    "Body bends are quantified from **lateral body displacement relative to a moving worm-centric axis**. "
    "All intermediate signals and detected bend events are shown."
)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Analysis Parameters")

FRAME_STEP = st.sidebar.slider("Frame subsampling - 1 Recommended", 1, 10, 1)
SMOOTH_SIGMA = st.sidebar.slider("Temporal smoothing (σ)", 1, 10, 1)
DISP_THRESHOLD = st.sidebar.slider("Displacement threshold (px)", 0.5, 10.0, 2.5)
MIN_AREA = st.sidebar.slider("Minimum worm area (px)", 50, 2000, 150)

SHOW_OVERLAY = st.sidebar.checkbox("Show axis + body points overlay", True)
SHOW_TABLE = st.sidebar.checkbox("Show table of bend events", True)

# =========================
# SEGMENTATION
# =========================
def segment_worm(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    return binary

# =========================
# EXTRACT WORM CONTOUR
# =========================
def get_worm_contour(binary):
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None

    worm = max(contours, key=cv2.contourArea)
    if cv2.contourArea(worm) < MIN_AREA:
        return None

    return worm.reshape(-1, 2)

# =========================
# PCA BODY AXIS
# =========================
def worm_axis(points):
    center = np.mean(points, axis=0)
    centered = points - center

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)

    normal = np.array([-axis[1], axis[0]])
    return center, axis, normal

# =========================
# MEAN LATERAL DISPLACEMENT
# =========================
def mean_lateral_displacement(points, center, normal):
    lateral_offsets = np.dot(points - center, normal)
    n = len(lateral_offsets)
    idx = np.argsort(np.abs(lateral_offsets))
    mid = idx[int(0.3*n):int(0.7*n)]
    return np.mean(lateral_offsets[mid])

# =========================
# VIDEO ANALYSIS
# =========================
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 15

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_trace = []
    debug_frame = None

    frame_idx = 0
    progress = st.progress(0)
    status = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        progress.progress(frame_idx / total_frames)
        status.text(f"Processing frame {frame_idx}/{total_frames}")
        if frame_idx % FRAME_STEP != 0:
            continue

        binary = segment_worm(frame)
        contour = get_worm_contour(binary)
        if contour is None:
            continue

        center, axis, normal = worm_axis(contour)
        disp = mean_lateral_displacement(contour, center, normal)
        raw_trace.append(disp)

        if SHOW_OVERLAY and debug_frame is None:
            vis = frame.copy()
            c = center.astype(int)

            axis_start = (c - 120 * axis).astype(int)
            axis_end   = (c + 120 * axis).astype(int)
            cv2.line(vis, tuple(axis_start), tuple(axis_end), (0, 255, 0), 2)

            for p in contour[::10]:
                cv2.circle(vis, tuple(p.astype(int)), 2, (0, 0, 255), -1)

            debug_frame = vis

        #progress.progress(frame_idx / total_frames)

    cap.release()
    progress.empty()
    status.empty()

    raw_trace = np.array(raw_trace)
    if len(raw_trace) < 10:
        return None

    smooth_trace = gaussian_filter1d(raw_trace, SMOOTH_SIGMA)

    peaks, _ = find_peaks(smooth_trace, height=DISP_THRESHOLD)
    troughs, _ = find_peaks(-smooth_trace, height=DISP_THRESHOLD)

    bends = min(len(peaks), len(troughs))
    duration_min = len(raw_trace) / (fps / FRAME_STEP) / 60
    bpm = bends / duration_min if duration_min > 0 else 0

    return raw_trace, smooth_trace, peaks, troughs, bends, bpm, fps, debug_frame


#==========================
#Annotated Video Part
#==========================


def save_annotated_video(
    video_path,
    output_path,
    peaks,
    troughs,
    fps,
    frame_step
):
    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    signal_idx = 0

    peak_set = set(peaks)
    trough_set = set(troughs)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Only annotate frames used in analysis
        if frame_idx % frame_step != 0:
            out.write(frame)
            continue

        # Time stamp
        time_sec = signal_idx / (fps / frame_step)

        # Overlay text
        label = None
        color = None

        if signal_idx in peak_set:
            label = "PEAK (bend)"
            color = (0, 0, 255)
        elif signal_idx in trough_set:
            label = "TROUGH (bend)"
            color = (255, 0, 0)

        if label:
            cv2.putText(
                frame,
                f"{label} | t = {time_sec:.2f}s",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )

            cv2.circle(
                frame,
                (60, 80),
                12,
                color,
                -1
            )

        out.write(frame)
        signal_idx += 1

    cap.release()
    out.release()


# =========================
# UI
# =========================
video = st.file_uploader("Upload worm locomotion video", type=["mp4", "avi", "mov"])

RESULTS_DIR = "D:\PipeLine Dump Data"
os.makedirs(RESULTS_DIR, exist_ok=True)


# New save section
if video:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    video_path = os.path.join(run_dir, "original_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video.read())

    path = video_path

    st.video(video)

    if st.button("▶ Run Analysis"):
        with st.spinner("Analyzing…"):
            result = analyze_video(path)

        if result is None:
            st.error("Not enough valid frames detected.")
        else:
            raw, smooth, peaks, troughs, bends, bpm, fps, overlay = result
            annotated_path = os.path.join(run_dir, "annotated_bends.avi")

            save_annotated_video(
                video_path=path,
                output_path=annotated_path,
                peaks=peaks,
                troughs=troughs,
                fps=fps,
                frame_step=FRAME_STEP
            )

            effective_fps = fps / FRAME_STEP
            time_sec = np.arange(len(raw)) / effective_fps


        # Save signals
        pd.DataFrame({
            "time_sec": time_sec,
            "raw_displacement": raw,
            "smoothed_displacement": smooth
        }).to_csv(os.path.join(run_dir, "displacement_signal.csv"), index=False)

        # Save bend events
        events = []
        for i in peaks:
            events.append({"time_sec": time_sec[i], "type": "Peak", "value": smooth[i]})
        for i in troughs:
            events.append({"time_sec": time_sec[i], "type": "Trough", "value": smooth[i]})

        pd.DataFrame(events).sort_values("time_sec") \
            .to_csv(os.path.join(run_dir, "bend_events.csv"), index=False)

    # Save summary
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write(f"Total bends: {bends}\n")
            f.write(f"Bends per minute: {bpm:.2f}\n")
            f.write(f"FPS: {fps}\n")
            f.write(f"Frame step: {FRAME_STEP}\n")


            # =========================
            # TABS
            # =========================
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Summary",
                "📈 Raw & Smoothed Signal",
                "🎯 Bend Detection",
                "📋 Bend Events Table"
            ])

            # ---------- TAB 1 ----------
            with tab1:
                col1, col2 = st.columns(2)
                col1.metric("Total body bends", bends)
                col2.metric("Bends per minute", f"{bpm:.2f}")

                if SHOW_OVERLAY and overlay is not None:
                    st.subheader("Axis & body points overlay")
                    st.image(overlay)
                    st.subheader("Annotated bend detection video")
                    st.video(annotated_path)


            # ---------- TAB 2 ----------
            with tab2:
                fig, ax = plt.subplots()
                ax.plot(time_sec, raw, label="Raw displacement", alpha=0.6)
                ax.plot(time_sec, smooth, label="Smoothed displacement", linewidth=2)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Displacement (px)")
                ax.set_title("Raw and smoothed displacement signals")
                ax.legend()
                st.pyplot(fig)

            # ---------- TAB 3 ----------
            with tab3:
                fig, ax = plt.subplots()
                ax.plot(time_sec, raw, color="gray", alpha=0.4, label="Raw")
                ax.plot(time_sec, smooth, color="black", linewidth=2, label="Smoothed")

                ax.scatter(time_sec[peaks], smooth[peaks],
                           color="red", s=60, label="Peaks (bend)")
                ax.scatter(time_sec[troughs], smooth[troughs],
                           color="blue", s=60, label="Troughs (bend)")

                ax.axhline(DISP_THRESHOLD, color="red", linestyle="--", alpha=0.3)
                ax.axhline(-DISP_THRESHOLD, color="blue", linestyle="--", alpha=0.3)

                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Displacement (px)")
                ax.set_title("Detected body bends")
                ax.legend()
                st.pyplot(fig)

            # ---------- TAB 4 ----------
            with tab4:
                if SHOW_TABLE:
                    events = []
                    for i in peaks:
                        events.append({
                            "time_sec": time_sec[i],
                            "type": "Peak",
                            "displacement_px": smooth[i]
                        })
                    for i in troughs:
                        events.append({
                            "time_sec": time_sec[i],
                            "type": "Trough",
                            "displacement_px": smooth[i]
                        })

                    df = pd.DataFrame(events).sort_values("time_sec")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Enable 'Show table of bend events' in sidebar.")

            st.success("Analysis complete ✔")

else:
    st.info("Upload a video to begin analysis.")
