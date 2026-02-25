import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from scipy.io.wavfile import write

st.set_page_config(page_title="Producer Data → Music Engine", layout="wide")

st.title("🎛 Producer-Level Graph → Music Engine")

# ----------------------
# Musical Scales (Hz)
# ----------------------
SCALES = {
    "C Major": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],
    "C Minor": [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16, 523.25],
    "Pentatonic": [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
}

# ----------------------
# Wave Generators
# ----------------------
def generate_wave(freq, duration, sr, wave_type):
    t = np.linspace(0, duration, int(sr * duration), False)

    if wave_type == "Sine":
        return np.sin(2 * np.pi * freq * t)
    elif wave_type == "Square":
        return np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == "Sawtooth":
        return 2 * (t * freq - np.floor(0.5 + t * freq))
    elif wave_type == "Triangle":
        return 2 * np.abs(generate_wave(freq, duration, sr, "Sawtooth")) - 1

# ----------------------
# Upload CSV
# ----------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=np.number).columns

    selected_columns = st.multiselect("Select Columns", numeric_columns)

    scale_name = st.selectbox("Scale", list(SCALES.keys()))
    bpm = st.slider("Tempo (BPM)", 60, 200, 120)
    wave_type = st.selectbox("Wave Type", ["Sine", "Square", "Sawtooth", "Triangle"])
    chords_enabled = st.checkbox("Enable Chord Harmonization")

    if selected_columns:

        sr = 44100
        beat = 60 / bpm
        scale = SCALES[scale_name]

        final_left = np.array([], dtype=np.float32)
        final_right = np.array([], dtype=np.float32)

        for col_index, col in enumerate(selected_columns):

            values = df[col].dropna().values
            if len(values) < 2:
                continue

            norm = (values - values.min()) / (values.max() - values.min() + 1e-8)

            pitch_indices = (norm * (len(scale) - 1)).astype(int)
            freqs = [scale[i] for i in pitch_indices]

            slopes = np.abs(np.diff(values, prepend=values[0]))
            slope_norm = slopes / (slopes.max() + 1e-8)
            durations = beat * (1 - 0.6 * slope_norm)

            track = np.array([], dtype=np.float32)

            for i, (freq, dur) in enumerate(zip(freqs, durations)):

                wave = generate_wave(freq, dur, sr, wave_type)

                # Add chords
                if chords_enabled:
                    third = freq * (5/4)
                    fifth = freq * (3/2)
                    wave += 0.5 * generate_wave(third, dur, sr, wave_type)
                    wave += 0.4 * generate_wave(fifth, dur, sr, wave_type)

                # Fade
                fade_len = int(0.01 * sr)
                if len(wave) > fade_len:
                    fade = np.linspace(0, 1, fade_len)
                    wave[:fade_len] *= fade
                    wave[-fade_len:] *= fade[::-1]

                track = np.concatenate((track, wave))

            # Stereo panning
            pan = col_index / max(1, len(selected_columns)-1)
            left = track * (1 - pan)
            right = track * pan

            if len(final_left) < len(left):
                final_left = np.pad(final_left, (0, len(left) - len(final_left)))
                final_right = np.pad(final_right, (0, len(right) - len(final_right)))

            final_left[:len(left)] += left
            final_right[:len(right)] += right

        # ----------------------
        # Drum Layer (Kick + Snare)
        # ----------------------
        drum_track = np.zeros_like(final_left)
        for i in range(0, len(drum_track), int(sr * beat)):
            drum_track[i:i+200] += np.random.uniform(-1, 1, min(200, len(drum_track)-i))

        final_left += 0.3 * drum_track
        final_right += 0.3 * drum_track

        # Normalize
        max_val = max(np.max(np.abs(final_left)), np.max(np.abs(final_right))) + 1e-8
        final_left /= max_val
        final_right /= max_val

        stereo = np.vstack((final_left, final_right)).T
        stereo_int16 = np.int16(stereo * 32767)

        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(tmp_wav.name, sr, stereo_int16)

        st.success("🔥 Producer-Level Track Generated!")
        st.audio(tmp_wav.name)

        with open(tmp_wav.name, "rb") as f:
            st.download_button("Download WAV", f, "producer_graph_music.wav")
