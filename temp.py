import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import easyocr
import matplotlib.pyplot as plt
import re
import pandas as pd

# =============================
# SETTINGS
# =============================

video_path = input("Enter video path: ").strip().strip('"')
sample_every = 2   # seconds between samples

# =============================
# INIT
# =============================

reader = easyocr.Reader(['en'], gpu=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

temps = []
times = []

print("Processing with simple EasyOCR...")

# =============================
# MAIN LOOP
# =============================

for sec in range(0, duration, sample_every):

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ret, frame = cap.read()
    if not ret:
        continue

    # Crop center (adjust if needed)
    h, w = frame.shape[:2]
    crop = frame[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    results = reader.readtext(gray, detail=0)

    found_temp = None

    for text in results:
        match = re.search(r"\d{2,3}\.\d", text)
        if match:
            temp = float(match.group())
            if 20 <= temp <= 150:
                found_temp = temp
                break

    if found_temp is not None:
        temps.append(found_temp)
        times.append(sec)
        print(f"{sec}s → {found_temp}")
    else:
        print(f"{sec}s → no reading")

cap.release()

# =============================
# POST PROCESSING
# =============================

if len(temps) == 0:
    print("No valid temperatures detected.")
    exit()

df = pd.DataFrame({"time": times, "temp": temps})
df = df.set_index("time")

# Fill missing timestamps
full_index = range(0, duration, sample_every)
df = df.reindex(full_index)

# Interpolate missing values
df["temp"] = df["temp"].interpolate()

# Smooth curve (removes OCR spikes)
df["temp"] = df["temp"].rolling(3, center=True).mean()

# =============================
# PLOT
# =============================

plt.figure(figsize=(8,5))
plt.plot(df.index, df["temp"])
plt.xlabel("Time (seconds)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature vs Time")
plt.grid(True)
plt.show()
