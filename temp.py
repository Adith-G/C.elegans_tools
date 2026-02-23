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

# Output filenames (saved in same folder as script)
csv_filename = "temperature_data.csv"
plot_filename = "temperature_plot.png"

# =============================
# INIT
# =============================

reader = easyocr.Reader(['en'], gpu=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

temps = []
times = []

print("Processing with EasyOCR...")

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

df = pd.DataFrame({"Time (s)": times, "Temperature (°C)": temps})
df = df.set_index("Time (s)")

# Fill missing timestamps
full_index = range(0, duration, sample_every)
df = df.reindex(full_index)

# Interpolate missing values
df["Temperature (°C)"] = df["Temperature (°C)"].interpolate()

# Smooth curve
df["Temperature (°C)"] = df["Temperature (°C)"].rolling(3, center=True).mean()

# =============================
# SAVE CSV
# =============================

df.to_csv(csv_filename)
print(f"\nCSV saved as: {csv_filename}")

# =============================
# SAVE PLOT
# =============================

plt.figure(figsize=(8,5))
plt.plot(df.index, df["Temperature (°C)"])
plt.xlabel("Time (seconds)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature vs Time")
plt.grid(True)

plt.savefig(plot_filename, dpi=300)
print(f"Plot saved as: {plot_filename}")

plt.show()
