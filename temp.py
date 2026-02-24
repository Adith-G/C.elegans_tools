import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import easyocr
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# ==================================================
# SETTINGS
# ==================================================

video_path = input("Enter video path: ").strip().strip('"')
sample_every = 1

L_mm = 3.0
T_hot_C = 120.0
T_amb_C = 24.0

rho_cp = 7.2e5
h_conv = 10.0
epsilon = 0.85
sigma = 5.67e-8

# ==================================================
# STEP 1: EASY OCR EXTRACTION
# ==================================================

reader = easyocr.Reader(['en'], gpu=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)

temps = []
times = []

print("\nExtracting temperature from video...\n")

for sec in range(0, duration, sample_every):

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ret, frame = cap.read()
    if not ret:
        continue

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
        print(f"[{sec:4d}s]  {found_temp:.2f} °C")

cap.release()

if len(temps) == 0:
    print("No temperature data extracted.")
    exit()

# ==================================================
# DATA CLEANING
# ==================================================

df = pd.DataFrame({"Time (s)": times, "Temperature (°C)": temps})
df = df.set_index("Time (s)")

full_index = range(0, duration, sample_every)
df = df.reindex(full_index)

df["Temperature (°C)"] = df["Temperature (°C)"].interpolate()
df["Temperature (°C)"] = df["Temperature (°C)"].rolling(3, center=True).mean()
df = df.dropna()

time_data = df.index.values
temp_data = df["Temperature (°C)"].values

# ==================================================
# THERMAL MODEL PREPARATION
# ==================================================

L = L_mm / 1000.0
T_hot = T_hot_C + 273.15
T_amb = T_amb_C + 273.15
T_init = temp_data[0] + 273.15
temp_data_K = temp_data + 273.15

# ==================================================
# FDM SIMULATION (k + contact fit)
# ==================================================

def simulate_top_temperature(t_eval, k_eff, h_contact):

    alpha = k_eff / rho_cp

    N = 25
    dx = L / (N - 1)
    dt = 0.4 * (dx**2) / alpha

    T = np.full(N, T_init)

    T_top_history = []
    t_current = 0.0
    eval_idx = 0

    while eval_idx < len(t_eval):

        if t_current + dt > t_eval[eval_idx]:
            dt_step = t_eval[eval_idx] - t_current
            save_frame = True
        else:
            dt_step = dt
            save_frame = False

        T_new = T.copy()

        # Internal nodes
        T_new[1:-1] = T[1:-1] + alpha * dt_step / dx**2 * \
            (T[:-2] - 2*T[1:-1] + T[2:])

        # Bottom boundary (contact heat transfer)
        q_contact = h_contact * (T_hot - T[0])

        T_new[0] = T[0] + dt_step * (
            (2 * alpha / dx**2) * (T[1] - T[0])
            + (2 / (rho_cp * dx)) * q_contact
        )

        # Top boundary (convection + radiation)
        q_conv = h_conv * (T[-1] - T_amb)
        q_rad = epsilon * sigma * (T[-1]**4 - T_amb**4)

        T_new[-1] = T[-1] + dt_step * (
            (2 * alpha / dx**2) * (T[-2] - T[-1])
            - (2 / (rho_cp * dx)) * (q_conv + q_rad)
        )

        T = T_new
        t_current += dt_step

        if save_frame:
            T_top_history.append(T[-1])
            eval_idx += 1

    return np.array(T_top_history)

# ==================================================
# OPTIMIZATION
# ==================================================

print("\nRunning inverse model (k + contact)...\n")

initial_guess = [0.15, 500]
bounds = ([0.01, 10], [5.0, 5000])

popt, _ = curve_fit(
    simulate_top_temperature,
    time_data,
    temp_data_K,
    p0=initial_guess,
    bounds=bounds
)

k_eff_opt = popt[0]
h_contact_opt = popt[1]
alpha_eff_opt = k_eff_opt / rho_cp

simulated_temp_K = simulate_top_temperature(time_data, k_eff_opt, h_contact_opt)
simulated_temp_C = simulated_temp_K - 273.15

r_squared = r2_score(temp_data, simulated_temp_C)
rmse = np.sqrt(mean_squared_error(temp_data, simulated_temp_C))

# ==================================================
# RESULTS
# ==================================================

print("\n========================================")
print("UPDATED MODEL RESULTS")
print("========================================")
print(f"Thermal Conductivity (k):     {k_eff_opt:.4f} W/(m·K)")
print(f"Contact Coefficient (h_c):    {h_contact_opt:.2f} W/m²K")
print(f"Thermal Diffusivity (α):      {alpha_eff_opt:.4e} m²/s")
print(f"R²:                           {r_squared:.4f}")
print(f"RMSE:                         {rmse:.4f} °C")
print("========================================")

plt.figure(figsize=(10,6))
plt.plot(time_data, temp_data, 'ko', markersize=3, alpha=0.5, label="Experimental")
plt.plot(time_data, simulated_temp_C, 'r-', linewidth=2.5,
         label=f"Model Fit (k={k_eff_opt:.3f}, h_c={h_contact_opt:.0f})")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Thermal Property Extraction (Contact Model)")
plt.legend()
plt.grid(True)
plt.show()
