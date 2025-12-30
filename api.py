from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from datetime import datetime
import shutil
import cv2
import numpy as np
import pandas as pd
import os

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load your YOLO model
model = YOLO(r"C:\Users\adith\Documents\yolo\my_model\my_model.pt")

# Create folders
os.makedirs("api_output/images", exist_ok=True)
os.makedirs("api_output/predictions", exist_ok=True)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # save uploaded image
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = f"api_output/images/{ts}_input.jpg"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read image
    img = cv2.imread(input_path)

    # YOLO prediction
    results = model(img)
    annotated = results[0].plot(line_width=2, font_size=20)

    # Save annotated image
    output_path = f"api_output/images/{ts}_annotated.jpg"
    cv2.imwrite(output_path, annotated)

    # Extract detections → CSV
    boxes = results[0].boxes.data.cpu().numpy()
    df = pd.DataFrame(
        boxes,
        columns=["xmin","ymin","xmax","ymax","confidence","label"]
    )

    #Aspect ratio part
    df["width"] = df["xmax"] - df["xmin"]
    df["height"] = df["ymax"] - df["ymin"]
    df["aspect_ratio"] = (df["width"] / df["height"]).round(3)

    #Area Part
    df["area"] = df["width"] * df["height"]


    #eccentricity part
    df["eccentricity"] = np.sqrt(
    1 - (np.minimum(df["width"], df["height"]) /
         np.maximum(df["width"], df["height"])) ** 2
    )
    
    csv_path = f"api_output/predictions/{ts}.csv"
    df.to_csv(csv_path, index=False)

    # Count detections
    obj_count = len(df)

    return {
        "status": "success",
        "total_objects": obj_count,
        "annotated_image": output_path,
        "csv": csv_path
    }

   
