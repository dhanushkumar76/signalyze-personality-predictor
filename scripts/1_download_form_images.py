import pandas as pd
import requests
import os
import shutil
import cv2
import numpy as np
from io import BytesIO

# === CONFIGURATION ===
CSV_PATH = r"D:\signalyze\signalyze-personality-predictor\data\form_responses.csv"
RAW_378_FOLDER = r"D:\signalyze\signalyze-personality-predictor\data\raw_378"
OUTPUT_FOLDER = r"D:\signalyze\signalyze-personality-predictor\data\all_images"
IMAGE_LINK_COLUMN = "Upload Your Signature Image"

# === CREATE OUTPUT FOLDER IF NOT EXISTS ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === LOAD CSV AND DOWNLOAD FIRST 12 IMAGES ===
df = pd.read_csv(CSV_PATH)

print("🔽 Downloading first 12 form-submitted images...")
for i in range(69):
    try:
        img_url = df.loc[i, IMAGE_LINK_COLUMN]

        if "drive.google.com" not in img_url:
            print(f"❌ Skipping non-Drive URL: {img_url}")
            continue

        # Convert to direct download link
        if "file/d/" in img_url:
            file_id = img_url.split("file/d/")[1].split("/")[0]
        elif "id=" in img_url:
            file_id = img_url.split("id=")[1]
        else:
            print(f"❌ Unrecognized Drive format: {img_url}")
            continue

        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        # Download image
        response = requests.get(direct_url, timeout=15)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        # Save image with default Drive image name (Screenshot or whatever it is)
        filename = f"form_image_{i+1}.png"
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, img)

        print(f"✅ Saved: {filename}")

    except Exception as e:
        print(f"❌ Error downloading image from row {i+1}: {e}")

# === COPY 378 IMAGES INTO all_images FOLDER ===
print("\n📂 Copying 378 existing images to all_images folder...")
raw_files = os.listdir(RAW_378_FOLDER)

copied = 0
for file in raw_files:
    src = os.path.join(RAW_378_FOLDER, file)
    dst = os.path.join(OUTPUT_FOLDER, file)
    try:
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1
    except Exception as e:
        print(f"❌ Failed copying {file}: {e}")

print(f"✅ Copied {copied} images from raw_378 folder.")
print("📁 all_images folder is now ready with 390 images total.")
