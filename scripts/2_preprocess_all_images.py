# === scripts/2_preprocess_all_images.py ===

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt

# === CONFIG ===
INPUT_FOLDER = r"D:\signalyze\signalyze-personality-predictor\data\all_images"
OUTPUT_FOLDER = r"D:\signalyze\signalyze-personality-predictor\data\preprocessed_images"
FEATURE_CSV = r"D:\signalyze\signalyze-personality-predictor\data\signature_traits.csv"

# FIX: Set TARGET_SIZE to match the model's input size (64, 64)
TARGET_SIZE = (64, 64)
DEBUG_SAMPLES = 6 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Utility Functions ===
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def needs_inversion(img):
    return np.mean(img) > 127

def deskew_image(img):
    gray = cv2.bitwise_not(img)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return np.full(TARGET_SIZE, 255, dtype=np.uint8), 0
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), angle

def compute_visual_traits(img):
    binary = (img < 200).astype(np.uint8)
    coords = np.column_stack(np.where(binary > 0))

    if coords.shape[0] == 0:
        return {"ink_density": 0, "aspect_ratio": 1.0, "slant_angle": 0}

    y_coords, x_coords = coords[:, 0], coords[:, 1]
    height = y_coords.max() - y_coords.min()
    width = x_coords.max() - x_coords.min()
    aspect_ratio = height / width if width != 0 else 1.0
    ink_density = np.sum(binary) / (img.shape[0] * img.shape[1])

    x = coords - np.mean(coords, axis=0)
    cov = np.cov(x.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    angle_deg = np.degrees(angle_rad)

    return {
        "ink_density": round(float(ink_density), 4),
        "aspect_ratio": round(float(aspect_ratio), 4),
        "slant_angle": round(float(angle_deg), 2)
    }

# === Global collector for batch mode ===
trait_records = []

def preprocess_image(input_path, output_path, debug=False):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None

    if needs_inversion(img):
        img = cv2.bitwise_not(img)

    img = apply_clahe(img)
    img, _ = deskew_image(img)

    # Resize and pad
    h, w = img.shape
    scale = min(TARGET_SIZE[0]/h, TARGET_SIZE[1]/w)
    resized = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    pad_vert = max((TARGET_SIZE[0] - resized.shape[0]) // 2, 0)
    pad_horz = max((TARGET_SIZE[1] - resized.shape[1]) // 2, 0)

    padded = cv2.copyMakeBorder(resized,
        top=pad_vert,
        bottom=TARGET_SIZE[0] - resized.shape[0] - pad_vert,
        left=pad_horz,
        right=TARGET_SIZE[1] - resized.shape[1] - pad_horz,
        borderType=cv2.BORDER_CONSTANT,
        value=255
    )

    cv2.imwrite(output_path, padded)

    traits = compute_visual_traits(padded)
    traits["filename"] = os.path.basename(input_path)

    if not debug:
        trait_records.append(traits)

    if debug:
        return img, padded
    return traits

# === Visual Debug (Optional) ===
debug_pairs = []
sample_files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg')))[:DEBUG_SAMPLES]
for fname in sample_files:
    ipath = os.path.join(INPUT_FOLDER, fname)
    opath = os.path.join(OUTPUT_FOLDER, fname)
    result = preprocess_image(ipath, opath, debug=True)
    if result:
        before, after = result
        debug_pairs.append((before, after))

if debug_pairs:
    plt.figure(figsize=(16, 4))
    for i, (before, after) in enumerate(debug_pairs):
        plt.subplot(2, DEBUG_SAMPLES, i+1), plt.imshow(before, cmap='gray'), plt.title("Before"), plt.axis('off')
        plt.subplot(2, DEBUG_SAMPLES, i+1+DEBUG_SAMPLES), plt.imshow(after, cmap='gray'), plt.title("After"), plt.axis('off')
    plt.suptitle("Signature Preprocessing (Before vs After)", fontsize=14)
    plt.tight_layout()
    plt.show()

# === Batch Processing ===
print("\n🔄 Starting batch preprocessing...")
all_files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

def process_wrapper(filename):
    ipath = os.path.join(INPUT_FOLDER, filename)
    opath = os.path.join(OUTPUT_FOLDER, filename)
    return preprocess_image(ipath, opath)

pool = ThreadPool(processes=4)
results = list(tqdm(pool.imap(process_wrapper, all_files), total=len(all_files)))
pool.close()
pool.join()

# Save visual traits
traits_df = pd.DataFrame([r for r in results if r is not None])
traits_df.to_csv(FEATURE_CSV, index=False)
print(f"\n✅ Preprocessing done. Saved {len(traits_df)} images and traits to CSV:")
print(f"📁 Images: {OUTPUT_FOLDER}\n📄 Traits CSV: {FEATURE_CSV}")