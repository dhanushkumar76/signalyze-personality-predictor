import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# === Paths ===
MODEL_PATH = "model/best_model.keras"
TARGET_SIZE = (128, 128)
CONF_MATRIX_DIR = "model"
LOSS_CURVE_PATH = "model/loss_curve.png"

# === Likert Scale Mapping ===
CLASS_MAP = {0: "Disagree", 1: "Neutral", 2: "Agree"}

# === Trait Names ===
TRAIT_NAMES = [
    "Confidence",
    "Emotional Stability",
    "Sociability",
    "Responsiveness",
    "Concentration",
    "Introversion",
    "Creativity",
    "Decision-Making"
]

# === Preprocessing Functions ===
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def deskew_image(img):
    gray = cv2.bitwise_not(img)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return np.full(TARGET_SIZE, 255, dtype=np.uint8)
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def center_pad(img):
    h, w = img.shape
    scale = min(TARGET_SIZE[0] / h, TARGET_SIZE[1] / w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pad_vert = max((TARGET_SIZE[0] - resized.shape[0]) // 2, 0)
    pad_horz = max((TARGET_SIZE[1] - resized.shape[1]) // 2, 0)
    padded = cv2.copyMakeBorder(resized,
                                top=pad_vert, bottom=TARGET_SIZE[0] - resized.shape[0] - pad_vert,
                                left=pad_horz, right=TARGET_SIZE[1] - resized.shape[1] - pad_horz,
                                borderType=cv2.BORDER_CONSTANT, value=255)
    return padded

def extract_visual_traits(img):
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

# === Streamlit App UI ===
st.set_page_config(page_title="Signalyze Personality Predictor", layout="centered")
st.title("🖋️ Signalyze: Signature-Based Personality Insight")
st.markdown("Upload a signature to predict key personality traits.")

uploaded_file = st.file_uploader("Upload Signature Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.subheader("📷 Original Signature")
    st.image(img, caption="Uploaded Signature", use_column_width=True)

    # === Preprocess ===
    img = apply_clahe(img)
    img = deskew_image(img)
    padded = center_pad(img)

    st.subheader("⚙️ Preprocessed Signature")
    st.image(padded, caption="Processed", use_column_width=True)

    # === Extract Traits ===
    traits = extract_visual_traits(padded)
    st.subheader("🧬 Extracted Visual Traits")
    st.table(pd.DataFrame([traits]))

    # === Model Prediction ===
    st.subheader("🧠 Predicted Personality Traits")

    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            padded_rgb = np.stack([padded] * 3, axis=-1).astype(np.float32)
            padded_rgb = tf.keras.applications.efficientnet.preprocess_input(padded_rgb)
            padded_rgb = padded_rgb.reshape(1, 128, 128, 3)

            features = np.array([[traits["ink_density"], traits["aspect_ratio"], traits["slant_angle"]]], dtype=np.float32)
            preds = model.predict([padded_rgb, features])

            for i, p in enumerate(preds):
                pred_class = np.argmax(p[0])
                confidence = float(p[0][pred_class])
                st.success(f"{TRAIT_NAMES[i]}: {CLASS_MAP[pred_class]} ({confidence:.2f})")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    else:
        st.warning("⚠️ Model not found or still training.")

# === Optional: Evaluation Visuals ===
st.sidebar.header("📊 Evaluation Visuals")

if st.sidebar.checkbox("Show Confusion Matrices"):
    st.subheader("📌 Confusion Matrices")
    for i in range(1, 9):
        path = os.path.join(CONF_MATRIX_DIR, f"conf_matrix_trait_{i}.png")
        if os.path.exists(path):
            st.image(path, caption=f"Trait {i}: {TRAIT_NAMES[i - 1]}", use_column_width=True)

if st.sidebar.checkbox("Show Loss Curve"):
    if os.path.exists(LOSS_CURVE_PATH):
        st.subheader("📉 Training Loss Curve")
        st.image(LOSS_CURVE_PATH, use_column_width=True)
