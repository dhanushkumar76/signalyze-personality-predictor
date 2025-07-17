import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from datetime import datetime
from keras.models import load_model
from keras.saving import register_keras_serializable
from tensorflow.keras.losses import CategoricalCrossentropy
import zipfile

# === Config ===
st.set_page_config(page_title="🖋️ Signalyze Personality Predictor", layout="centered")
st.title("🖋️ Signalyze: Signature-Based Personality Insight")

MODEL_PATH = "model/best_model.keras"
CONF_MATRIX_DIR = "model"
LOG_FILE = "logs/prediction_log.csv"
TARGET_SIZE = (64, 64)

CLASS_MAP = {0: "Disagree", 1: "Neutral", 2: "Agree"}
CLASS_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}
TRAIT_NAMES = [
    "Confidence", "Emotional Stability", "Creativity", "Decision-Making"
]
NUM_TRAITS = 4

# --- Register Custom Losses (MUST match 3_train_model.py) ---
@register_keras_serializable()
def focal_loss_inner(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(weight * ce, axis=1)

@register_keras_serializable()
def weighted_loss_trait_1(y_true, y_pred):
    loss = CategoricalCrossentropy(label_smoothing=0.1)(y_true, y_pred)
    return tf.reduce_mean(loss)

@register_keras_serializable()
def weighted_loss_trait_2(y_true, y_pred):
    loss = CategoricalCrossentropy(label_smoothing=0.1)(y_true, y_pred)
    return tf.reduce_mean(loss)

@register_keras_serializable()
def weighted_loss_trait_3(y_true, y_pred):
    loss = focal_loss_inner(y_true, y_pred)
    return tf.reduce_mean(loss)

@register_keras_serializable()
def weighted_loss_trait_4(y_true, y_pred):
    loss = CategoricalCrossentropy(label_smoothing=0.1)(y_true, y_pred)
    return tf.reduce_mean(loss)

custom_objects_for_loading = {
    "weighted_loss_trait_1": weighted_loss_trait_1,
    "weighted_loss_trait_2": weighted_loss_trait_2,
    "weighted_loss_trait_3": weighted_loss_trait_3,
    "weighted_loss_trait_4": weighted_loss_trait_4,
    'focal_loss_inner': focal_loss_inner
}

# === Load Model ===
try:
    model = load_model(MODEL_PATH, custom_objects=custom_objects_for_loading)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.write("Please ensure the model file is in the 'model' folder and the custom loss functions are correctly defined.")
    st.stop()

# === Utility Functions ===
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
    return cv2.copyMakeBorder(
        resized, pad_vert, TARGET_SIZE[0] - resized.shape[0] - pad_vert,
        pad_horz, TARGET_SIZE[1] - resized.shape[1] - pad_horz,
        borderType=cv2.BORDER_CONSTANT, value=255
    )

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
    return {
        "ink_density": round(float(ink_density), 4),
        "aspect_ratio": round(float(aspect_ratio), 4),
        "slant_angle": round(float(np.degrees(angle_rad)), 2)
    }

# === Upload Image or ZIP ===
uploaded_file = st.file_uploader("Upload Signature Image or ZIP", type=["png", "jpg", "jpeg", "zip"])

if uploaded_file:
    if uploaded_file.type == "application/zip":
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            batch_results = []
            for fname in image_files:
                img_bytes = zip_ref.read(fname)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = apply_clahe(img)
                img = deskew_image(img)
                padded = center_pad(img)
                traits = extract_visual_traits(padded)
                
                padded_rgb = np.stack([padded] * 3, axis=-1).astype(np.float32)
                padded_rgb = padded_rgb / 255.0
                padded_rgb = padded_rgb.reshape(1, TARGET_SIZE[0], TARGET_SIZE[1], 3)
                
                features = np.array([[traits["ink_density"], traits["aspect_ratio"], traits["slant_angle"]]], dtype=np.float32)
                
                preds = model.predict([padded_rgb, features], verbose=0)
                predictions_dict = {}
                confidences = []
                for i, p in enumerate(preds):
                    pred_class = np.argmax(p[0])
                    confidence = float(p[0][pred_class])
                    trait_name = TRAIT_NAMES[i]
                    predictions_dict[trait_name] = f"{CLASS_MAP[pred_class]} ({confidence:.2f})"
                    confidences.append(confidence)
                batch_results.append({
                    "filename": fname,
                    **traits,
                    **predictions_dict
                })
            st.dataframe(pd.DataFrame(batch_results))
            st.download_button("Download Results (CSV)", pd.DataFrame(batch_results).to_csv(index=False), "batch_predictions.csv", "text/csv")
            st.stop()

    img = cv2.imdecode(np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    st.subheader("📷 Original Signature")
    st.image(img, caption="Uploaded", use_column_width=True)

    img = apply_clahe(img)
    img = deskew_image(img)
    padded = center_pad(img)

    st.subheader("⚙️ Preprocessed Signature")
    st.image(padded, caption="Preprocessed", use_column_width=True)

    traits = extract_visual_traits(padded)
    st.subheader("🧬 Extracted Visual Traits")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ink Density", f"{traits['ink_density']:.4f}")
    col2.metric("Aspect Ratio", f"{traits['aspect_ratio']:.4f}")
    col3.metric("Slant Angle", f"{traits['slant_angle']:.2f}°")

    st.subheader("🧠 Personality Predictions")

    try:
        padded_rgb = np.stack([padded] * 3, axis=-1).astype(np.float32)
        padded_rgb = padded_rgb / 255.0
        padded_rgb = padded_rgb.reshape(1, TARGET_SIZE[0], TARGET_SIZE[1], 3)
        
        features = np.array([[traits["ink_density"], traits["aspect_ratio"], traits["slant_angle"]]], dtype=np.float32)

        preds = model.predict([padded_rgb, features], verbose=0)

        predictions_dict = {}
        for i, p in enumerate(preds):
            pred_class = np.argmax(p[0])
            confidence = float(p[0][pred_class])
            trait_name = TRAIT_NAMES[i]
            predictions_dict[trait_name] = f"{CLASS_MAP[pred_class]} ({confidence:.2f})"
            st.success(f"{CLASS_COLORS[pred_class]} {trait_name}: {CLASS_MAP[pred_class]} ({confidence:.2f})")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "filename": uploaded_file.name,
            "ink_density": traits["ink_density"],
            "aspect_ratio": traits["aspect_ratio"],
            "slant_angle": traits["slant_angle"]
        }
        log_data.update(predictions_dict)
        
        if not os.path.exists(os.path.dirname(LOG_FILE)):
            os.makedirs(os.path.dirname(LOG_FILE))
            
        if os.path.exists(LOG_FILE):
            df_existing = pd.read_csv(LOG_FILE)
            df_combined = pd.concat([df_existing, pd.DataFrame([log_data])], ignore_index=True)
        else:
            df_combined = pd.DataFrame([log_data])
            
        df_combined.to_csv(LOG_FILE, index=False)
        st.info("✅ Prediction logged successfully")
        
        confidences = [float(p[0][np.argmax(p[0])]) for p in preds]
        st.subheader("🔬 Per-Trait Confidence")
        st.bar_chart(pd.DataFrame({"Trait": TRAIT_NAMES, "Confidence": confidences}).set_index("Trait"))
        
        st.download_button("Download Prediction (CSV)", pd.DataFrame([log_data]).to_csv(index=False), f"prediction_{timestamp}.csv", "text/csv")
        st.download_button("Download Prediction (JSON)", pd.DataFrame([log_data]).to_json(orient='records'), f"prediction_{timestamp}.json", "application/json")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# === Sidebar Visuals ===
st.sidebar.header("📊 Evaluation Visuals")
TRAINING_LOG_PATH = os.path.join("model", "training_log.csv")
EVAL_RESULTS_DIR = os.path.join("model", "evaluation_results")

# FIX 1: Loss Curve from CSV
if st.sidebar.checkbox("Show Loss Curve"):
    st.subheader("📉 Training Loss Curve (from logs)")
    if os.path.exists(TRAINING_LOG_PATH):
        df_log = pd.read_csv(TRAINING_LOG_PATH)
        if 'loss' in df_log.columns and 'val_loss' in df_log.columns:
            st.line_chart(df_log[["loss", "val_loss"]])
        else:
            st.info("Training log found, but loss data is missing. Please check the log file.")
    else:
        st.info("Training log not found. Run scripts/3_train_model.py to generate it.")

if st.sidebar.checkbox("Show Recent Predictions"):
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        if not df.empty:
            st.subheader("📘 Recent Predictions")
            st.dataframe(df.tail(10), use_container_width=True)
        else:
            st.info("No predictions logged yet.")
    else:
        st.info("No prediction history found.")

# FIX 2: Evaluation Results (Test Set)
st.sidebar.header("📈 Evaluation Results (Test Set)")
if st.sidebar.checkbox("Show Evaluation Visualizations (Test Set)"):
    eval_viz_path = os.path.join(EVAL_RESULTS_DIR, "evaluation_visualization.png")
    eval_summary_path = os.path.join(EVAL_RESULTS_DIR, "evaluation_summary.csv")

    if os.path.exists(eval_viz_path):
        st.subheader("📊 Evaluation Visualizations (Test Set)")
        st.image(eval_viz_path, caption="Evaluation Visualizations (Bar, Scatter, Confusion Matrices)", use_column_width=True)
    
    if os.path.exists(eval_summary_path):
        st.subheader("📋 Evaluation Summary Table (Test Set)")
        eval_df = pd.read_csv(eval_summary_path)
        st.dataframe(eval_df, use_container_width=True)
    
    if not os.path.exists(eval_viz_path) and not os.path.exists(eval_summary_path):
        st.info("No evaluation results found. Please run scripts/evaluate_model.py to generate them.")