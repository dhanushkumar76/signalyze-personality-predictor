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
import matplotlib.pyplot as plt

# === Setup ===
st.set_page_config(page_title="🖋️ Signalyze Personality Predictor", layout="centered")
st.title("🖋️ Signalyze: Signature-Based Personality Insight")

# === Paths ===
MODEL_PATH = "model/best_model.keras"
LOG_FILE = "logs/prediction_log.csv"
TRAINING_LOG_PATH = "model/training_log.csv"
EVAL_RESULTS_DIR = "model/evaluation_results"

# === Constants ===
TARGET_SIZE = (64, 64)
NUM_TRAITS = 5
TRAIT_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
CLASS_MAP = {0: "Disagree", 1: "Neutral", 2: "Agree"}
CLASS_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}

# === Register Loss Functions ===
@register_keras_serializable(name="weighted_loss")
def weighted_loss(y_true, y_pred):
    return CategoricalCrossentropy(label_smoothing=0.1)(y_true, y_pred)

for i in range(1, NUM_TRAITS + 1):
    @register_keras_serializable(name=f"weighted_loss_trait_{i}")
    def trait_loss(y_true, y_pred):
        return CategoricalCrossentropy(label_smoothing=0.1)(y_true, y_pred)

custom_objects = {
    "weighted_loss": weighted_loss,
    **{f"weighted_loss_trait_{i}": trait_loss for i in range(1, NUM_TRAITS + 1)}
}

# === Load Model ===
try:
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()

# === Preprocessing ===
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
    M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)
    return cv2.warpAffine(img, M, img.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def center_pad(img):
    h, w = img.shape
    scale = min(TARGET_SIZE[0] / h, TARGET_SIZE[1] / w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    pad_vert = max((TARGET_SIZE[0] - resized.shape[0]) // 2, 0)
    pad_horz = max((TARGET_SIZE[1] - resized.shape[1]) // 2, 0)
    return cv2.copyMakeBorder(resized, pad_vert, TARGET_SIZE[0] - resized.shape[0] - pad_vert,
                              pad_horz, TARGET_SIZE[1] - resized.shape[1] - pad_horz,
                              borderType=cv2.BORDER_CONSTANT, value=255)

# === File Upload ===
uploaded_file = st.file_uploader("Upload Signature Image or ZIP", type=["png", "jpg", "jpeg", "zip"])

# === Prediction ===
def predict_and_display(img, filename="uploaded_image"):
    img = apply_clahe(img)
    img = deskew_image(img)
    padded = center_pad(img)
    st.image(padded, caption="🖼️ Preprocessed", use_column_width=True)

    padded_rgb = np.stack([padded]*3, axis=-1).astype(np.float32) / 255.0
    padded_rgb = padded_rgb.reshape(1, *TARGET_SIZE, 3)

    preds = model.predict([padded_rgb], verbose=0)
    predictions_dict = {}
    confidences = []

    st.subheader("🧠 Personality Predictions")
    for i, p in enumerate(preds):
        pred_class = np.argmax(p[0])
        confidence = float(p[0][pred_class])
        trait_name = TRAIT_NAMES[i]
        predictions_dict[trait_name] = f"{CLASS_MAP[pred_class]} ({confidence:.2f})"
        confidences.append(confidence)
        st.success(f"{trait_name}: {CLASS_MAP[pred_class]} ({confidence:.2f})")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"timestamp": timestamp, "filename": filename}
    log_entry.update(predictions_dict)

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    try:
        if os.path.exists(LOG_FILE):
            df_existing = pd.read_csv(LOG_FILE, on_bad_lines='skip')
            df_combined = pd.concat([df_existing, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            df_combined = pd.DataFrame([log_entry])
        df_combined.to_csv(LOG_FILE, index=False)
    except Exception as e:
        st.warning(f"⚠️ Logging error: {e}")

    st.subheader("🔬 Confidence Scores")
    conf_df = pd.DataFrame({"Trait": TRAIT_NAMES, "Confidence": confidences}).set_index("Trait")
    st.bar_chart(conf_df)

    st.download_button("Download Prediction (CSV)", pd.DataFrame([log_entry]).to_csv(index=False),
                       f"prediction_{timestamp}.csv", "text/csv")
    st.download_button("Download Prediction (JSON)", pd.DataFrame([log_entry]).to_json(orient='records'),
                       f"prediction_{timestamp}.json", "application/json")

# === ZIP Processing ===
if uploaded_file:
    if uploaded_file.type == "application/zip":
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            batch_results = []
            for fname in image_files:
                img_bytes = zip_ref.read(fname)
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    padded = center_pad(deskew_image(apply_clahe(img)))
                    padded_rgb = np.stack([padded]*3, axis=-1).astype(np.float32) / 255.0
                    padded_rgb = padded_rgb.reshape(1, *TARGET_SIZE, 3)
                    preds = model.predict([padded_rgb], verbose=0)
                    prediction = {TRAIT_NAMES[i]: f"{CLASS_MAP[np.argmax(p[0])]} ({p[0][np.argmax(p[0])]:.2f})" for i, p in enumerate(preds)}
                    batch_results.append({"filename": fname, **prediction})
            df_batch = pd.DataFrame(batch_results)
            st.dataframe(df_batch)
            st.download_button("Download Batch Results (CSV)", df_batch.to_csv(index=False),
                               "batch_predictions.csv", "text/csv")
            st.stop()
    else:
        img = cv2.imdecode(np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        st.image(img, caption="📃 Original Signature", use_column_width=True)
        predict_and_display(img, uploaded_file.name)

# === Sidebar Visuals ===
st.sidebar.header("📊 Evaluation & Training Visuals")

# Load training log
try:
    df_log = pd.read_csv(TRAINING_LOG_PATH, on_bad_lines='skip')
except Exception as e:
    st.sidebar.warning(f"⚠️ Couldn't read training log: {e}")
    df_log = pd.DataFrame()

# Loss Curve
if st.sidebar.checkbox("Show Loss Curve"):
    st.subheader("📉 Training Loss Curve")
    if not df_log.empty and {'loss', 'val_loss'}.issubset(df_log.columns):
        st.line_chart(df_log[["loss", "val_loss"]])
    else:
        st.info("Loss columns missing in training log.")

# Accuracy Curve
if st.sidebar.checkbox("Show Trait Accuracy Curves"):
    acc_cols = [f"trait_{i+1}_accuracy" for i in range(NUM_TRAITS)]
    val_acc_cols = [f"val_trait_{i+1}_accuracy" for i in range(NUM_TRAITS)]

    if all(col in df_log.columns for col in acc_cols + val_acc_cols):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, trait_name in enumerate(TRAIT_NAMES):
            ax.plot(df_log.index, df_log[acc_cols[i]], label=f'Train {trait_name}', linestyle='-')
            ax.plot(df_log.index, df_log[val_acc_cols[i]], label=f'Val {trait_name}', linestyle='--')
        ax.set_title("📈 Training and Validation Accuracy by Trait")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Accuracy columns missing in training log.")

# === Individual Trait Accuracy Curve ===
if st.sidebar.checkbox("Show Individual Trait Accuracy Curve"):
    trait_map = {TRAIT_NAMES[i]: i for i in range(NUM_TRAITS)}
    selected_trait = st.sidebar.selectbox("📌 Select Trait", options=TRAIT_NAMES)

    trait_idx = trait_map[selected_trait]
    train_col = f"trait_{trait_idx+1}_accuracy"
    val_col = f"val_trait_{trait_idx+1}_accuracy"

    if train_col in df_log.columns and val_col in df_log.columns:
        st.subheader(f"📈 Accuracy Curve: {selected_trait}")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_log.index, df_log[train_col], label="Train", color="blue")
        ax.plot(df_log.index, df_log[val_col], label="Validation", color="orange")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{selected_trait} Accuracy Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("Trait accuracy columns not found in training log.")

# === Recent Predictions ===
if st.sidebar.checkbox("Show Recent Predictions"):
    if os.path.exists(LOG_FILE):
        try:
            df_pred_log = pd.read_csv(LOG_FILE, on_bad_lines='skip')
            if not df_pred_log.empty:
                st.subheader("📘 Recent Predictions")
                st.dataframe(df_pred_log.tail(10), use_container_width=True)
            else:
                st.info("No predictions logged yet.")
        except Exception as e:
            st.warning(f"⚠️ Could not read prediction log: {e}")
    else:
        st.info("Prediction log not found.")

# === Evaluation Results (Test Set) ===
st.sidebar.header("📈 Evaluation Results (Test Set)")

if st.sidebar.checkbox("Show Evaluation Visualizations (Test Set)"):
    EVAL_RESULTS_DIR = os.path.join("model", "evaluation_results")
    eval_viz_path = os.path.join(EVAL_RESULTS_DIR, "f1_accuracy_chart.png")
    eval_summary_path = os.path.join(EVAL_RESULTS_DIR, "metrics_report.csv")
    combined_cm_path = os.path.join(EVAL_RESULTS_DIR, "combined_conf_matrix_grid.png")

    st.subheader("📊 Trait Evaluation Metrics")
    if os.path.exists(eval_viz_path):
        st.image(eval_viz_path, use_column_width=True)
    else:
        st.info("No evaluation metrics visualization found.")

    # === Individual Confusion Matrices ===
    st.subheader("📌 Individual Confusion Matrices")
    trait_cm_images = []
    for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]:
        image_path = os.path.join(EVAL_RESULTS_DIR, f"conf_matrix_{trait}.png")
        if os.path.exists(image_path):
            trait_cm_images.append((trait, image_path))

    if trait_cm_images:
        for trait, path in trait_cm_images:
            st.markdown(f"**{trait}**")
            st.image(path, use_column_width=True)
    else:
        st.info("ℹ️ No individual confusion matrix images found.")

    st.subheader("📋 Evaluation Summary Table")
    if os.path.exists(eval_summary_path):
        try:
            eval_df = pd.read_csv(eval_summary_path)
            st.dataframe(eval_df, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not read evaluation summary: {e}")
    else:
        st.info("No metrics report CSV found.")

    
