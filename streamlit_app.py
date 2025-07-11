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
LOSS_CURVE_PATH = "model/loss_curve.png"
LOG_FILE = "logs/prediction_log.csv"
TARGET_SIZE = (128, 128)

CLASS_MAP = {0: "Disagree", 1: "Neutral", 2: "Agree"}
CLASS_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}
TRAIT_NAMES = [
    "Confidence", "Emotional Stability", "Creativity", "Decision-Making"
]
NUM_TRAITS = 4

# Only 4 traits are used throughout the codebase and UI.
# All trait-related logic, analytics, and visualizations are for:
# - Confidence
# - Emotional Stability
# - Creativity
# - Decision-Making
# If you see any reference to 8 traits, it is legacy and should be ignored or removed.

# === Register Custom Losses ===
trait_weights = {f"trait_{i+1}": tf.Variable(1.0, trainable=False, dtype=tf.float32) for i in range(NUM_TRAITS)}

@register_keras_serializable()
def focal_loss_inner(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(weight * ce, axis=1)

def make_registered_loss(trait_name, base_loss):
    @register_keras_serializable(name=f"weighted_loss_{trait_name}")
    def trait_loss(y_true, y_pred):
        return trait_weights[trait_name] * base_loss(y_true, y_pred)
    return trait_loss

loss_map = {}
for i in range(NUM_TRAITS):
    key = f"trait_{i+1}"
    # Use focal loss only for 'Creativity' (index 2)
    use_focal = i == 2
    base = focal_loss_inner if use_focal else CategoricalCrossentropy()
    loss_map[key] = make_registered_loss(key, base)

# === Load Model ===
model = load_model(MODEL_PATH, custom_objects=loss_map)

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
        # Batch prediction for ZIP
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
                padded_rgb = tf.keras.applications.efficientnet.preprocess_input(padded_rgb)
                padded_rgb = padded_rgb.reshape(1, 128, 128, 3)
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
            # Download batch results
            st.download_button("Download Results (CSV)", pd.DataFrame(batch_results).to_csv(index=False), "batch_predictions.csv", "text/csv")
            st.stop()
    # ...existing code for single image...
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

    # === Prediction ===
    st.subheader("🧠 Personality Predictions")

    try:
        padded_rgb = np.stack([padded] * 3, axis=-1).astype(np.float32)
        padded_rgb = tf.keras.applications.efficientnet.preprocess_input(padded_rgb)
        padded_rgb = padded_rgb.reshape(1, 128, 128, 3)
        features = np.array([[traits["ink_density"], traits["aspect_ratio"], traits["slant_angle"]]], dtype=np.float32)

        preds = model.predict([padded_rgb, features], verbose=0)

        predictions_dict = {}
        for i, p in enumerate(preds):
            pred_class = np.argmax(p[0])
            confidence = float(p[0][pred_class])
            trait_name = TRAIT_NAMES[i]
            predictions_dict[trait_name] = f"{CLASS_MAP[pred_class]} ({confidence:.2f})"
            st.success(f"{CLASS_COLORS[pred_class]} {trait_name}: {CLASS_MAP[pred_class]} ({confidence:.2f})")

        # Ensure timestamp is always present and robust log creation
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "filename": uploaded_file.name,
            "ink_density": traits["ink_density"],
            "aspect_ratio": traits["aspect_ratio"],
            "slant_angle": traits["slant_angle"]
        }
        log_data.update(predictions_dict)
        df_log = pd.DataFrame([log_data])
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        if os.path.exists(LOG_FILE):
            df_existing = pd.read_csv(LOG_FILE)
            df_combined = pd.concat([df_existing, df_log], ignore_index=True)
        else:
            df_combined = df_log
        df_combined.to_csv(LOG_FILE, index=False)
        st.info("✅ Prediction logged successfully")
        # Per-trait confidence bar plot
        confidences = [float(p[0][np.argmax(p[0])]) for p in preds]
        st.subheader("🔬 Per-Trait Confidence")
        st.bar_chart(pd.DataFrame({"Trait": TRAIT_NAMES, "Confidence": confidences}).set_index("Trait"))
        # Download this prediction
        st.download_button("Download Prediction (CSV)", df_log.to_csv(index=False), f"prediction_{timestamp}.csv", "text/csv")
        st.download_button("Download Prediction (JSON)", df_log.to_json(orient='records'), f"prediction_{timestamp}.json", "application/json")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# === Sidebar Visuals ===
st.sidebar.header("📊 Evaluation Visuals")

# --- Confusion Matrices ---
if st.sidebar.checkbox("Show Confusion Matrices"):
    st.subheader("📌 Confusion Matrices (Per Trait)")
    for i in range(NUM_TRAITS):
        path = os.path.join(CONF_MATRIX_DIR, f"conf_matrix_trait_{i+1}.png")
        if os.path.exists(path):
            st.image(path, caption=f"Confusion Matrix: {TRAIT_NAMES[i]}", use_column_width=True)

# --- Loss Curve and Accuracy Bar Chart ---
if st.sidebar.checkbox("Show Loss Curve"):
    if os.path.exists(LOSS_CURVE_PATH):
        st.subheader("📉 Training Loss Curve (Combined Traits)")
        st.image(LOSS_CURVE_PATH, use_column_width=True)
    # Show accuracy bar chart from training_log.csv
    log_path = os.path.join("model", "training_log.csv")
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        st.subheader("📊 Final Accuracy by Trait (Combined)")
        acc_cols = [col for col in df_log.columns if col.endswith('_accuracy') and not col.startswith('val_')]
        val_acc_cols = [col for col in df_log.columns if col.startswith('val_') and col.endswith('_accuracy')]
        if acc_cols and val_acc_cols:
            train_acc = df_log[acc_cols].iloc[-1].values
            val_acc = df_log[val_acc_cols].iloc[-1].values
            acc_df = pd.DataFrame({
                'Trait': TRAIT_NAMES,
                'Train': train_acc,
                'Val': val_acc
            }).set_index('Trait')
            st.bar_chart(acc_df)

        # --- Scatterplot: Accuracy vs F1-Score (if available) ---
        f1_cols = [col for col in df_log.columns if col.endswith('_f1') and not col.startswith('val_')]
        val_f1_cols = [col for col in df_log.columns if col.startswith('val_') and col.endswith('_f1')]
        if f1_cols and val_f1_cols:
            train_f1 = df_log[f1_cols].iloc[-1].values
            val_f1 = df_log[val_f1_cols].iloc[-1].values
            st.subheader("🔬 Validation Accuracy vs F1-Score by Trait (Scatterplot)")
            import plotly.express as px
            scatter_df = pd.DataFrame({
                'Trait': TRAIT_NAMES,
                'Train Accuracy': train_acc,
                'Val Accuracy': val_acc,
                'Train F1': train_f1,
                'Val F1': val_f1
            })
            fig = px.scatter(scatter_df, x='Val Accuracy', y='Val F1', text='Trait',
                             labels={'Val Accuracy': 'Validation Accuracy', 'Val F1': 'Validation F1-Score'},
                             title='Validation Accuracy vs F1-Score by Trait (Scatterplot)')
            st.plotly_chart(fig, use_container_width=True)

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

# === Sidebar: Evaluation Results from evaluate_model.py ===
# ...existing code...

# === Sidebar: Evaluation Results from evaluate_model.py ===
st.sidebar.header("📈 Evaluation Results (Test Set)")
EVAL_RESULTS_DIR = "model/evaluation_results"

if st.sidebar.checkbox("Show Evaluation Visualizations (Test Set)"):
    # Show main evaluation visualization (bar, scatter, confusion)
    eval_viz_path = os.path.join(EVAL_RESULTS_DIR, "evaluation_visualization.png")
    if os.path.exists(eval_viz_path):
        st.subheader("📊 Evaluation Visualizations (Test Set)")
        st.image(eval_viz_path, caption="Evaluation Visualizations (Bar, Scatter, Confusion Matrices)", use_column_width=True)
    else:
        st.info("No evaluation visualizations found. Run scripts/evaluate_model.py to generate.")

    # Show summary table and filter to 4 traits
    eval_summary_path = os.path.join(EVAL_RESULTS_DIR, "evaluation_summary.csv")
    if os.path.exists(eval_summary_path):
        st.subheader("📋 Evaluation Summary Table (Test Set)")
        eval_df = pd.read_csv(eval_summary_path)
        # Ensure only the 4 traits are shown
        trait_names = ["Confidence", "Emotional Stability", "Creativity", "Decision-Making"]
        eval_df = eval_df[eval_df['Trait'].isin(trait_names)]
        st.dataframe(eval_df, use_container_width=True)

        # Bar chart for accuracy
        st.subheader("📊 Final Accuracy by Trait (Test Set)")
        st.bar_chart(eval_df.set_index("Trait")["Accuracy"])

        # Scatterplot for Accuracy vs F1-Score
        st.subheader("📈 Accuracy vs F1-Score (Test Set)")
        st.scatter_chart(eval_df.set_index("Trait")[["Accuracy", "F1_Macro"]])
    else:
        st.info("No evaluation summary found. Run scripts/evaluate_model.py to generate.")

    # Show overall metrics
    overall_metrics_path = os.path.join(EVAL_RESULTS_DIR, "overall_metrics.csv")
    if os.path.exists(overall_metrics_path):
        st.subheader("📈 Overall Metrics (Test Set)")
        overall_df = pd.read_csv(overall_metrics_path)
        st.dataframe(overall_df, use_container_width=True)

# ...existing