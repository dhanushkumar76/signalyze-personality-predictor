import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from utils.logger import log_prediction
from sklearn.metrics import f1_score, accuracy_score, classification_report

# === Paths ===
MODEL_PATH = "model/best_model.keras"
TARGET_SIZE = (128, 128)
CONF_MATRIX_DIR = "model"
LOSS_CURVE_PATH = "model/loss_curve.png"

# === Likert Scale Mapping ===
CLASS_MAP = {0: "Disagree", 1: "Neutral", 2: "Agree"}
CLASS_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}

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
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def deskew_image(img):
    """Correct the skew of the image using minimum area rectangle"""
    try:
        gray = cv2.bitwise_not(img)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return np.full(TARGET_SIZE, 255, dtype=np.uint8)
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        st.warning(f"Deskewing failed: {e}. Using original image.")
        return img

def center_pad(img):
    """Center the image and pad to target size"""
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
    """Extract ink density, aspect ratio, and slant angle from signature"""
    try:
        binary = (img < 200).astype(np.uint8)
        coords = np.column_stack(np.where(binary > 0))
        if coords.shape[0] == 0:
            return {"ink_density": 0, "aspect_ratio": 1.0, "slant_angle": 0}
        
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        height = y_coords.max() - y_coords.min()
        width = x_coords.max() - x_coords.min()
        aspect_ratio = height / width if width != 0 else 1.0
        ink_density = np.sum(binary) / (img.shape[0] * img.shape[1])
        
        # Calculate slant angle using PCA
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
    except Exception as e:
        st.error(f"Error extracting visual traits: {e}")
        return {"ink_density": 0, "aspect_ratio": 1.0, "slant_angle": 0}

def load_evaluation_metrics():
    """Load model evaluation metrics if available"""
    try:
        if os.path.exists("model/training_log.csv"):
            log_df = pd.read_csv("model/training_log.csv")
            if not log_df.empty:
                return {
                    "final_val_loss": log_df["val_loss"].iloc[-1],
                    "final_train_loss": log_df["loss"].iloc[-1],
                    "best_val_loss": log_df["val_loss"].min(),
                    "epochs_trained": len(log_df)
                }
    except Exception as e:
        st.warning(f"Could not load training metrics: {e}")
    return None

# === Streamlit App UI ===
st.set_page_config(page_title="Signalyze Personality Predictor", layout="centered")
st.title("🖋️ Signalyze: Signature-Based Personality Insight")
st.markdown("Upload a signature to predict key personality traits using deep learning.")

# Show model metrics if available
metrics = load_evaluation_metrics()
if metrics:
    st.sidebar.header("🎯 Model Performance")
    st.sidebar.metric("Best Validation Loss", f"{metrics['best_val_loss']:.4f}")
    st.sidebar.metric("Final Training Loss", f"{metrics['final_train_loss']:.4f}")
    st.sidebar.metric("Epochs Trained", metrics['epochs_trained'])

uploaded_file = st.file_uploader("Upload Signature Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read and display original image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        st.error("❌ Could not process the uploaded image. Please try a different format.")
        st.stop()

    st.subheader("📷 Original Signature")
    st.image(img, caption="Uploaded Signature", use_column_width=True)

    # === Preprocess ===
    with st.spinner("Processing image..."):
        img = apply_clahe(img)
        img = deskew_image(img)
        padded = center_pad(img)

    st.subheader("⚙️ Preprocessed Signature")
    st.image(padded, caption="Processed & Normalized", use_column_width=True)

    # === Extract Traits ===
    traits = extract_visual_traits(padded)
    st.subheader("🧬 Extracted Visual Traits")
    
    # Display traits in a nice format
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ink Density", f"{traits['ink_density']:.4f}")
    with col2:
        st.metric("Aspect Ratio", f"{traits['aspect_ratio']:.4f}")
    with col3:
        st.metric("Slant Angle", f"{traits['slant_angle']:.2f}°")

    # === Model Prediction ===
    st.subheader("🧠 Predicted Personality Traits")

    if os.path.exists(MODEL_PATH):
        try:
            with st.spinner("Loading model and making predictions..."):
                model = load_model(MODEL_PATH)
                
                # Prepare image input
                padded_rgb = np.stack([padded] * 3, axis=-1).astype(np.float32)
                padded_rgb = tf.keras.applications.efficientnet.preprocess_input(padded_rgb)
                padded_rgb = padded_rgb.reshape(1, 128, 128, 3)

                # Prepare features input
                features = np.array([[traits["ink_density"], traits["aspect_ratio"], traits["slant_angle"]]], dtype=np.float32)
                
                # Make predictions
                preds = model.predict([padded_rgb, features], verbose=0)

                # Process and display predictions
                predictions_dict = {}
                
                for i, trait_pred in enumerate(preds):
                    pred_class = np.argmax(trait_pred[0])
                    confidence = float(trait_pred[0][pred_class])
                    trait_name = TRAIT_NAMES[i]
                    prediction_text = CLASS_MAP[pred_class]
                    
                    # Store for logging
                    predictions_dict[trait_name] = f"{prediction_text} ({confidence:.2f})"
                    
                    # Display with color coding
                    color_icon = CLASS_COLORS[pred_class]
                    st.success(f"{color_icon} **{trait_name}**: {prediction_text} (Confidence: {confidence:.2f})")

                # Log the prediction
                try:
                    log_prediction(
                        filename=uploaded_file.name,
                        traits=traits,
                        predictions=predictions_dict
                    )
                    st.info("✅ Prediction logged successfully")
                except Exception as log_error:
                    st.warning(f"⚠️ Logging failed: {log_error}")

                # Display prediction summary
                st.subheader("📊 Prediction Summary")
                summary_df = pd.DataFrame([
                    {
                        "Trait": TRAIT_NAMES[i],
                        "Prediction": CLASS_MAP[np.argmax(preds[i][0])],
                        "Confidence": f"{preds[i][0][np.argmax(preds[i][0])]:.3f}",
                        "Probabilities": f"[{', '.join([f'{p:.2f}' for p in preds[i][0]])}]"
                    }
                    for i in range(len(preds))
                ])
                st.dataframe(summary_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.error("This might be due to model compatibility issues. Please check the model file.")
    else:
        st.warning("⚠️ Model not found. Please ensure 'best_model.keras' exists in the model/ directory.")

# === Optional: Evaluation Visuals ===
st.sidebar.header("📊 Model Evaluation")

if st.sidebar.checkbox("Show Confusion Matrices"):
    st.subheader("📌 Confusion Matrices by Trait")
    
    # Display confusion matrices in a grid
    cols = st.columns(2)
    for i in range(1, 9):
        path = os.path.join(CONF_MATRIX_DIR, f"conf_matrix_trait_{i}.png")
        if os.path.exists(path):
            with cols[(i-1) % 2]:
                st.image(path, caption=f"Trait {i}: {TRAIT_NAMES[i - 1]}", use_column_width=True)
        else:
            st.warning(f"Confusion matrix for trait {i} not found")

if st.sidebar.checkbox("Show Training Progress"):
    if os.path.exists(LOSS_CURVE_PATH):
        st.subheader("📉 Training Loss Curve")
        st.image(LOSS_CURVE_PATH, use_column_width=True)
    else:
        st.warning("Loss curve not found. Run training script to generate.")

# Show prediction history
if st.sidebar.checkbox("Show Recent Predictions"):
    log_file = "logs/prediction_log.csv"
    if os.path.exists(log_file):
        try:
            log_df = pd.read_csv(log_file)
            if not log_df.empty:
                st.subheader("📝 Recent Predictions")
                st.dataframe(log_df.tail(10), use_container_width=True)
            else:
                st.info("No predictions logged yet.")
        except Exception as e:
            st.error(f"Error loading prediction log: {e}")
    else:
        st.info("No prediction log found yet.")

# Footer
st.markdown("---")
st.markdown("**Signalyze** - Personality prediction through signature analysis using EfficientNetB0 + visual trait fusion")
