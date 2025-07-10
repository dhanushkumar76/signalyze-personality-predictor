import os, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.saving import register_keras_serializable
from tensorflow.keras.losses import CategoricalCrossentropy

# === Paths ===
MODEL_PATH = "model/best_model.keras"
TARGET_SIZE = (128, 128)
CONF_MATRIX_DIR = "model"
LOSS_CURVE_PATH = "model/loss_curve.png"

CLASS_MAP = {0: "Disagree", 1: "Neutral", 2: "Agree"}
CLASS_COLORS = {0: "🔴", 1: "🟡", 2: "🟢"}
TRAIT_NAMES = [
    "Confidence", "Emotional Stability", "Sociability", "Responsiveness",
    "Concentration", "Introversion", "Creativity", "Decision-Making"
]

NUM_TRAITS = 8
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
    trait_loss.__name__ = f"weighted_loss_{trait_name}"
    return trait_loss

loss_map = {}
for i in range(NUM_TRAITS):
    trait_key = f"trait_{i+1}"
    use_focal = i in [2, 5, 6]
    base = focal_loss_inner if use_focal else CategoricalCrossentropy()
    loss_map[f"weighted_loss_{trait_key}"] = make_registered_loss(trait_key, base)

model = load_model(MODEL_PATH, custom_objects=loss_map)

# === Preprocessing Utilities ===
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def deskew_image(img):
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
    except Exception:
        return img

def center_pad(img):
    h, w = img.shape
    scale = min(TARGET_SIZE[0] / h, TARGET_SIZE[1] / w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pad_vert = max((TARGET_SIZE[0] - resized.shape[0]) // 2, 0)
    pad_horz = max((TARGET_SIZE[1] - resized.shape[1]) // 2, 0)
    padded = cv2.copyMakeBorder(
        resized,
        top=pad_vert, bottom=TARGET_SIZE[0] - resized.shape[0] - pad_vert,
        left=pad_horz, right=TARGET_SIZE[1] - resized.shape[1] - pad_horz,
        borderType=cv2.BORDER_CONSTANT, value=255
    )
    return padded

def extract_visual_traits(img):
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
    except Exception:
        return {"ink_density": 0, "aspect_ratio": 1.0, "slant_angle": 0}
    
# === Streamlit App UI ===
st.set_page_config(page_title="Signalyze Personality Predictor", layout="centered")
st.title("🖋️ Signalyze: Signature-Based Personality Insight")
st.markdown("Upload a signature to predict key personality traits using deep learning.")

uploaded_file = st.file_uploader("Upload Signature Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("❌ Could not process the uploaded image. Try a different file.")
        st.stop()

    st.subheader("📷 Original Signature")
    st.image(img, caption="Uploaded Signature", use_column_width=True)

    with st.spinner("Processing image..."):
        img = apply_clahe(img)
        img = deskew_image(img)
        padded = center_pad(img)

    st.subheader("⚙️ Preprocessed Signature")
    st.image(padded, caption="Processed & Normalized", use_column_width=True)

    traits = extract_visual_traits(padded)
    st.subheader("🧬 Extracted Visual Traits")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ink Density", f"{traits['ink_density']:.4f}")
    col2.metric("Aspect Ratio", f"{traits['aspect_ratio']:.4f}")
    col3.metric("Slant Angle", f"{traits['slant_angle']:.2f}°")

    st.subheader("🧠 Predicted Personality Traits")
    try:
        with st.spinner("Running prediction..."):
            padded_rgb = np.stack([padded] * 3, axis=-1).astype(np.float32)
            padded_rgb = tf.keras.applications.efficientnet.preprocess_input(padded_rgb)
            padded_rgb = padded_rgb.reshape(1, 128, 128, 3)
            features = np.array([[traits["ink_density"], traits["aspect_ratio"], traits["slant_angle"]]], dtype=np.float32)
            preds = model.predict([padded_rgb, features], verbose=0)

            predictions_dict = {}
            for i, trait_pred in enumerate(preds):
                pred_class = int(np.argmax(trait_pred[0]))
                confidence = float(trait_pred[0][pred_class])
                trait_name = TRAIT_NAMES[i]
                prediction_text = CLASS_MAP[pred_class]
                predictions_dict[trait_name] = f"{prediction_text} ({confidence:.2f})"
                color_icon = CLASS_COLORS[pred_class]
                st.success(f"{color_icon} **{trait_name}**: {prediction_text} (Confidence: {confidence:.2f})")

            # Optional logging (safe fallback)
            try:
                log_df = pd.read_csv("logs/prediction_log.csv") if os.path.exists("logs/prediction_log.csv") else pd.DataFrame()
                new_log = {"filename": uploaded_file.name}
                new_log.update(traits)
                new_log.update(predictions_dict)
                log_df = pd.concat([log_df, pd.DataFrame([new_log])])
                os.makedirs("logs", exist_ok=True)
                log_df.to_csv("logs/prediction_log.csv", index=False)
                st.info("✅ Prediction logged successfully")
            except Exception as log_error:
                st.warning(f"⚠️ Logging failed: {log_error}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.error("This might be due to model compatibility or loss registration issues.")

# === Sidebar Evaluation Controls ===
st.sidebar.header("📊 Model Evaluation")

# Show Confusion Matrices
if st.sidebar.checkbox("Show Confusion Matrices"):
    st.subheader("📌 Confusion Matrices by Trait")
    cols = st.columns(2)
    for i in range(1, NUM_TRAITS + 1):
        path = os.path.join(CONF_MATRIX_DIR, f"conf_matrix_trait_{i}.png")
        if os.path.exists(path):
            with cols[(i - 1) % 2]:
                st.image(path, caption=f"Trait {i}: {TRAIT_NAMES[i - 1]}", use_column_width=True)
        else:
            st.warning(f"Confusion matrix for trait {i} not found.")

# Show Loss Curve
if st.sidebar.checkbox("Show Training Progress"):
    if os.path.exists(LOSS_CURVE_PATH):
        st.subheader("📉 Training Loss Curve")
        st.image(LOSS_CURVE_PATH, use_column_width=True)
    else:
        st.warning("Loss curve not found. Run training script to generate.")

# Show Recent Prediction Logs
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
        st.info("No prediction log found yet. Make a few predictions to start tracking.")

# === Footer ===
st.markdown("---")
st.markdown("**Signalyze** — Personality prediction through signature analysis using EfficientNetB0 + trait fusion")