import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Visual Traits Explainer", layout="wide")
st.title("📊 Visual Traits Explainer")
st.markdown("""
This section explains the key visual traits extracted from signature images.  
These traits are used by our model to predict personality insights.
""")

# --- Trait Definitions ---
st.header("🧬 Trait Definitions")

trait_info = {
    "Ink Density": "Proportion of inked pixels (dark areas) in the signature. A denser signature may indicate strong expression or assertiveness.",
    "Aspect Ratio": "The height-to-width ratio of the signature. Taller signatures may reflect ambition or confidence; wider ones may suggest openness.",
    "Slant Angle": "The tilt of the signature letters. Right slant may indicate expressiveness or openness; left slant may reflect reserved nature; no slant may suggest logical and balanced personality."
}

for trait, explanation in trait_info.items():
    st.subheader(trait)
    st.write(explanation)

# --- Visual Demos ---
st.header("🔍 Visual Examples")

def create_demo_image(trait):
    img = np.ones((128, 128), dtype=np.uint8) * 255

    if trait == "Ink Density":
        cv2.putText(img, "ThickSig", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0), 5)
    elif trait == "Aspect Ratio":
        cv2.putText(img, "Tall", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0), 3)
    elif trait == "Slant Angle":
        pts = np.array([[20, 30], [30, 80], [40, 30], [50, 80], [60, 30]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, (0), 5)

    return img

cols = st.columns(3)
for i, trait in enumerate(["Ink Density", "Aspect Ratio", "Slant Angle"]):
    with cols[i]:
        st.image(create_demo_image(trait), caption=f"Example: {trait}", use_column_width=True)

# --- Optional Upload ---
st.header("🧪 Try Your Own Signature")
st.markdown("Upload a signature to visualize its extracted traits.")

file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Original Signature", use_column_width=True)

    # Simple CLAHE and padding (no deskew for now)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    h, w = img.shape
    scale = min(128/h, 128/w)
    resized = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    pad_v = (128 - resized.shape[0]) // 2
    pad_h = (128 - resized.shape[1]) // 2
    padded = cv2.copyMakeBorder(resized, pad_v, 128-resized.shape[0]-pad_v,
                                pad_h, 128-resized.shape[1]-pad_h,
                                borderType=cv2.BORDER_CONSTANT, value=255)

    st.image(padded, caption="Preprocessed", use_column_width=True)

    # Compute traits
    binary = (padded < 200).astype(np.uint8)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        y, x = coords[:, 0], coords[:, 1]
        height = y.max() - y.min()
        width = x.max() - x.min()
        ink_density = np.sum(binary) / (128 * 128)
        aspect_ratio = height / width if width > 0 else 1.0

        centered = coords - np.mean(coords, axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        slant = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    else:
        ink_density = 0
        aspect_ratio = 1.0
        slant = 0

    st.markdown("### ✨ Extracted Visual Traits")
    st.write(f"- **Ink Density**: {ink_density:.4f}")
    st.write(f"- **Aspect Ratio**: {aspect_ratio:.4f}")
    st.write(f"- **Slant Angle**: {slant:.2f}°")

