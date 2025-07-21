import streamlit as st

st.set_page_config(page_title="About Project", layout="centered")

st.title("📘 About Signalyze")

st.markdown("""
### 🧠 What is Signalyze?

**Signalyze** is a signature-based personality prediction system that uses deep learning and visual handwriting analysis techniques to provide insights into human personality traits based on signature styles.

The core idea behind this project is that **a person’s handwriting and signature reflect deep psychological and behavioral patterns**, and we can use this data to extract personality indicators through machine learning.

---

### 🔍 How It Works

1. **User Uploads Signature**: The user provides a handwritten signature image via the Streamlit web interface.
2. **Preprocessing & Feature Extraction**:
   - The image is preprocessed (CLAHE, deskewing, resizing).
   - Key visual traits are extracted, such as ink density, aspect ratio, and slant angle.
3. **Model Prediction**:
   - The preprocessed image is fed into a deep learning model trained to predict 5 psychological traits based on the new dataset.
4. **Output**:
   - The model outputs Likert-scale predictions (Agree / Neutral / Disagree) for each personality trait.

---

### 🧪 Traits Predicted

Based on the dataset and psychological literature, the system predicts the following 5 traits from the Big Five Personality Model:

1. Openness
2. Conscientiousness
3. Extraversion
4. Agreeableness
5. Neuroticism

---

### 🛠️ Tech Stack Used

- Python, TensorFlow, Keras
- OpenCV for Image Preprocessing
- Streamlit for Web Interface
- Pandas, NumPy, Matplotlib for data processing & display
- Custom CNN model trained on preprocessed signature images

---

### 📈 Applications

- Behavioral analysis and psychological profiling
- Signature verification & forensics
- Educational / Recruitment personality insights
- Personal self-awareness & growth tools

---

### 🙌 Team & Acknowledgements

Developed as a mini-project using real-time signature data, tools, and open-source platforms. Inspired by the real-world applicability of psychological computing and graphology.

---
""")