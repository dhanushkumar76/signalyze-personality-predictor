# === scripts/evaluate_model.py ===

"""
Comprehensive model evaluation script for Signalyze personality predictor.
This script ensures the data preprocessing and model loading are identical
to the training pipeline for a fair and reproducible evaluation.
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
# FIX: Added ConfusionMatrixDisplay to the import statement
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.saving import register_keras_serializable

# --- Configuration (MUST match 3_train_model.py) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")
CSV_PATH = os.path.join(BASE_DIR, "data", "form_responses.csv")
FEATURE_CSV = os.path.join(BASE_DIR, "data", "signature_traits.csv")
IMG_FOLDER = os.path.join(BASE_DIR, "data", "preprocessed_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "evaluation_results")

# This must be the same as the training script
IMAGE_SIZE = (64, 64)
NUM_TRAITS = 4
NUM_CLASSES = 3
LIKERT_MAP = {"Strongly Disagree": 0, "Disagree": 0, "Neutral": 1, "Agree": 2, "Strongly Agree": 2}
TRAIT_NAMES = ["Confidence", "Emotional Stability", "Creativity", "Decision-Making"]
PLOT_LABELS = ["Disagree", "Neutral", "Agree"]
# The same random state is CRITICAL for reproducibility
RANDOM_STATE = 42

# --- Custom loss functions for model loading (MUST match 3_train_model.py) ---
# This section must be identical to your training script to load the model correctly.
trait_weights = {}

@register_keras_serializable()
def focal_loss_inner(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(weight * ce, axis=1)

# FIX: Define top-level, serializable loss functions for each trait
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

def load_evaluation_data():
    """Loads and preprocesses data, recreating the exact test set from training."""
    print("📂 Loading and splitting data to recreate the test set...")
    
    df = pd.read_csv(CSV_PATH)
    traits_df = pd.read_csv(FEATURE_CSV)
    files = sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    min_size = min(len(files), len(df), len(traits_df))
    df, traits_df, files = df.iloc[:min_size], traits_df.iloc[:min_size], files[:min_size]
    
    X, Y_feats, y_outputs = [], [], [[] for _ in range(NUM_TRAITS)]
    trait_indices = [0, 1, 6, 7]

    print("⚙️  Processing images and labels...")
    for i, fname in enumerate(files):
        img_path = os.path.join(IMG_FOLDER, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        trait_vals, valid = [], True
        for j, trait_col in enumerate(trait_indices):
            val = str(df.iloc[i, -8 + trait_col]).strip()
            if val not in LIKERT_MAP:
                valid = False
                break
            trait_vals.append(LIKERT_MAP[val])
        if not valid:
            continue
        
        # FIX: Ensure image size and normalization match training
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.stack([img] * 3, axis=-1)
        img = img.astype(np.float32) / 255.0
        
        X.append(img)
        Y_feats.append([
            traits_df.iloc[i]["ink_density"],
            traits_df.iloc[i]["aspect_ratio"],
            traits_df.iloc[i]["slant_angle"]
        ])
        for j in range(NUM_TRAITS):
            y_outputs[j].append(to_categorical(trait_vals[j], NUM_CLASSES))
            
    X_full, Y_full = np.array(X), np.array(Y_feats)
    y_full = [np.array(y) for y in y_outputs]
    
    # FIX: Recreate the same stratified split as in the training script
    # The split key must be the label used for stratification (Confidence trait)
    split_key = np.argmax(y_full[0], axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss.split(X_full, split_key))
    
    X_test, Y_test = X_full[val_idx], Y_full[val_idx]
    y_test = [y[val_idx] for y in y_full]
    
    print(f"✅ Loaded {len(X_full)} samples. Test set size: {len(X_test)} samples.")
    
    return X_test, Y_test, y_test


def evaluate_model(model, X_test, Y_test, y_test):
    """Predicts on test data and generates a detailed report."""
    print("\n🧠 Making predictions...")
    predictions = model.predict([X_test, Y_test], verbose=1)

    # Convert true and predicted labels to integer format for metrics
    y_true_int = [np.argmax(y, axis=1) for y in y_test]
    y_pred_int = [np.argmax(y, axis=1) for y in predictions]
    
    print("\n📊 Enhanced Evaluation Report")
    
    for i in range(NUM_TRAITS):
        trait_name = TRAIT_NAMES[i]
        true_labels = y_true_int[i]
        pred_labels = y_pred_int[i]
        
        print(f"\n🧠 {trait_name}")
        report_string = classification_report(true_labels, pred_labels, zero_division=0)
        print(report_string)
        
        # Plot and save confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(NUM_CLASSES))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=PLOT_LABELS)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {trait_name}')
        plt.savefig(os.path.join(OUTPUT_DIR, f"conf_matrix_{trait_name}.png"))
        plt.close()
        print(f"✅ Saved confusion matrix for {trait_name}")
        
    print("\n✅ Evaluation completed successfully!")


def main():
    print("🚀 Starting Signalyze Model Evaluation")
    print("="*50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("   Please train the model first using scripts/3_train_model.py")
        return
    
    try:
        print(f"🧠 Loading model from: {MODEL_PATH}")
        custom_objects = {
            "weighted_loss_trait_1": weighted_loss_trait_1,
            "weighted_loss_trait_2": weighted_loss_trait_2,
            "weighted_loss_trait_3": weighted_loss_trait_3,
            "weighted_loss_trait_4": weighted_loss_trait_4,
            'focal_loss_inner': focal_loss_inner
        }
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    X_test, Y_test, y_test = load_evaluation_data()

    evaluate_model(model, X_test, Y_test, y_test)
    
    print("\n✅ Evaluation pipeline finished. See results in the 'evaluation_results' folder.")

if __name__ == "__main__":
    main()