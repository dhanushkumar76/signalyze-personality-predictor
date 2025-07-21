# === evaluate_model.py ===

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.saving import register_keras_serializable

# === Paths & Config ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")
MASTER_CSV_PATH = os.path.join(BASE_DIR, "data", "handwriting_personality_large_dataset.csv")
IMG_FOLDER = os.path.join(BASE_DIR, "data", "preprocessed_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "model", "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = (64, 64)
NUM_TRAITS = 5
NUM_CLASSES = 3
TRAIT_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
PLOT_LABELS = ["Disagree", "Neutral", "Agree"]

# === Register custom losses ===
@register_keras_serializable(name="weighted_loss")
def weighted_loss(y_true, y_pred):
    return CategoricalCrossentropy(label_smoothing=0.1)(y_true, y_pred)

for i in range(1, NUM_TRAITS + 1):
    @register_keras_serializable(name=f"weighted_loss_trait_{i}")
    def trait_loss(y_true, y_pred):
        return CategoricalCrossentropy(label_smoothing=0.1)(y_true, y_pred)

custom_objects = {
    "weighted_loss": weighted_loss,
    **{f"weighted_loss_trait_{i+1}": trait_loss for i in range(NUM_TRAITS)}
}

# === Map continuous traits to categorical ===
def map_to_categorical(value):
    if value <= 0.33: return 0
    elif value <= 0.66: return 1
    else: return 2

# === Load dataset ===
def load_data():
    df = pd.read_csv(MASTER_CSV_PATH)
    files = sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    df.set_index("Handwriting_Sample", inplace=True)
    df = df.loc[files]
    X, y_outputs = [], [[] for _ in range(NUM_TRAITS)]
    for fname, row in df.iterrows():
        img_path = os.path.join(IMG_FOLDER, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None: continue
        if len(img.shape) == 2: img = np.stack([img]*3, axis=-1)
        img = cv2.resize(img.astype(np.float32) / 255.0, IMAGE_SIZE)
        X.append(img)
        for j in range(NUM_TRAITS):
            y_outputs[j].append(map_to_categorical(row[TRAIT_NAMES[j]]))
    y_traits = [to_categorical(np.array(y), num_classes=NUM_CLASSES) for y in y_outputs]
    return np.array(X), y_traits

# === Generator for evaluation ===
class MultiOutputGenerator(Sequence):
    def __init__(self, X, labels, batch_size=32, shuffle=True):
        self.X, self.labels = X, labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.X) / self.batch_size))
    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[indices].copy()
        return {"image_input": X_batch}, {f"trait_{i+1}": y[indices] for i, y in enumerate(self.labels)}
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

# === Evaluation ===
def evaluate_model(model, X_test, y_test):
    print("\n🚀 Running Predictions...")
    preds_raw = model.predict(X_test, verbose=1)
    y_true = [np.argmax(y, axis=1) for y in y_test]
    y_pred = [np.argmax(p, axis=1) for p in preds_raw]

    report_data = []
    for i in range(NUM_TRAITS):
        trait = TRAIT_NAMES[i]
        f1 = f1_score(y_true[i], y_pred[i], average='macro', zero_division=0)
        prec = precision_score(y_true[i], y_pred[i], average='macro', zero_division=0)
        rec = recall_score(y_true[i], y_pred[i], average='macro', zero_division=0)
        acc = np.mean(y_true[i] == y_pred[i])
        report_data.append([trait, f1, prec, rec, acc])
        print(f"\n🧠 {trait} Report\n{classification_report(y_true[i], y_pred[i], zero_division=0)}")
        print(f"✅ Accuracy: {acc:.2%}")

        cm = confusion_matrix(y_true[i], y_pred[i], labels=np.arange(NUM_CLASSES))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=PLOT_LABELS)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {trait}')
        plt.savefig(os.path.join(OUTPUT_DIR, f"conf_matrix_{trait}.png"))
        plt.close()

    df_report = pd.DataFrame(report_data, columns=["Trait", "F1 Score", "Precision", "Recall", "Accuracy"])
    df_report.to_csv(os.path.join(OUTPUT_DIR, "metrics_report.csv"), index=False)

    correct_total = sum(np.sum(y_true[i] == y_pred[i]) for i in range(NUM_TRAITS))
    total_preds = sum(len(y_true[i]) for i in range(NUM_TRAITS))
    overall_acc = correct_total / total_preds
    print(f"\n🔎 Overall Accuracy Across Traits: {overall_acc:.2%}")

    plt.figure(figsize=(8, 5))
    idx = np.arange(NUM_TRAITS)
    width = 0.2
    plt.bar(idx, df_report["F1 Score"], width, label="F1")
    plt.bar(idx + width, df_report["Precision"], width, label="Precision")
    plt.bar(idx + 2*width, df_report["Recall"], width, label="Recall")
    plt.bar(idx + 3*width, df_report["Accuracy"], width, label="Accuracy")
    plt.xticks(idx + 1.5 * width, TRAIT_NAMES, rotation=45)
    plt.ylabel("Score")
    plt.title("Trait Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "f1_accuracy_chart.png"), dpi=300)
    plt.close()
    print("✅ Evaluation visuals saved successfully.")

# === Entry Point ===
def main():
    print("🚀 Launching Signalyze Evaluation")
    print("=" * 50)
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model loaded successfully.")
    X, y = load_data()
    split_key = np.argmax(y[0], axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    _, val_idx = next(sss.split(X, split_key))
    X_test = X[val_idx]
    y_test = [y[i][val_idx] for i in range(NUM_TRAITS)]
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()