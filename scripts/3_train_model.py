# === scripts/3_train_model.py ===
import os, numpy as np, pandas as pd, cv2, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.mixed_precision import set_global_policy

# Add parent directory to path to import custom objects
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.custom_objects import (
    weighted_loss_trait_1, weighted_loss_trait_2, weighted_loss_trait_3, weighted_loss_trait_4,
    weighted_loss_trait_5, weighted_loss_trait_6, weighted_loss_trait_7, weighted_loss_trait_8
)

# Enable mixed precision for faster training
set_global_policy("mixed_float16")

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_FOLDER = os.path.join(BASE_DIR, "data", "preprocessed_images")
CSV_PATH = os.path.join(BASE_DIR, "data", "form_responses.csv")
FEATURE_CSV = os.path.join(BASE_DIR, "data", "signature_traits.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")
LOG_PATH = os.path.join(BASE_DIR, "model", "training_log.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "model")

IMAGE_SIZE = (128, 128)
NUM_TRAITS = 8
NUM_CLASSES = 3
BATCH_SIZE = 32

LIKERT_MAP = {"Strongly Disagree": 0, "Disagree": 0, "Neutral": 1, "Agree": 2, "Strongly Agree": 2}
TRAIT_NAMES = ["Confidence", "Emotional Stability", "Sociability", "Responsiveness",
               "Concentration", "Introversion", "Creativity", "Decision-Making"]

print(f"🔧 Configuration:")
print(f"   Base Directory: {BASE_DIR}")
print(f"   Image Folder: {IMG_FOLDER}")
print(f"   Model Output: {MODEL_PATH}")
print(f"   Training Log: {LOG_PATH}")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Data Loading ===
def load_data():
    """Load and preprocess image and trait data"""
    print("📂 Loading data...")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Survey responses not found: {CSV_PATH}")
    if not os.path.exists(FEATURE_CSV):
        raise FileNotFoundError(f"Visual traits not found: {FEATURE_CSV}")
    if not os.path.exists(IMG_FOLDER):
        raise FileNotFoundError(f"Image folder not found: {IMG_FOLDER}")
    
    df = pd.read_csv(CSV_PATH)
    traits_df = pd.read_csv(FEATURE_CSV)
    files = sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    
    print(f"   Found {len(files)} images")
    print(f"   Survey responses: {len(df)} rows")
    print(f"   Visual traits: {len(traits_df)} rows")
    
    # Align data sizes
    min_size = min(len(files), len(df), len(traits_df))
    df, traits_df, files = df.iloc[:min_size], traits_df.iloc[:min_size], files[:min_size]
    
    X, Y_feats, y_outputs = [], [], [[] for _ in range(NUM_TRAITS)]
    
    print("⚙️  Processing images and extracting features...")
    for i, fname in enumerate(files):
        if i % 50 == 0:
            print(f"   Processing image {i+1}/{len(files)}")
            
        img_path = os.path.join(IMG_FOLDER, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            print(f"   Warning: Could not load {fname}")
            continue
            
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.stack([img]*3, axis=-1)
        img = preprocess_input(img.astype(np.float32))
        X.append(img)
        
        # Extract visual features
        Y_feats.append([
            traits_df.iloc[i]["ink_density"],
            traits_df.iloc[i]["aspect_ratio"],
            traits_df.iloc[i]["slant_angle"]
        ])
        
        # Process personality trait labels
        for j in range(NUM_TRAITS):
            val = str(df.iloc[i, -NUM_TRAITS + j]).strip()
            y_outputs[j].append(LIKERT_MAP.get(val, 2))
    
    print(f"✅ Loaded {len(X)} samples successfully")
    return np.array(X), np.array(Y_feats).astype(np.float32), [to_categorical(np.array(y), NUM_CLASSES) for y in y_outputs]

# Load data
X, Y_feats, y_traits = load_data()

# === Data Splitting ===
print("🔄 Splitting data...")
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_feats, test_size=0.15, random_state=42)
y_train, y_val = [], []
for y in y_traits:
    y_t, y_v = train_test_split(y, test_size=0.15, random_state=42)
    y_train.append(y_t)
    y_val.append(y_v)

print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_val)}")

# === Data Generator ===
class MultiInputGenerator(Sequence):
    """Custom data generator for multi-input model"""
    def __init__(self, X, traits, labels, batch_size=32, shuffle=True):
        self.X, self.traits, self.labels = X, traits, labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
        
    def __len__(self): 
        return int(np.ceil(len(self.X) / self.batch_size))
        
    def __getitem__(self, index):
        idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return {'image_input': self.X[idx], 'trait_input': self.traits[idx]}, {
            f'trait_{i+1}': y[idx] for i, y in enumerate(self.labels)
        }
        
    def on_epoch_end(self):
        if self.shuffle: 
            np.random.shuffle(self.indices)

train_gen = MultiInputGenerator(X_train, Y_train, y_train)
val_gen = MultiInputGenerator(X_val, Y_val, y_val, shuffle=False)

# === Define loss functions using registered custom objects ===
loss_map = {
    "trait_1": weighted_loss_trait_1,
    "trait_2": weighted_loss_trait_2,
    "trait_3": weighted_loss_trait_3,
    "trait_4": weighted_loss_trait_4,
    "trait_5": weighted_loss_trait_5,
    "trait_6": weighted_loss_trait_6,
    "trait_7": weighted_loss_trait_7,
    "trait_8": weighted_loss_trait_8,
}

# === Model Architecture ===
print("🏗️  Building model architecture...")

# Image input branch
image_input = Input(shape=(128, 128, 3), name="image_input")
trait_input = Input(shape=(3,), name="trait_input")

# EfficientNetB0 backbone
base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=image_input, name="efficientnetb0")
base.trainable = False  # Start with frozen backbone
x = GlobalAveragePooling2D()(base.output)

# Visual traits branch
traits = Dense(32, activation='relu')(trait_input)
traits = BatchNormalization()(traits)
traits = Dropout(0.3)(traits)

# Feature fusion
merged = Concatenate()([x, traits])
shared = Dense(128, activation='relu')(merged)
shared = Dropout(0.3)(shared)

# Multi-head outputs
outputs = []
for i in range(NUM_TRAITS):
    d = Dropout(0.35 if i in [5, 7] else 0.25)(shared)
    h = Dense(64, activation='relu')(d)
    o = Dense(NUM_CLASSES, activation='softmax', name=f"trait_{i+1}", dtype='float32')(h)
    outputs.append(o)

model = Model(inputs=[image_input, trait_input], outputs=outputs)

# === Custom Callbacks ===
class DynamicLossUpdater(Callback):
    """Monitor validation performance - simplified without dynamic weights"""
    def __init__(self, baseline=0.55): 
        self.baseline = baseline
        
    def on_epoch_end(self, epoch, logs=None):
        # Log performance for monitoring (weights are now fixed in loss functions)
        for i in range(NUM_TRAITS):
            key = f"trait_{i+1}"
            acc = logs.get(f"val_{key}_accuracy")
            if acc is not None and not np.isnan(acc) and acc < self.baseline:
                print(f"\n   Note: {key} validation accuracy ({acc:.3f}) below baseline ({self.baseline})")

class UnfreezeBackbone(Callback):
    """Unfreeze EfficientNetB0 after initial training"""
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 10:
            print("\n🔓 Unfreezing EfficientNetB0 backbone for fine-tuning")
            base.trainable = True

# === Training Configuration ===
steps = len(train_gen)
lr_schedule = CosineDecayRestarts(
    initial_learning_rate=2e-5,
    first_decay_steps=steps * 8,
    t_mul=2,
    m_mul=1,
    alpha=1e-6
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule, clipnorm=1.0),
    loss=loss_map,
    metrics={f"trait_{i+1}": "accuracy" for i in range(NUM_TRAITS)}
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger(LOG_PATH, append=True),
    UnfreezeBackbone(),
    DynamicLossUpdater()
]

# === Training ===
model.summary()
print(f"\n🚀 Starting training with {len(train_gen)} steps per epoch...")
history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=60, 
    callbacks=callbacks, 
    verbose=1
)
print(f"\n✅ Model saved to: {MODEL_PATH}")

# === Evaluation & Visualization ===
print("\n📊 Generating evaluation metrics...")

y_true_flat = [np.argmax(arr, axis=1) for arr in y_val]
y_pred_raw = model.predict(val_gen, verbose=0)
y_pred_flat = [np.argmax(arr, axis=1) for arr in y_pred_raw]

# Generate confusion matrices and save them
for i in range(NUM_TRAITS):
    print(f"\n� Trait: {TRAIT_NAMES[i]}")
    print(classification_report(y_true_flat[i], y_pred_flat[i], target_names=["Disagree", "Neutral", "Agree"]))
    
    cm = confusion_matrix(y_true_flat[i], y_pred_flat[i])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Disagree", "Neutral", "Agree"], yticklabels=["Disagree", "Neutral", "Agree"])
    plt.title(f"Confusion Matrix: {TRAIT_NAMES[i]}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    # Save to model directory
    cm_path = os.path.join(OUTPUT_DIR, f"conf_matrix_trait_{i+1}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n🎯 All confusion matrices saved to {OUTPUT_DIR}")

# Generate and save training curve
plt.figure(figsize=(12, 4))
try:
    history_df = pd.read_csv(LOG_PATH)
    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['loss'], label='Train Loss', color='blue')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot average accuracy across traits
    plt.subplot(1, 2, 2)
    train_accs = [history_df[f'trait_{i+1}_accuracy'].iloc[-1] for i in range(NUM_TRAITS)]
    val_accs = [history_df[f'val_trait_{i+1}_accuracy'].iloc[-1] for i in range(NUM_TRAITS)]
    
    x = np.arange(len(TRAIT_NAMES))
    plt.bar(x - 0.2, train_accs, 0.4, label='Train Accuracy', color='skyblue')
    plt.bar(x + 0.2, val_accs, 0.4, label='Val Accuracy', color='lightcoral')
    plt.xlabel("Personality Traits")
    plt.ylabel("Accuracy")
    plt.title("Final Accuracy by Trait")
    plt.xticks(x, [name[:10] for name in TRAIT_NAMES], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Training curves saved to {loss_curve_path}")
    
    # Print summary statistics
    print(f"\n� Training Summary:")
    print(f"   Final Training Loss: {history_df['loss'].iloc[-1]:.4f}")
    print(f"   Final Validation Loss: {history_df['val_loss'].iloc[-1]:.4f}")
    print(f"   Best Validation Loss: {history_df['val_loss'].min():.4f}")
    print(f"   Average Train Accuracy: {np.mean(train_accs):.3f}")
    print(f"   Average Val Accuracy: {np.mean(val_accs):.3f}")
    
except Exception as e:
    print(f"⚠️  Could not generate training curves: {e}")

print(f"\n🎉 Training completed successfully!")
print(f"   Model: {MODEL_PATH}")
print(f"   Logs: {LOG_PATH}")
print(f"   Confusion Matrices: {OUTPUT_DIR}/conf_matrix_trait_*.png")
