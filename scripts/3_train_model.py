import os, numpy as np, pandas as pd, cv2, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.mixed_precision import set_global_policy
from keras.saving import register_keras_serializable

set_global_policy("mixed_float16")

# Resolve base path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_FOLDER = os.path.join(DATA_DIR, "preprocessed_images")
CSV_PATH = os.path.join(DATA_DIR, "form_responses.csv")
FEATURE_CSV = os.path.join(DATA_DIR, "signature_traits.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
LOG_PATH = os.path.join(MODEL_DIR, "training_log.csv")

IMAGE_SIZE = (128, 128)
NUM_TRAITS = 8
NUM_CLASSES = 3
BATCH_SIZE = 32

LIKERT_MAP = {"Strongly Disagree": 0, "Disagree": 0, "Neutral": 1, "Agree": 2, "Strongly Agree": 2}
TRAIT_NAMES = [
    "Confidence", "Emotional Stability", "Sociability", "Responsiveness",
    "Concentration", "Introversion", "Creativity", "Decision-Making"
]

def load_data():
    """Load and preprocess images, traits, and survey responses"""
    print("📂 Loading data...")

    df = pd.read_csv(CSV_PATH)
    traits_df = pd.read_csv(FEATURE_CSV)
    files = sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # Align datasets by smallest length
    min_size = min(len(files), len(df), len(traits_df))
    df, traits_df, files = df.iloc[:min_size], traits_df.iloc[:min_size], files[:min_size]

    X, Y_feats, y_outputs = [], [], [[] for _ in range(NUM_TRAITS)]

    for i, fname in enumerate(files):
        img_path = os.path.join(IMG_FOLDER, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.stack([img] * 3, axis=-1)
        img = preprocess_input(img.astype(np.float32))
        X.append(img)

        Y_feats.append([
            traits_df.iloc[i]["ink_density"],
            traits_df.iloc[i]["aspect_ratio"],
            traits_df.iloc[i]["slant_angle"]
        ])

        for j in range(NUM_TRAITS):
            val = str(df.iloc[i, -NUM_TRAITS + j]).strip()
            y_outputs[j].append(LIKERT_MAP.get(val, 2))  # fallback: Agree → class 2

    print(f"✅ Loaded {len(X)} samples")
    return np.array(X), np.array(Y_feats).astype(np.float32), [
        to_categorical(np.array(y), NUM_CLASSES) for y in y_outputs
    ]

X, Y_feats, y_traits = load_data()

print("🔀 Splitting train and validation sets...")
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_feats, test_size=0.15, random_state=42)

y_train, y_val = [], []
for y in y_traits:
    yt, yv = train_test_split(y, test_size=0.15, random_state=42)
    y_train.append(yt)
    y_val.append(yv)

class MultiInputGenerator(Sequence):
    def __init__(self, X, traits, labels, batch_size=32, shuffle=True):
        self.X = X
        self.traits = traits
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return {
            'image_input': self.X[indices],
            'trait_input': self.traits[indices]
        }, {
            f'trait_{i+1}': y[indices] for i, y in enumerate(self.labels)
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_gen = MultiInputGenerator(X_train, Y_train, y_train)
val_gen = MultiInputGenerator(X_val, Y_val, y_val, shuffle=False)

# === Trait Weights for Dynamic Loss Adjustment ===
trait_weights = {
    f"trait_{i+1}": tf.Variable(1.0, trainable=False, dtype=tf.float32)
    for i in range(NUM_TRAITS)
}

# === Focal Loss Function (Registered) ===
@register_keras_serializable()
def focal_loss_inner(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    ce = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_sum(weight * ce, axis=1)

# === Create Named and Registered Weighted Losses ===
def make_registered_loss(trait_name, base_loss):
    @register_keras_serializable(name=f"weighted_loss_{trait_name}")
    def trait_loss(y_true, y_pred):
        return trait_weights[trait_name] * base_loss(y_true, y_pred)
    trait_loss.__name__ = f"weighted_loss_{trait_name}"
    return trait_loss

# === Assign Loss Functions to Each Trait ===
loss_map = {}
for i in range(NUM_TRAITS):
    trait_key = f"trait_{i+1}"
    use_focal = i in [2, 5, 6]  # Traits that benefit from focal loss
    base = focal_loss_inner if use_focal else CategoricalCrossentropy()
    loss_map[trait_key] = make_registered_loss(trait_key, base)


print("🏗️ Building model architecture...")

# Inputs
image_input = Input(shape=(128, 128, 3), name="image_input")
trait_input = Input(shape=(3,), name="trait_input")

# EfficientNetB0 Backbone
base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=image_input)
base.trainable = False
x = GlobalAveragePooling2D()(base.output)

# Trait Features Path
traits = Dense(32, activation='relu')(trait_input)
traits = BatchNormalization()(traits)
traits = Dropout(0.3)(traits)

# Merge Vision + Trait Streams
merged = Concatenate()([x, traits])
shared = Dense(128, activation='relu')(merged)
shared = Dropout(0.3)(shared)

# Trait-Specific Output Heads
outputs = []
for i in range(NUM_TRAITS):
    drop_rate = 0.35 if i in [5, 7] else 0.25
    d = Dropout(drop_rate)(shared)
    h = Dense(64, activation='relu')(d)
    o = Dense(NUM_CLASSES, activation='softmax', name=f"trait_{i+1}", dtype='float32')(h)
    outputs.append(o)

# Compile Model
model = Model(inputs=[image_input, trait_input], outputs=outputs)

class DynamicLossUpdater(Callback):
    """Adjust trait-wise loss weights based on validation accuracy"""
    def __init__(self, baseline=0.55): 
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        for i in range(NUM_TRAITS):
            key = f"trait_{i+1}"
            acc = logs.get(f"val_{key}_accuracy")
            if acc is not None and not np.isnan(acc):
                new_weight = tf.clip_by_value(1.0 + (self.baseline - acc) * 2, 0.6, 1.8)
                trait_weights[key].assign(new_weight)

class UnfreezeBackbone(Callback):
    """Unfreeze EfficientNetB0 after warmup"""
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 10:
            print("\n🔓 Unfreezing EfficientNetB0 layers")
            model.get_layer(index=2).trainable = True

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

model.summary()
print(f"\n🚀 Starting training with {len(train_gen)} batches per epoch...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=60,
    callbacks=callbacks,
    verbose=1
)

# Save and reload to verify serialization
model.save(MODEL_PATH)
model = load_model(MODEL_PATH)

print("\n📊 Evaluating model...")

y_true = [np.argmax(y, axis=1) for y in y_val]
y_pred_raw = model.predict(val_gen, verbose=0)
y_pred = [np.argmax(y, axis=1) for y in y_pred_raw]

for i in range(NUM_TRAITS):
    print(f"\n🧠 Trait: {TRAIT_NAMES[i]}")
    print(classification_report(y_true[i], y_pred[i], target_names=["Disagree", "Neutral", "Agree"]))
    
    cm = confusion_matrix(y_true[i], y_pred[i])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Disagree", "Neutral", "Agree"],
                yticklabels=["Disagree", "Neutral", "Agree"])
    plt.title(f"Confusion Matrix: {TRAIT_NAMES[i]}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"conf_matrix_trait_{i+1}.png"), dpi=300)
    plt.close()

try:
    history_df = pd.read_csv(LOG_PATH)
    plt.figure(figsize=(12, 4))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()

    # Accuracy Bar Plot
    plt.subplot(1, 2, 2)
    train_accs = [history_df[f"trait_{i+1}_accuracy"].iloc[-1] for i in range(NUM_TRAITS)]
    val_accs = [history_df[f"val_trait_{i+1}_accuracy"].iloc[-1] for i in range(NUM_TRAITS)]
    x = np.arange(NUM_TRAITS)
    plt.bar(x - 0.2, train_accs, 0.4, label="Train", color="skyblue")
    plt.bar(x + 0.2, val_accs, 0.4, label="Val", color="salmon")
    plt.xticks(x, TRAIT_NAMES, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Final Accuracy by Trait")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "loss_curve.png"), dpi=300)
    plt.close()

    print(f"\n🎯 Final Metrics:")
    print(f"   Final Train Loss: {history_df['loss'].iloc[-1]:.4f}")
    print(f"   Final Val Loss:   {history_df['val_loss'].iloc[-1]:.4f}")
    print(f"   Best Val Loss:    {history_df['val_loss'].min():.4f}")
    print(f"   Avg Train Accuracy: {np.mean(train_accs):.3f}")
    print(f"   Avg Val Accuracy:   {np.mean(val_accs):.3f}")
except Exception as e:
    print(f"⚠️ Visualization failed: {e}")

print("\n✅ Training complete and model saved.")