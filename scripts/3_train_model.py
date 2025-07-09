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

set_global_policy("mixed_float16")

IMG_FOLDER = "D:/signalyze/signalyze-personality-predictor/data/preprocessed_images"
CSV_PATH = "D:/signalyze/signalyze-personality-predictor/data/form_responses.csv"
FEATURE_CSV = "D:/signalyze/signalyze-personality-predictor/data/signature_traits.csv"
MODEL_PATH = "D:/signalyze/signalyze-personality-predictor/model/best_model.keras"
LOG_PATH = "D:/signalyze/signalyze-personality-predictor/model/training_log.csv"
IMAGE_SIZE = (128, 128)
NUM_TRAITS = 8
NUM_CLASSES = 3
BATCH_SIZE = 32

LIKERT_MAP = {"Strongly Disagree": 0, "Disagree": 0, "Neutral": 1, "Agree": 2, "Strongly Agree": 2}
TRAIT_NAMES = ["Confidence", "Emotional Stability", "Sociability", "Responsiveness",
               "Concentration", "Introversion", "Creativity", "Decision-Making"]

# === Data Loading ===
def load_data():
    df = pd.read_csv(CSV_PATH)
    traits_df = pd.read_csv(FEATURE_CSV)
    files = sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    df, traits_df = df.iloc[:len(files)], traits_df.iloc[:len(files)]
    X, Y_feats, y_outputs = [], [], [[] for _ in range(NUM_TRAITS)]
    for i, fname in enumerate(files):
        img = cv2.imread(os.path.join(IMG_FOLDER, fname), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        img = cv2.resize(img, IMAGE_SIZE)
        img = np.stack([img]*3, axis=-1)
        img = preprocess_input(img.astype(np.float32))
        X.append(img)
        Y_feats.append([
            traits_df.iloc[i]["ink_density"],
            traits_df.iloc[i]["aspect_ratio"],
            traits_df.iloc[i]["slant_angle"]
        ])
        for j in range(NUM_TRAITS):
            val = str(df.iloc[i, -NUM_TRAITS + j]).strip()
            y_outputs[j].append(LIKERT_MAP.get(val, 2))
    return np.array(X), np.array(Y_feats).astype(np.float32), [to_categorical(np.array(y), NUM_CLASSES) for y in y_outputs]

X, Y_feats, y_traits = load_data()
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_feats, test_size=0.15, random_state=42)
y_train, y_val = [], []
for y in y_traits:
    y_t, y_v = train_test_split(y, test_size=0.15, random_state=42)
    y_train.append(y_t)
    y_val.append(y_v)

class MultiInputGenerator(Sequence):
    def __init__(self, X, traits, labels, batch_size=32, shuffle=True):
        self.X, self.traits, self.labels = X, traits, labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.X) / self.batch_size))
    def __getitem__(self, index):
        idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        return {'image_input': self.X[idx], 'trait_input': self.traits[idx]}, {
            f'trait_{i+1}': y[idx] for i, y in enumerate(self.labels)
        }
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

train_gen = MultiInputGenerator(X_train, Y_train, y_train)
val_gen = MultiInputGenerator(X_val, Y_val, y_val, shuffle=False)

trait_weights = {f"trait_{i+1}": tf.Variable(1.0, trainable=False, dtype=tf.float32) for i in range(NUM_TRAITS)}

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=1)
    return loss

def make_weighted_loss(key, base_loss):
    def loss(y_true, y_pred): return trait_weights[key] * base_loss(y_true, y_pred)
    return loss

loss_map = {
    f"trait_{i+1}": make_weighted_loss(
        f"trait_{i+1}", focal_loss() if i in [2, 5, 6] else CategoricalCrossentropy()
    ) for i in range(NUM_TRAITS)
}

image_input = Input(shape=(128, 128, 3), name="image_input")
trait_input = Input(shape=(3,), name="trait_input")
base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=image_input, name="efficientnetb0")
base.trainable = False
x = GlobalAveragePooling2D()(base.output)

traits = Dense(32, activation='relu')(trait_input)
traits = BatchNormalization()(traits)
traits = Dropout(0.3)(traits)

merged = Concatenate()([x, traits])
shared = Dense(128, activation='relu')(merged)
shared = Dropout(0.3)(shared)

outputs = []
for i in range(NUM_TRAITS):
    d = Dropout(0.35 if i in [5, 7] else 0.25)(shared)
    h = Dense(64, activation='relu')(d)
    o = Dense(NUM_CLASSES, activation='softmax', name=f"trait_{i+1}", dtype='float32')(h)
    outputs.append(o)

model = Model(inputs=[image_input, trait_input], outputs=outputs)

class DynamicLossUpdater(Callback):
    def __init__(self, baseline=0.55): self.baseline = baseline
    def on_epoch_end(self, epoch, logs=None):
        for i in range(NUM_TRAITS):
            key = f"trait_{i+1}"
            acc = logs.get(f"val_{key}_accuracy")
            if acc is not None and not np.isnan(acc):
                updated = tf.clip_by_value(1.0 + (self.baseline - acc) * 2, 0.6, 1.8)
                trait_weights[key].assign(updated)

class UnfreezeBackbone(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 10:
            print("\n🔓 Unfreezing EfficientNetB0")
            base.trainable = True

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
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True),
    CSVLogger(LOG_PATH, append=True),
    UnfreezeBackbone(),
    DynamicLossUpdater()
]

model.summary()
print("\n🚀 Training started...")
history = model.fit(train_gen, validation_data=val_gen, epochs=60, callbacks=callbacks, verbose=1)
print(f"\n✅ Model saved to: {MODEL_PATH}")

# === Evaluation ===
y_true_flat = [np.argmax(arr, axis=1) for arr in y_val]
y_pred_raw = model.predict(val_gen)
y_pred_flat = [np.argmax(arr, axis=1) for arr in y_pred_raw]

for i in range(NUM_TRAITS):
    print(f"\n📊 Trait: {TRAIT_NAMES[i]}")
    print(classification_report(y_true_flat[i], y_pred_flat[i], target_names=list(LIKERT_MAP.keys())[:3]))
    cm = confusion_matrix(y_true_flat[i], y_pred_flat[i])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {TRAIT_NAMES[i]}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"conf_matrix_trait_{i+1}.png")
    plt.close()

print("\n🎯 All confusion matrices saved as PNGs in current directory.")

# Training Curve Plot
plt.figure(figsize=(10, 4))
history_df = pd.read_csv(LOG_PATH)
plt.plot(history_df['epoch'], history_df['loss'], label='Train Loss')
plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.close()

print("\n📈 Loss curve saved as loss_curve.png")
