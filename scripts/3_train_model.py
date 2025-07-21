# === train_model.py ===

import os, numpy as np, pandas as pd, cv2, tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.keras.mixed_precision.set_global_policy("float32")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
IMG_FOLDER = os.path.join(DATA_DIR, "preprocessed_images")
MASTER_CSV_PATH = os.path.join(DATA_DIR, "handwriting_personality_large_dataset.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
LOG_PATH = os.path.join(MODEL_DIR, "training_log.csv")

IMAGE_SIZE = (64, 64)
NUM_TRAITS = 5
NUM_CLASSES = 3
BATCH_SIZE = 32
TRAIT_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
PLOT_LABELS = ["Disagree", "Neutral", "Agree"]
EARLY_STOP_PATIENCE = 30
REDUCE_LR_PATIENCE = 15

def map_to_categorical(value):
    if value <= 0.33: return 0
    elif value <= 0.66: return 1
    else: return 2

def load_data():
    df = pd.read_csv(MASTER_CSV_PATH)
    files = sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    df.set_index("Handwriting_Sample", inplace=True)
    df = df.loc[files]
    X, y_outputs = [], [[] for _ in range(NUM_TRAITS)]
    for fname, row in df.iterrows():
        img = cv2.imread(os.path.join(IMG_FOLDER, fname), cv2.IMREAD_COLOR)
        if img is None: continue
        if len(img.shape) == 2: img = np.stack([img]*3, axis=-1)
        img = cv2.resize(img.astype(np.float32) / 255.0, IMAGE_SIZE)
        X.append(img)
        for j in range(NUM_TRAITS):
            y_outputs[j].append(map_to_categorical(row[TRAIT_NAMES[j]]))
    y_encoded = [to_categorical(np.array(y), num_classes=NUM_CLASSES) for y in y_outputs]
    return np.array(X), y_encoded

X, y_traits = load_data()

for i, trait in enumerate(TRAIT_NAMES):
    counts = np.bincount(np.argmax(y_traits[i], axis=1))
    print(f"{trait} distribution:", dict(zip(PLOT_LABELS, counts)))

split_key = np.argmax(y_traits[0], axis=1)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(sss.split(X, split_key))
X_train, X_val = X[train_idx], X[val_idx]
y_train = [y[train_idx] for y in y_traits]
y_val = [y[val_idx] for y in y_traits]

datagen = ImageDataGenerator(rotation_range=8, zoom_range=0.08, width_shift_range=0.06, height_shift_range=0.06, brightness_range=(0.9, 1.1))

class MultiOutputGenerator(Sequence):
    def __init__(self, X, labels, batch_size=32, shuffle=True, augment=False):
        self.X, self.labels = X, labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.X) / self.batch_size))
    def __getitem__(self, idx):
        idxs = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[idxs].copy()
        if self.augment:
            X_batch = np.stack([datagen.random_transform(img) for img in X_batch])
        return {"image_input": X_batch}, {f"trait_{i+1}": y[idxs] for i, y in enumerate(self.labels)}
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indices)

def build_model():
    input_img = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image_input")
    x = Conv2D(64, 3, activation="relu", kernel_regularizer=regularizers.l2(0.005))(input_img)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, activation="relu", kernel_regularizer=regularizers.l2(0.005))(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, 3, activation="relu", kernel_regularizer=regularizers.l2(0.005))(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, 3, activation="relu", kernel_regularizer=regularizers.l2(0.005))(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = []
    for i in range(NUM_TRAITS):
        t = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
        t = Dropout(0.3)(t)
        outputs.append(Dense(NUM_CLASSES, activation="softmax", name=f"trait_{i+1}")(t))
    return Model(inputs=input_img, outputs=outputs)

model = build_model()

trait_weights = {}
for i in range(NUM_TRAITS):
    y_labels = np.argmax(y_train[i], axis=1)
    weights = compute_class_weight(class_weight='balanced', classes=np.arange(NUM_CLASSES), y=y_labels)
    trait_weights[f"trait_{i+1}"] = dict(zip(range(NUM_CLASSES), weights))

def get_weighted_loss(trait_idx):
    name = f"weighted_loss_trait_{trait_idx}"
    weights = trait_weights[f"trait_{trait_idx}"]
    @register_keras_serializable(name=name)
    def weighted_loss(y_true, y_pred):
        y_true_label = tf.argmax(y_true, axis=1)
        sample_weight = tf.gather(tf.constant([weights.get(i, 1.0) for i in range(NUM_CLASSES)], dtype=tf.float32), y_true_label)
        base_loss = CategoricalCrossentropy(label_smoothing=0.1)(y_true, y_pred)
        return tf.reduce_mean(base_loss * sample_weight)
    return weighted_loss

loss_map = {f"trait_{i+1}": get_weighted_loss(i+1) for i in range(NUM_TRAITS)}
metrics_map = {f"trait_{i+1}": "accuracy" for i in range(NUM_TRAITS)}
model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_map, metrics=metrics_map)

# === Filter low-confidence training samples ===
# === Filter low-confidence training samples ===
# === Filter low-confidence training samples with fallback ===
filter_model = build_model()
filter_model.compile(optimizer=Adam(), loss=loss_map, metrics=metrics_map)
filter_model.load_weights(MODEL_PATH)

print("\n🔍 Running confidence-based filtering using saved model...")
raw_preds = filter_model.predict(X_train, verbose=0)
threshold = 0.5
filtered_mask = np.ones(len(X_train), dtype=bool)

for i in range(NUM_TRAITS):
    confidences = np.max(raw_preds[i], axis=1)
    filtered_mask &= (confidences > threshold)

num_kept = np.sum(filtered_mask)
print(f"🔎 Filtered training samples retained: {num_kept} / {len(X_train)}")

# === Fallback if filtering wipes too many samples ===
MIN_KEEP = 100
if num_kept < MIN_KEEP:
    print(f"⚠️ Too few samples retained. Skipping filtering and restoring full training set.")
else:
    X_train = X_train[filtered_mask]
    y_train = [y[filtered_mask] for y in y_train]
    print(f"✅ Proceeding with filtered dataset of size: {X_train.shape[0]}")

train_gen = MultiOutputGenerator(X_train, y_train, augment=True)
val_gen = MultiOutputGenerator(X_val, y_val, shuffle=False)

# === Training ===
callbacks = [
    EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    CSVLogger(LOG_PATH, append=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=REDUCE_LR_PATIENCE, min_lr=1e-6, verbose=1),
]

model.fit(train_gen, validation_data=val_gen, epochs=500, callbacks=callbacks, verbose=1)

# === Add fallback loss registration for reload ===
@register_keras_serializable(name="weighted_loss")
def weighted_loss(y_true, y_pred):
    return CategoricalCrossentropy(label_smoothing=0.1)(y_true, y_pred)

custom_objects = {
    "weighted_loss": weighted_loss,
    **{f"weighted_loss_trait_{i+1}": loss_map[f"trait_{i+1}"] for i in range(NUM_TRAITS)}
}

model = load_model(MODEL_PATH, custom_objects=custom_objects)

# === Generate Evaluation Report ===
print("\n📊 Enhanced Evaluation Report")
y_true = [np.argmax(y, axis=1) for y in y_val]
y_pred_raw = model.predict(val_gen, verbose=0)
y_pred = [np.argmax(y, axis=1) for y in y_pred_raw]

report_data = []
for i in range(NUM_TRAITS):
    print(f"\n🧠 {TRAIT_NAMES[i]}")
    report_string = classification_report(y_true[i], y_pred[i], zero_division=0)
    print(report_string)

    f1 = f1_score(y_true[i], y_pred[i], average='macro', zero_division=0)
    prec = precision_score(y_true[i], y_pred[i], average='macro', zero_division=0)
    rec = recall_score(y_true[i], y_pred[i], average='macro', zero_division=0)
    acc = np.mean(np.array(y_true[i]) == np.array(y_pred[i]))
    report_data.append([TRAIT_NAMES[i], f1, prec, rec, acc])

    print(f"✅ Accuracy for {TRAIT_NAMES[i]}: {acc:.2%}")

    cm = confusion_matrix(y_true[i], y_pred[i], labels=np.arange(NUM_CLASSES))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=PLOT_LABELS)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {TRAIT_NAMES[i]}')
    plt.savefig(os.path.join(MODEL_DIR, f"conf_matrix_trait_{i+1}.png"))
    plt.close()
    print(f"✅ Saved confusion matrix for {TRAIT_NAMES[i]}")

# === Save Report ===
report_df = pd.DataFrame(report_data, columns=['Trait', 'F1 Score', 'Precision', 'Recall', 'Accuracy'])
report_df.to_csv(os.path.join(MODEL_DIR, 'metrics_report.csv'), index=False)

# === Overall Accuracy ===
total_correct = sum(np.sum(y_true[i] == y_pred[i]) for i in range(NUM_TRAITS))
total_preds = sum(len(y_true[i]) for i in range(NUM_TRAITS))
overall_accuracy = total_correct / total_preds
print(f"\n🔎 Overall combined accuracy across all traits: {overall_accuracy:.2%}")

# === Visualize Trait Metrics ===
plt.figure(figsize=(8, 5))
bar_width = 0.2
index = np.arange(NUM_TRAITS)
plt.bar(index, report_df['F1 Score'], bar_width, label='F1 Score')
plt.bar(index + bar_width, report_df['Precision'], bar_width, label='Precision')
plt.bar(index + 2 * bar_width, report_df['Recall'], bar_width, label='Recall')
plt.bar(index + 3 * bar_width, report_df['Accuracy'], bar_width, label='Accuracy')
plt.xticks(index + 1.5 * bar_width, TRAIT_NAMES, rotation=45)
plt.ylabel("Score")
plt.title("Trait Evaluation Metrics")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "f1_accuracy_chart.png"), dpi=300)
plt.close()

print("\n✅ Model training, saving, and evaluation completed successfully!")