# --- Imports ---
import os, numpy as np, pandas as pd, cv2, tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable
from tensorflow.keras import regularizers
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("mixed_float16")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
IMG_FOLDER = os.path.join(DATA_DIR, "preprocessed_images")
CSV_PATH = os.path.join(DATA_DIR, "form_responses.csv")
FEATURE_CSV = os.path.join(DATA_DIR, "signature_traits.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
LOG_PATH = os.path.join(MODEL_DIR, "training_log.csv")

# --- Constants ---
IMAGE_SIZE = (128, 128)
NUM_TRAITS = 4
NUM_CLASSES = 3
BATCH_SIZE = 32
LIKERT_MAP = {"Strongly Disagree": 0, "Disagree": 0, "Neutral": 1, "Agree": 2, "Strongly Agree": 2}
TRAIT_NAMES = ["Confidence", "Emotional Stability", "Creativity", "Decision-Making"]

# --- Load Data ---
def load_data():
    df = pd.read_csv(CSV_PATH)
    traits_df = pd.read_csv(FEATURE_CSV)
    files = sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    min_size = min(len(files), len(df), len(traits_df))
    df, traits_df, files = df.iloc[:min_size], traits_df.iloc[:min_size], files[:min_size]

    X, Y_feats, y_outputs = [], [], [[] for _ in range(NUM_TRAITS)]
    trait_indices = [0, 1, 6, 7]
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

        for j, trait_col in enumerate(trait_indices):
            val = str(df.iloc[i, -8 + trait_col]).strip()
            y_outputs[j].append(LIKERT_MAP.get(val, 2))

    return np.array(X), np.array(Y_feats).astype(np.float32), [
        to_categorical(np.array(y), NUM_CLASSES) for y in y_outputs
    ]

X, Y_feats, y_traits = load_data()

# --- Train-Test Split ---
trait0_labels = np.argmax(y_traits[0], axis=1)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(sss.split(X, trait0_labels))
X_train, X_val = X[train_idx], X[val_idx]
Y_train, Y_val = Y_feats[train_idx], Y_feats[val_idx]
y_train, y_val = [], []
for y in y_traits:
    y_train.append(y[train_idx])
    y_val.append(y[val_idx])

# --- Data Augmentation ---
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_images(images):
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.15, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest'
    )
    return np.array([datagen.random_transform(img.copy()) for img in images])

class MultiInputGenerator(Sequence):
    def __init__(self, X, traits, labels, batch_size=32, shuffle=True, augment=False):
        self.X = X
        self.traits = traits
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = self.X[indices].copy()
        traits_batch = self.traits[indices].copy()
        if self.augment:
            X_batch = augment_images(X_batch)
        return {
            'image_input': X_batch,
            'trait_input': traits_batch
        }, {
            f'trait_{i+1}': y[indices] for i, y in enumerate(self.labels)
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

train_gen = MultiInputGenerator(X_train, Y_train, y_train, augment=True)
val_gen = MultiInputGenerator(X_val, Y_val, y_val, shuffle=False, augment=False)

# --- Model ---
image_input = Input(shape=(128, 128, 3), name="image_input")
trait_input = Input(shape=(3,), name="trait_input")

base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=image_input)
base.trainable = True
x = GlobalAveragePooling2D()(base.output)

traits = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(trait_input)
traits = BatchNormalization()(traits)
traits = Dropout(0.5)(traits)

merged = Concatenate()([x, traits])
shared = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(merged)
shared = Dropout(0.5)(shared)

outputs = []
for i in range(NUM_TRAITS):
    h = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(shared)
    h = Dropout(0.5)(h)
    o = Dense(NUM_CLASSES, activation='softmax', name=f"trait_{i+1}", dtype='float32')(h)
    outputs.append(o)

model = Model(inputs=[image_input, trait_input], outputs=outputs)

# --- Loss Functions ---
trait_weights = {
    f"trait_{i+1}": tf.Variable(1.0, trainable=False, dtype=tf.float32)
    for i in range(NUM_TRAITS)
}

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
    use_focal = i == 2  # Creativity gets focal loss
    base = focal_loss_inner if use_focal else CategoricalCrossentropy()
    loss_map[trait_key] = make_registered_loss(trait_key, base)

model.compile(
    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=loss_map,
    metrics={f"trait_{i+1}": "accuracy" for i in range(NUM_TRAITS)}
)

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    CSVLogger(LOG_PATH, append=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# --- Training ---
model.summary()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=60,
    callbacks=callbacks,
    verbose=1
)

# --- Save Model ---
model.save(MODEL_PATH)

# --- Reload with Custom Losses ---
custom_objects = {f"weighted_loss_trait_{i+1}": loss_map[f"trait_{i+1}"] for i in range(NUM_TRAITS)}
model = load_model(MODEL_PATH, custom_objects=custom_objects)

# --- Evaluation ---
print("\n📊 Enhanced Evaluation Report")
y_true = [np.argmax(y, axis=1) for y in y_val]
y_pred_raw = model.predict(val_gen, verbose=0)
y_pred = [np.argmax(y, axis=1) for y in y_pred_raw]

report_data = []
for i in range(NUM_TRAITS):
    f1 = f1_score(y_true[i], y_pred[i], average='macro')
    prec = precision_score(y_true[i], y_pred[i], average='macro')
    rec = recall_score(y_true[i], y_pred[i], average='macro')
    acc = np.mean(np.array(y_true[i]) == np.array(y_pred[i]))
    report_data.append([TRAIT_NAMES[i], f1, prec, rec, acc])
    print(f"\n🧠 {TRAIT_NAMES[i]}\n", classification_report(y_true[i], y_pred[i]))

report_df = pd.DataFrame(report_data, columns=['Trait', 'F1 Score', 'Precision', 'Recall', 'Accuracy'])
report_df.to_csv(os.path.join(MODEL_DIR, 'metrics_report.csv'), index=False)

# --- Combined Accuracy ---
total_correct = sum(np.sum(y_true[i] == y_pred[i]) for i in range(NUM_TRAITS))
total_preds = sum(len(y_true[i]) for i in range(NUM_TRAITS))
overall_accuracy = total_correct / total_preds
print(f"\n🔎 Overall combined accuracy across all traits: {overall_accuracy:.2%}")

# --- Plots ---
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