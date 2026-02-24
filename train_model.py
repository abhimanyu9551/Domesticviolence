import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# PATHS
# ----------------------------
DATA_PATH = "dataset_split/train"
VAL_PATH = "dataset_split/val"
TEST_PATH = "dataset_split/test"

SAMPLE_RATE = 16000
N_MELS = 128

# ----------------------------
# FEATURE EXTRACTION
# ----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Safe normalization
    mean = np.mean(mel_db)
    std = np.std(mel_db)

    mel_db = (mel_db - mean) / (std + 1e-6)

    # Extra safety: remove any NaNs
    mel_db = np.nan_to_num(mel_db)

    return mel_db
# ----------------------------
# LOAD DATA
# ----------------------------
def load_data(base_path):
    X = []
    y = []
    classes = os.listdir(base_path)

    for label in classes:
        class_path = os.path.join(base_path, label)

        for file in os.listdir(class_path):
            file_path = os.path.join(class_path, file)

            features = extract_features(file_path)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


print("Loading training data...")
X_train, y_train = load_data(DATA_PATH)

print("Loading validation data...")
X_val, y_val = load_data(VAL_PATH)

print("Loading test data...")
X_test, y_test = load_data(TEST_PATH)

# ----------------------------
# ENCODE LABELS
# ----------------------------
le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# ----------------------------
# COMPUTE CLASS WEIGHTS
# ----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# ----------------------------
# BUILD IMPROVED CNN MODEL
# ----------------------------
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=X_train.shape[1:]),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# TRAIN MODEL
# ----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_val, y_val),
    class_weight=class_weights
)

# ----------------------------
# EVALUATE MODEL
# ----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", test_acc)

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes,
                            target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("dv_model_v2.keras")

print("\nModel saved as dv_model_v2.keras")
