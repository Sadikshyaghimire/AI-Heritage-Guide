import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -----------------------------
# SETTINGS
# -----------------------------

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

TRAIN_DIR = "dataset/final_dataset/train"
VAL_DIR = "dataset/final_dataset/val"
TEST_DIR = "dataset/final_dataset/test"

BASE_MODEL_PATH = "model_training/saved_models/heritage_efficientnet.keras"
FINETUNED_MODEL_PATH = "model_training/saved_models/heritage_efficientnet_finetuned.keras"

LABELS_SAVE_PATH = "model_training/saved_models/class_names.json"

# -----------------------------
# LOAD DATASETS
# -----------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

with open(LABELS_SAVE_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

# -----------------------------
# PERFORMANCE
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# -----------------------------
# MODEL
# -----------------------------

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# -----------------------------
# Fine-tuning
# -----------------------------

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])

# Load previously trained weights

model.load_weights(BASE_MODEL_PATH)
print("Loaded pretrained EfficientNet model.")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    ModelCheckpoint(
        FINETUNED_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

]

# -----------------------------
# TRAIN
# -----------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------
# SAVE MODEL
# -----------------------------

print(f"Best model already saved to {FINETUNED_MODEL_PATH}")

model = tf.keras.models.load_model(FINETUNED_MODEL_PATH)

# -----------------------------
# PLOT ACCURACY
# -----------------------------

plt.figure()

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.title("Training vs Validation Accuracy")

plt.savefig("model_training/evaluation/accuracy_plot.png")

plt.close()

# -----------------------------
# PLOT LOSS
# -----------------------------

plt.figure()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.title("Training vs Validation Loss")

plt.savefig("model_training/evaluation/loss_plot.png")

plt.close()

# -----------------------------
# TEST
# -----------------------------

test_loss, test_acc = model.evaluate(test_ds)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(images)

    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
)

print(report)

with open(
    "model_training/evaluation/classification_report.txt",
    "w"
) as f:
    f.write(report)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "model_training/evaluation/confusion_matrix.png"
)

plt.close()

print("Evaluation files saved in model_training/evaluation/")