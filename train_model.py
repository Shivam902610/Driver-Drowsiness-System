import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# ================================================
# CONFIGURATION — matches your exact folder names
# ================================================
DATASET_PATH = "Drowsiness_dataset"
OPEN_FOLDER  = "open_eyes"
CLOSE_FOLDER = "close_eyes"
IMG_SIZE     = (64, 64)
BATCH_SIZE   = 32
EPOCHS       = 30
MAX_IMAGES   = 5000

print("=" * 50)
print("   DRIVER DROWSINESS DETECTION — TRAINING")
print("=" * 50)

def load_images_from_folder(folder_path, label, max_images):
    images = []
    labels = []
    files  = os.listdir(folder_path)
    files  = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".pgm"))]
    files  = files[:max_images]
    print(f"Loading {len(files)} images from '{os.path.basename(folder_path)}'...")
    for filename in files:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label)
    return images, labels

def load_dataset():
    open_path  = os.path.join(DATASET_PATH, OPEN_FOLDER)
    close_path = os.path.join(DATASET_PATH, CLOSE_FOLDER)
    if not os.path.exists(open_path):
        print(f"ERROR: Folder not found: {open_path}"); exit()
    if not os.path.exists(close_path):
        print(f"ERROR: Folder not found: {close_path}"); exit()
    open_imgs,  open_lbls  = load_images_from_folder(open_path,  label=0, max_images=MAX_IMAGES)
    close_imgs, close_lbls = load_images_from_folder(close_path, label=1, max_images=MAX_IMAGES)
    all_images = open_imgs  + close_imgs
    all_labels = open_lbls + close_lbls
    X = np.array(all_images, dtype="float32") / 255.0
    y = np.array(all_labels)
    print(f"\nDataset Summary:")
    print(f"  Open  (Alert)  : {np.sum(y == 0)} images")
    print(f"  Closed (Drowsy): {np.sum(y == 1)} images")
    print(f"  Total          : {len(X)} images")
    return X, y

def build_model():
    base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)
    print(f"\nModel built! Total params: {model.count_params():,}")
    return model, base_model

def train():
    X, y = load_dataset()
    X, y = shuffle(X, y, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    model, base_model = build_model()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy", verbose=1),
        ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7)
    ]

    print("\n" + "="*50)
    print("PHASE 1: Training custom head (base frozen)")
    print("="*50)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    history1 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

    print("\n" + "="*50)
    print("PHASE 2: Fine-tuning top layers")
    print("="*50)
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    history2 = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

    print("\n" + "="*50)
    print("FINAL TEST EVALUATION")
    print("="*50)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Accuracy: {acc*100:.2f}%")

    model.save("drowsiness_model.keras")
    print("Model saved as 'drowsiness_model.keras'")

    # Plot
    acc_all     = history1.history["accuracy"]     + history2.history["accuracy"]
    val_acc_all = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss_all    = history1.history["loss"]         + history2.history["loss"]
    val_loss_all= history1.history["val_loss"]     + history2.history["val_loss"]
    epochs_range = range(len(acc_all))

    plt.figure(figsize=(14, 5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc_all,     label="Train Accuracy", color="blue")
    plt.plot(epochs_range, val_acc_all, label="Val Accuracy",   color="orange")
    plt.axvline(x=len(history1.history["accuracy"]), color="gray", linestyle="--", label="Fine-tune start")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss_all,     label="Train Loss", color="blue")
    plt.plot(epochs_range, val_loss_all, label="Val Loss",   color="orange")
    plt.axvline(x=len(history1.history["loss"]), color="gray", linestyle="--", label="Fine-tune start")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()
    print("Graph saved as 'training_results.png'")

if __name__ == "__main__":
    train()