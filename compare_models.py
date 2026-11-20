import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import warnings
warnings.filterwarnings('ignore')

# ================================================
# CONFIGURATION
# ================================================
DATASET_PATH = "Drowsiness_dataset"
OPEN_FOLDER  = "open_eyes"
CLOSE_FOLDER = "close_eyes"
IMG_SIZE     = (64, 64)
MAX_IMAGES   = 3000
BATCH_SIZE   = 32

print("=" * 60)
print("   DRIVER DROWSINESS - FULL MODEL COMPARISON")
print("=" * 60)

# ================================================
# STEP 1 - LOAD DATASET
# ================================================
def load_dataset():
    print("\n[STEP 1] Loading dataset...")
    open_path  = os.path.join(DATASET_PATH, OPEN_FOLDER)
    close_path = os.path.join(DATASET_PATH, CLOSE_FOLDER)
    images, labels = [], []
    for folder, label, name in [(open_path, 0, "open_eyes"), (close_path, 1, "close_eyes")]:
        files = os.listdir(folder)
        files = [f for f in files if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".pgm"))]
        files = files[:MAX_IMAGES]
        print(f"  Loading {len(files)} images from '{name}'...")
        for filename in files:
            img = cv2.imread(os.path.join(folder, filename))
            if img is None: continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(label)
    X = np.array(images, dtype="float32") / 255.0
    y = np.array(labels)
    X, y = shuffle(X, y, random_state=42)
    print(f"  Total: {len(X)} images | Open: {np.sum(y==0)} | Closed: {np.sum(y==1)}")
    return X, y

X, y = load_dataset()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

X_train_flat   = X_train.reshape(len(X_train), -1)
X_test_flat    = X_test.reshape(len(X_test),  -1)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled  = scaler.transform(X_test_flat)

results = {}

# ================================================
# STEP 2 - LOGISTIC REGRESSION
# ================================================
print("\n" + "="*60)
print("[STEP 2] Training Logistic Regression...")
print("="*60)
start    = time.time()
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train_scaled, y_train)
lr_acc   = accuracy_score(y_test, lr_model.predict(X_test_scaled)) * 100
results["Logistic\nRegression"] = lr_acc
print(f"  Accuracy: {lr_acc:.2f}% | Time: {time.time()-start:.1f}s")

# ================================================
# STEP 3 - SVM
# ================================================
print("\n" + "="*60)
print("[STEP 3] Training SVM...")
print("  (This may take a few minutes...)")
print("="*60)
start     = time.time()
svm_size  = min(2000, len(X_train_scaled))
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_scaled[:svm_size], y_train[:svm_size])
svm_acc   = accuracy_score(y_test, svm_model.predict(X_test_scaled)) * 100
results["SVM"] = svm_acc
print(f"  Accuracy: {svm_acc:.2f}% | Time: {time.time()-start:.1f}s")

# ================================================
# STEP 4 - RANDOM FOREST
# ================================================
print("\n" + "="*60)
print("[STEP 4] Training Random Forest...")
print("="*60)
start    = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_flat, y_train)
rf_acc   = accuracy_score(y_test, rf_model.predict(X_test_flat)) * 100
results["Random\nForest"] = rf_acc
print(f"  Accuracy: {rf_acc:.2f}% | Time: {time.time()-start:.1f}s")

# ================================================
# STEP 5 - BASIC CNN
# ================================================
print("\n" + "="*60)
print("[STEP 5] Training Basic CNN (from scratch)...")
print("="*60)
start = time.time()
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=15, batch_size=BATCH_SIZE,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)
_, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
cnn_acc   *= 100
results["Basic\nCNN"] = cnn_acc
print(f"\n  Accuracy: {cnn_acc:.2f}% | Time: {time.time()-start:.1f}s")
cnn_model.save("basic_cnn_model.h5")

# ================================================
# STEP 6 - MOBILENETV2
# ================================================
print("\n" + "="*60)
print("[STEP 6] Training MobileNetV2 (Transfer Learning)...")
print("="*60)
start    = time.time()
base     = MobileNetV2(input_shape=(64,64,3), include_top=False, weights='imagenet')
base.trainable = False
x   = GlobalAveragePooling2D()(base.output)
x   = BatchNormalization()(x)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.5)(x)
x   = Dense(128, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)
mob_model = Model(inputs=base.input, outputs=out)
mob_model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
mob_model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=15, batch_size=BATCH_SIZE,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)
_, mob_acc = mob_model.evaluate(X_test, y_test, verbose=0)
mob_acc   *= 100
results["MobileNetV2\n(Ours)"] = mob_acc
print(f"\n  Accuracy: {mob_acc:.2f}% | Time: {time.time()-start:.1f}s")
mob_model.save("mobilenetv2_comparison.h5")

# ================================================
# STEP 7 - RESNET50 (replacing EfficientNet)
# ================================================
print("\n" + "="*60)
print("[STEP 7] Training ResNet50 (Transfer Learning)...")
print("="*60)
start    = time.time()
base2    = ResNet50(input_shape=(64,64,3), include_top=False, weights='imagenet')
base2.trainable = False
x2   = GlobalAveragePooling2D()(base2.output)
x2   = BatchNormalization()(x2)
x2   = Dense(256, activation='relu')(x2)
x2   = Dropout(0.5)(x2)
x2   = Dense(128, activation='relu')(x2)
x2   = Dropout(0.3)(x2)
out2 = Dense(1, activation='sigmoid')(x2)
res_model = Model(inputs=base2.input, outputs=out2)
res_model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
res_model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=15, batch_size=BATCH_SIZE,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)], verbose=1)
_, res_acc = res_model.evaluate(X_test, y_test, verbose=0)
res_acc   *= 100
results["ResNet50"] = res_acc
print(f"\n  Accuracy: {res_acc:.2f}% | Time: {time.time()-start:.1f}s")
res_model.save("resnet50_model.h5")

# ================================================
# STEP 8 - PRINT RESULTS
# ================================================
print("\n" + "="*60)
print("   FINAL COMPARISON RESULTS")
print("="*60)
sorted_results = sorted(results.items(), key=lambda x: x[1])
for name, acc in sorted_results:
    bar    = "█" * int(acc / 2)
    marker = " ← BEST!" if acc == max(results.values()) else ""
    clean  = name.replace("\n", " ")
    print(f"  {clean:<25} {acc:>6.2f}%  {bar}{marker}")
print("="*60)
best = max(results, key=results.get).replace("\n"," ")
print(f"  WINNER: {best} with {max(results.values()):.2f}%")
print("="*60)

# ================================================
# STEP 9 - CHARTS
# ================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0d1117')

names = list(results.keys())
accs  = list(results.values())
colors = ['#00ff88' if a == max(accs) else '#00d4ff' if a >= 90 else '#ffaa00' if a >= 80 else '#ff6666' for a in accs]

ax1 = axes[0]
ax1.set_facecolor('#161b22')
bars = ax1.bar(names, accs, color=colors, edgecolor='#333', linewidth=1.5, width=0.6)
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
ax1.set_ylim(50, 108)
ax1.set_ylabel('Accuracy (%)', color='white', fontsize=12)
ax1.set_title('Model Accuracy Comparison', color='white', fontsize=14, fontweight='bold', pad=15)
ax1.tick_params(axis='x', colors='white', labelsize=9)
ax1.tick_params(axis='y', colors='white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#333')
ax1.spines['left'].set_color('#333')
ax1.axhline(y=90, color='#444', linestyle='--', linewidth=1, alpha=0.7)
ax1.text(len(names)-0.5, 90.5, '90% line', color='#888', fontsize=8)
legend_patches = [
    mpatches.Patch(color='#ff6666', label='Traditional ML'),
    mpatches.Patch(color='#ffaa00', label='Basic Deep Learning'),
    mpatches.Patch(color='#00d4ff', label='Transfer Learning'),
    mpatches.Patch(color='#00ff88', label='Best Model'),
]
ax1.legend(handles=legend_patches, loc='lower right', facecolor='#1f2937',
           edgecolor='#333', labelcolor='white', fontsize=9)

ax2 = axes[1]
ax2.set_facecolor('#161b22')
sorted_names  = [n for n, _ in sorted(results.items(), key=lambda x: x[1])]
sorted_accs   = [a for _, a in sorted(results.items(), key=lambda x: x[1])]
sorted_colors = ['#00ff88' if a == max(accs) else '#00d4ff' if a >= 90 else '#ffaa00' if a >= 80 else '#ff6666' for a in sorted_accs]
hbars = ax2.barh(sorted_names, sorted_accs, color=sorted_colors, edgecolor='#333', height=0.6)
for bar, acc in zip(hbars, sorted_accs):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{acc:.2f}%', va='center', color='white', fontsize=10, fontweight='bold')
ax2.set_xlim(50, 110)
ax2.set_xlabel('Accuracy (%)', color='white', fontsize=12)
ax2.set_title('Ranked Model Performance', color='white', fontsize=14, fontweight='bold', pad=15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#333')
ax2.spines['left'].set_color('#333')
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white', labelsize=9)

fig.suptitle('Driver Drowsiness Detection — Model Comparison Study',
             color='white', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.show()
print("\nChart saved as 'model_comparison.png'")
print("All done!")