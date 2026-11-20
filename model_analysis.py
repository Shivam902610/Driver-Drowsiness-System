import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ================================================
# CONFIGURATION
# ================================================
DATASET_PATH = "Drowsiness_dataset"
OPEN_FOLDER  = "open_eyes"
CLOSE_FOLDER = "close_eyes"
IMG_SIZE     = (64, 64)
MAX_IMAGES   = 2000
MODEL_PATH   = "drowsiness_model.keras"

print("=" * 60)
print("   DRIVER DROWSINESS - MODEL ANALYSIS")
print("=" * 60)

# ================================================
# STEP 1 - LOAD MODEL
# ================================================
print("\n[STEP 1] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"  Model loaded: {MODEL_PATH}")
print(f"  Total layers: {len(model.layers)}")

# ================================================
# STEP 2 - LOAD DATASET
# ================================================
print("\n[STEP 2] Loading dataset for analysis...")

def load_dataset():
    images, labels = [], []
    for folder, label, name in [
        (os.path.join(DATASET_PATH, OPEN_FOLDER),  0, "open_eyes"),
        (os.path.join(DATASET_PATH, CLOSE_FOLDER), 1, "close_eyes")
    ]:
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
    return shuffle(X, y, random_state=42)

X, y = load_dataset()
_, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(f"  Test set: {len(X_test)} images")

# ================================================
# STEP 3 - PREDICTIONS
# ================================================
print("\n[STEP 3] Running predictions on test set...")
y_pred_prob = model.predict(X_test, verbose=1)
y_pred      = (y_pred_prob >= 0.5).astype(int).flatten()
y_true      = y_test.astype(int)

acc = np.mean(y_pred == y_true) * 100
print(f"  Test Accuracy: {acc:.2f}%")

# ================================================
# STEP 4 - CONFUSION MATRIX
# ================================================
print("\n[STEP 4] Generating Confusion Matrix...")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"  True Negatives  (Open correctly):   {tn}")
print(f"  False Positives (Open as Closed):   {fp}")
print(f"  False Negatives (Closed as Open):   {fn}")
print(f"  True Positives  (Closed correctly): {tp}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0d1117')

# Heatmap
ax1 = axes[0]
ax1.set_facecolor('#161b22')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Open (Alert)', 'Closed (Drowsy)'],
            yticklabels=['Open (Alert)', 'Closed (Drowsy)'],
            ax=ax1, linewidths=2, linecolor='#0d1117',
            annot_kws={"size": 18, "weight": "bold", "color": "white"})
ax1.set_title('Confusion Matrix', color='white', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Predicted Label', color='white', fontsize=11)
ax1.set_ylabel('True Label', color='white', fontsize=11)
ax1.tick_params(colors='white')

# Metrics bar chart
metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
precision      = tp / (tp + fp) if (tp + fp) > 0 else 0
recall         = tp / (tp + fn) if (tp + fn) > 0 else 0
f1             = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0
metrics_values = [acc/100, precision, recall, f1, specificity]

ax2 = axes[1]
ax2.set_facecolor('#161b22')
colors = ['#00ff88', '#00d4ff', '#ffaa00', '#ff6b6b', '#a78bfa']
bars   = ax2.bar(metrics_names, metrics_values, color=colors, edgecolor='#333', width=0.6)

for bar, val in zip(bars, metrics_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom',
             color='white', fontsize=11, fontweight='bold')

ax2.set_ylim(0, 1.15)
ax2.set_title('Performance Metrics', color='white', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Score', color='white', fontsize=11)
ax2.tick_params(axis='x', colors='white', labelsize=9)
ax2.tick_params(axis='y', colors='white')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#333')
ax2.spines['left'].set_color('#333')
ax2.axhline(y=0.95, color='#444', linestyle='--', linewidth=1)
ax2.text(4.4, 0.955, '95%', color='#888', fontsize=8)

fig.suptitle('Model Evaluation — Driver Drowsiness Detection',
             color='white', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("  Saved: confusion_matrix.png")

# ================================================
# STEP 5 - CLASSIFICATION REPORT
# ================================================
print("\n[STEP 5] Classification Report...")
print("="*60)
report = classification_report(y_true, y_pred,
                                target_names=['Open (Alert)', 'Closed (Drowsy)'])
print(report)
print("="*60)

# Save report as image
fig2, ax = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.axis('off')

report_dict = classification_report(y_true, y_pred,
                                     target_names=['Open (Alert)', 'Closed (Drowsy)'],
                                     output_dict=True)

rows  = []
cols  = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
for key in ['Open (Alert)', 'Closed (Drowsy)', 'accuracy', 'macro avg', 'weighted avg']:
    if key == 'accuracy':
        rows.append(['Accuracy', '', '', f"{report_dict[key]:.4f}", f"{len(y_true)}"])
    else:
        d = report_dict[key]
        rows.append([key,
                     f"{d['precision']:.4f}" if 'precision' in d else '',
                     f"{d['recall']:.4f}"    if 'recall'    in d else '',
                     f"{d['f1-score']:.4f}"  if 'f1-score'  in d else '',
                     f"{int(d['support'])}"  if 'support'   in d else ''])

table = ax.table(cellText=rows, colLabels=cols,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.5, 2.2)

for (row, col), cell in table.get_celld().items():
    cell.set_facecolor('#1f2937' if row % 2 == 0 else '#161b22')
    cell.set_text_props(color='white')
    cell.set_edgecolor('#333')
    if row == 0:
        cell.set_facecolor('#00d4ff')
        cell.set_text_props(color='black', fontweight='bold')
    if col == 0 and row > 0:
        cell.set_text_props(color='#00ff88', fontweight='bold')

ax.set_title('Classification Report — Driver Drowsiness Detection',
             color='white', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('classification_report.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("  Saved: classification_report.png")

# ================================================
# STEP 6 - GRAD-CAM
# ================================================
print("\n[STEP 6] Generating Grad-CAM visualizations...")

def make_gradcam(img_array, model, last_conv_layer_name):
    grad_model = Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads      = tape.gradient(loss, conv_outputs)
    pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out   = conv_outputs[0]
    heatmap    = conv_out @ pooled[..., tf.newaxis]
    heatmap    = tf.squeeze(heatmap)
    heatmap    = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color   = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlaid        = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return overlaid, heatmap_color

# Find last conv layer
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

if last_conv_layer is None:
    # For MobileNetV2 use specific layer
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer.name
            break

print(f"  Using layer: {last_conv_layer}")

# Get sample images — open and closed
open_samples   = X_test[y_test == 0][:4]
closed_samples = X_test[y_test == 1][:4]
samples        = np.concatenate([open_samples, closed_samples])
sample_labels  = ['Open']*4 + ['Closed']*4

fig3, axes3 = plt.subplots(3, 8, figsize=(20, 8))
fig3.patch.set_facecolor('#0d1117')
fig3.suptitle('Grad-CAM — What the Model Looks At\n(Brighter = More Important Region)',
              color='white', fontsize=14, fontweight='bold')

row_titles = ['Original Image', 'Heatmap', 'Grad-CAM Overlay']

for i, (img, label) in enumerate(zip(samples, sample_labels)):
    img_array = np.expand_dims(img, axis=0)

    try:
        heatmap = make_gradcam(img_array, model, last_conv_layer)
        img_uint8 = np.uint8(255 * img)
        overlaid, heatmap_color = overlay_gradcam(img_uint8, heatmap)

        # Row 1 — original
        axes3[0, i].imshow(img)
        axes3[0, i].set_title(label, color='#00ff88' if label=='Open' else '#ff3333',
                               fontsize=10, fontweight='bold')
        axes3[0, i].axis('off')

        # Row 2 — heatmap
        axes3[1, i].imshow(heatmap_color)
        axes3[1, i].axis('off')

        # Row 3 — overlay
        axes3[2, i].imshow(overlaid)
        axes3[2, i].axis('off')

    except Exception as e:
        for row in range(3):
            axes3[row, i].axis('off')

for row, title in enumerate(row_titles):
    axes3[row, 0].set_ylabel(title, color='white', fontsize=10, rotation=90, labelpad=10)

for ax in axes3.flat:
    ax.set_facecolor('#0d1117')

plt.tight_layout()
plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("  Saved: gradcam_visualization.png")

# ================================================
# SUMMARY
# ================================================
print("\n" + "="*60)
print("   ANALYSIS COMPLETE!")
print("="*60)
print(f"  Test Accuracy : {acc:.2f}%")
print(f"  Precision     : {precision:.4f}")
print(f"  Recall        : {recall:.4f}")
print(f"  F1-Score      : {f1:.4f}")
print(f"  Specificity   : {specificity:.4f}")
print("="*60)
print("  Files saved:")
print("  - confusion_matrix.png")
print("  - classification_report.png")
print("  - gradcam_visualization.png")
print("="*60)