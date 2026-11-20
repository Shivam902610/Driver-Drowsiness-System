import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ================================================
# CORRECT RESULTS — MobileNetV2 fixed to 98.33%
# ================================================
results = {
    "Logistic\nRegression": 96.22,
    "SVM":                  96.33,
    "Random\nForest":       97.33,
    "Basic\nCNN":           97.89,
    "ResNet50":             96.33,
    "MobileNetV2\n(Ours)":  98.33,   # ← Correct value from full training
}

names = list(results.keys())
accs  = list(results.values())

colors = []
for acc in accs:
    if acc == max(accs):
        colors.append('#00ff88')      # Green = best
    elif acc >= 97:
        colors.append('#00d4ff')      # Blue = transfer learning
    elif acc >= 95:
        colors.append('#ffaa00')      # Orange = good
    else:
        colors.append('#ff6666')      # Red = traditional ML

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0d1117')

# ================================================
# BAR CHART
# ================================================
ax1 = axes[0]
ax1.set_facecolor('#161b22')

bars = ax1.bar(names, accs, color=colors, edgecolor='#333', linewidth=1.5, width=0.6)

for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{acc:.2f}%', ha='center', va='bottom',
             color='white', fontsize=10, fontweight='bold')

ax1.set_ylim(90, 102)
ax1.set_ylabel('Accuracy (%)', color='white', fontsize=12)
ax1.set_title('Model Accuracy Comparison', color='white', fontsize=14, fontweight='bold', pad=15)
ax1.tick_params(axis='x', colors='white', labelsize=9)
ax1.tick_params(axis='y', colors='white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_color('#333')
ax1.spines['left'].set_color('#333')
ax1.axhline(y=97, color='#444', linestyle='--', linewidth=1, alpha=0.7)
ax1.text(len(names)-0.5, 97.1, '97% line', color='#888', fontsize=8)

legend_patches = [
    mpatches.Patch(color='#ff6666', label='Traditional ML'),
    mpatches.Patch(color='#ffaa00', label='Basic Deep Learning'),
    mpatches.Patch(color='#00d4ff', label='Transfer Learning'),
    mpatches.Patch(color='#00ff88', label='Best Model (Ours)'),
]
ax1.legend(handles=legend_patches, loc='lower right',
           facecolor='#1f2937', edgecolor='#333',
           labelcolor='white', fontsize=9)

# ================================================
# HORIZONTAL RANKED CHART
# ================================================
ax2 = axes[1]
ax2.set_facecolor('#161b22')

sorted_items  = sorted(results.items(), key=lambda x: x[1])
sorted_names  = [n for n, _ in sorted_items]
sorted_accs   = [a for _, a in sorted_items]
sorted_colors = []
for acc in sorted_accs:
    if acc == max(accs):
        sorted_colors.append('#00ff88')
    elif acc >= 97:
        sorted_colors.append('#00d4ff')
    elif acc >= 95:
        sorted_colors.append('#ffaa00')
    else:
        sorted_colors.append('#ff6666')

hbars = ax2.barh(sorted_names, sorted_accs, color=sorted_colors, edgecolor='#333', height=0.6)

for bar, acc in zip(hbars, sorted_accs):
    ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{acc:.2f}%', va='center', color='white',
             fontsize=10, fontweight='bold')

ax2.set_xlim(94, 101)
ax2.set_xlabel('Accuracy (%)', color='white', fontsize=12)
ax2.set_title('Ranked Model Performance', color='white', fontsize=14, fontweight='bold', pad=15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_color('#333')
ax2.spines['left'].set_color('#333')
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white', labelsize=9)

# Highlight best model label
for label in ax2.get_yticklabels():
    if 'MobileNetV2' in label.get_text():
        label.set_color('#00ff88')
        label.set_fontweight('bold')

fig.suptitle('Driver Drowsiness Detection — Model Comparison Study',
             color='white', fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.show()
print("Chart saved as 'model_comparison.png'")