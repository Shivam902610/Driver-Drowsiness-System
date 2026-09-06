# Driver-Drowsiness-System
# 🚗 Driver Drowsiness Detection System

A real-time **Driver Drowsiness Detection System** that uses **Computer Vision and Deep Learning** to detect whether a driver's eyes are open or closed and identify potential drowsiness.

## 🔍 Overview

Driver fatigue is a major cause of road accidents. This project monitors the driver's eyes through a webcam and detects prolonged eye closure. When drowsiness is detected, the system triggers an alert.

## ✨ Features

* Real-time webcam monitoring
* Eye-state classification: **Open / Closed**
* Deep learning-based detection
* Drowsiness detection based on prolonged eye closure
* Alert notification system
* Model trained using transfer learning

## 🛠️ Tech Stack

* **Python**
* **OpenCV**
* **TensorFlow / Keras**
* **MobileNetV2**
* **NumPy**
* **Matplotlib**

## 🧠 Model

The project uses **MobileNetV2 with Transfer Learning** for eye-state classification.

```text
Webcam
   ↓
Eye Detection
   ↓
Image Preprocessing
   ↓
MobileNetV2
   ↓
Open / Closed
   ↓
Drowsiness Detection
   ↓
🚨 Alert
```

The model was trained in two phases:

1. **Feature Extraction** – pretrained MobileNetV2 layers are frozen.
2. **Fine-Tuning** – selected layers are unfrozen to improve performance.

### Training Configuration

* **Input Size:** 64 × 64 × 3
* **Optimizer:** Adam
* **Loss:** Binary Cross-Entropy
* **Output:** Open / Closed
* **Dataset:** MRL Eye Dataset
* **Images:** ~5,000 images per class

## 📂 Dataset Structure

```text
Drowsiness_dataset/
├── open_eyes/
└── close_eyes/
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run

Start the real-time detection system:

```bash
python detection.py
```

Make sure your system has a working webcam.

## 📊 Results

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

> Add your actual accuracy/result here once finalized.

## 🚀 Future Improvements

* Yawn detection
* Voice-based warning system
* Improved low-light detection
* Facial landmark-based fatigue analysis
* Mobile application integration
* Vehicle/fleet monitoring integration

## 👨‍💻 Author

**Shivam Singh**

B.Tech CSE — AI/ML

## 📜 License

This project is developed for **educational and research purposes**.

