# Driver Drowsiness Detection System

An AI-powered driver drowsiness detection prototype that combines deep learning and computer vision to monitor driver alertness in real time. The system analyses eye state and facial landmarks through a webcam and displays drowsiness information on a Flask-based dashboard.

## Features

- **Eye State Classification:** Uses a trained MobileNetV2-based model to classify eye images as open or closed.
- **Facial Landmark Detection:** Uses MediaPipe Face Mesh to detect facial landmarks.
- **Drowsiness Indicators:** Uses Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and eye-closure information to assess driver alertness.
- **Real-Time Monitoring:** Captures webcam frames using OpenCV.
- **Web Dashboard:** Displays the camera feed, driver status, drowsiness score, session statistics, and alert history.
- **Alert System:** Provides visual notifications when drowsiness is detected.
- **Voice Commands:** Supports browser/application voice-control functionality.
- **Driving Simulation:** Demonstrates vehicle behaviour and simulated responses to different alertness states. It does not control a real vehicle.
- **Model Analysis:** Includes model comparison, confusion matrix, classification report, and Grad-CAM visualization.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- MediaPipe
- OpenCV
- Flask
- Flask-SocketIO
- NumPy
- SciPy
- Scikit-learn
- HTML, CSS, JavaScript
- Chart.js

## Model Information

The project uses MobileNetV2 with transfer learning for eye-state classification.

- Dataset: MRL Eye Dataset
- Classes: Open Eyes and Closed Eyes
- Input Image Size: 64 × 64 pixels
- Preprocessing: Image resizing, RGB conversion, and normalization
- Output: Eye-state prediction

The dashboard currently displays a reported accuracy of 98.33%. Refer to the model evaluation files for the corresponding experimental results.

## Project Structure

```text
Driver-Drowsiness-System/
│
├── app.py
├── train_model.py
├── model_analysis.py
├── compare_models.py
├── detect_realtime.py
├── voice_commands.py
│
├── drowsiness_model.keras
├── best_model.keras
├── resnet50_model.h5
├── basic_cnn_model.h5
├── efficientnet_model.keras
│
├── templates/
│   ├── index.html
│   └── simulation.html
│
├── confusion_matrix.png
├── classification_report.png
├── gradcam_visualization.png
├── model_comparison.png
├── training_results.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shivam902610/Driver-Drowsiness-System.git
```

### 2. Navigate to the project directory

```bash
cd Driver-Drowsiness-System
```

### 3. Create a virtual environment

```bash
python -m venv drowsiness_env
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.\drowsiness_env\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

### 7. Open the dashboard

Visit:

```text
http://localhost:5000
```

Allow webcam access if prompted.

## Usage

1. Start the Flask application.
2. Open the dashboard in your browser.
3. Click **Start** to begin drowsiness monitoring.
4. Observe the live webcam feed and detection indicators.
5. Enable voice commands if required.
6. Open the driving simulation to see the simulated response to driver alertness states.

## Important Notes

- The trained model and dataset must be available at the paths expected by the application.
- The dataset is not included in this repository.
- Webcam access is required for real-time monitoring.
- The driving simulation is a demonstration only. It does not interact with or control a physical vehicle.
- This project is an academic prototype and is not a certified automotive safety system.

## GitHub Repository

[Driver Drowsiness Detection System](https://github.com/Shivam902610/Driver-Drowsiness-System)

## Author

**Shivam Singh**