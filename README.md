Driver Drowsiness Detection System
A real-time driver drowsiness monitoring prototype built using deep learning and computer vision. The system analyses eye state and facial landmarks from a webcam feed, estimates signs of drowsiness, and displays the results through a Flask-based web interface.
Features
- Eye-state classification: Uses a trained MobileNetV2-based model to classify eye images as open or closed.
- Facial landmark detection: Uses MediaPipe to detect facial landmarks.
- Drowsiness indicators: Uses Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and PERCLOS-based eye-closure information.
- Live monitoring: Captures webcam frames using OpenCV.
- Web dashboard: Displays the camera feed and drowsiness-related information through a Flask application.
- Alerts and voice commands: Includes visual notifications and browser/application voice-command functionality, as implemented in the project.
- Driving simulation: Demonstrates different alertness states in a simulated environment. It does not control a real vehicle.
Technologies Used
- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- MediaPipe
- Flask
- NumPy and SciPy
- Scikit-learn
- HTML, CSS and JavaScript
- Chart.js
Project Files
The repository includes the application and supporting scripts, trained model files, visualisation outputs, and documentation. Important files include:
File	Purpose
app.py	Main Flask web application
train_model.py	Script for training the eye-state classification model
model_analysis.py	Model evaluation and visualisation script
detect_realtime.py	Real-time detection support script
voice_commands.py	Voice-command functionality
drowsiness_model.keras	Trained model used by the application
requirements.txt	Python package dependencies


Other model files and PNG outputs in the repository are retained for comparison, analysis, or visualisation.
Requirements
- Python 3.10 is recommended.
- A working webcam is required for live monitoring.
- Windows, Linux, or macOS with compatible Python packages.
Installation
1. Clone the repository:
   git clone https://github.com/Shivam902610/Driver-Drowsiness-System.git
   cd Driver-Drowsiness-System
2. (Recommended) Create and activate a virtual environment:
   Windows
   python -m venv venv
   venv\Scripts\activate
   Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
3. Install the dependencies:
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   Some audio or webcam-related packages may require operating-system-specific setup.
Running the Application
1. Make sure the trained model file drowsiness_model.keras is present in the location expected by app.py.
2. Connect or enable your webcam.
3. Start the application:
   python app.py
4. Open the local address printed in the terminal (commonly http://127.0.0.1:5000) in your browser.
5. Allow camera access if the browser requests permission, then use the dashboard controls to start monitoring.
If the application reports a missing module or model file, check that the dependency is installed and that the required file is in the expected path.
Dataset and Model Training
The training script is configured to load the eye-image dataset from a local Drowsiness_dataset directory, with open_eyes and close_eyes class folders.

Expected layout:
Drowsiness_dataset/
├── open_eyes/
└── close_eyes/
The training workflow resizes images to 64 × 64 pixels, converts them to RGB, normalises pixel values, and uses a stratified training/validation/test split. The MobileNetV2 base is first used with its pretrained layers frozen, followed by fine-tuning with a lower learning rate.

The dataset itself is not included in this repository. Place the dataset in the expected local folder before running train_model.py. Training is not required to run the application if the trained model file is already available.

To train the model:
python train_model.py
To run the analysis script:
python model_analysis.py
The analysis script expects the dataset and trained model at the paths configured inside that script.
Model Results
The project presentation reports 98.33% accuracy for MobileNetV2 in its model-comparison experiment. It also presents a separate evaluation on 600 images reporting 98.67% accuracy. These are results from separate reported evaluations and should not be interpreted as one experiment.
Limitations
- Results can be affected by lighting, camera quality, face position, and visibility of facial landmarks.
- The prototype is intended for demonstration and educational use.
- The driving simulation is visual only; the system does not operate vehicle controls or provide a certified safety function.
- Performance should be tested under varied real-world conditions before any practical safety use.
License
See the LICENSE file in this repository.