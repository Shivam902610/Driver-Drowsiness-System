from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import threading
from datetime import datetime
from scipy.spatial import distance
from voice_commands import VoiceCommandHandler

app = Flask(__name__)

# ================================================
# CONFIGURATION
# ================================================
MODEL_PATH       = "drowsiness_model.keras"
IMG_SIZE         = (64, 64)
DROWSY_THRESHOLD = 0.6
EAR_THRESHOLD    = 0.25
MAR_THRESHOLD    = 0.6
ALERT_FRAMES     = 20

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 0, 17, 269, 405]

# ================================================
# LOAD MODEL
# ================================================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ================================================
# GLOBAL STATE
# ================================================
def reset_state():
    return {
        "running":        False,
        "status":         "STANDBY",
        "score":          0,
        "eye_status":     "---",
        "yawn_status":    "---",
        "alerts":         0,
        "alert_log":      [],
        "session_time":   0,
        "ear":            0.0,
        "mar":            0.0,
        "voice_command":  None,
        "alert_override": False,
        "voice_active":   False
    }

state             = reset_state()
cap               = None
session_start     = None
consec_eye_closed = 0
consec_yawn       = 0
alert_triggered   = False
ear_history       = []
camera_thread     = None
voice_handler     = None
latest_frame      = None
frame_lock        = threading.Lock()

# ================================================
# HELPER FUNCTIONS
# ================================================
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C), pts

def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    A = distance.euclidean(pts[2], pts[5])
    B = distance.euclidean(pts[0], pts[1])
    return A / B, pts

def preprocess_eye(eye_img):
    eye_img = cv2.resize(eye_img, IMG_SIZE)
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    eye_img = eye_img.astype("float32") / 255.0
    return np.expand_dims(eye_img, axis=0)

def predict_eye(eye_img):
    pred = model.predict(preprocess_eye(eye_img), verbose=0)[0][0]
    return "CLOSED" if pred >= DROWSY_THRESHOLD else "OPEN"

# ================================================
# FRAME PROCESSING
# ================================================
def process_frame(frame):
    global consec_eye_closed, consec_yawn, alert_triggered, ear_history

    # Handle voice command alert override
    if state.get("alert_override"):
        state["alert_override"] = False
        alert_triggered = False
        consec_eye_closed = 0
        consec_yawn = 0
        state["score"] = 0

    h, w    = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    face_found = False
    ear_avg    = 0.0
    mar_val    = 0.0

    if results.multi_face_landmarks:
        face_found = True
        lm = results.multi_face_landmarks[0].landmark

        left_ear,  left_pts  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        right_ear, right_pts = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear_avg = (left_ear + right_ear) / 2.0
        ear_history.append(ear_avg)
        if len(ear_history) > 30:
            ear_history.pop(0)

        mar_val, mouth_pts = mouth_aspect_ratio(lm, MOUTH, w, h)

        for pt in left_pts + right_pts:
            cv2.circle(frame, pt, 2, (0, 255, 255), -1)
        for pt in mouth_pts:
            cv2.circle(frame, pt, 2, (255, 165, 0), -1)

        if ear_avg < EAR_THRESHOLD:
            state["eye_status"] = "CLOSED"
            consec_eye_closed += 1
        else:
            state["eye_status"] = "OPEN"
            consec_eye_closed = max(0, consec_eye_closed - 2)

        if mar_val > MAR_THRESHOLD:
            state["yawn_status"] = "YAWNING"
            consec_yawn += 1
        else:
            state["yawn_status"] = "NO YAWN"
            consec_yawn = max(0, consec_yawn - 1)

        perclos    = sum(1 for e in ear_history if e < EAR_THRESHOLD) / max(len(ear_history), 1) * 100
        eye_score  = min(consec_eye_closed / ALERT_FRAMES, 1.0) * 50
        yawn_score = min(consec_yawn / 15, 1.0) * 25
        perc_score = min(perclos / 50, 1.0) * 25
        state["score"] = int(eye_score + yawn_score + perc_score)
        state["ear"]   = round(ear_avg, 3)
        state["mar"]   = round(mar_val, 3)
    else:
        state["score"] = max(0, state["score"] - 1)

    if state["score"] >= 70:
        state["status"] = "DROWSY"
    elif state["score"] >= 40:
        state["status"] = "TIRED"
    else:
        state["status"] = "ALERT"

    if state["score"] >= 70 and not alert_triggered:
        alert_triggered = True
        state["alerts"] += 1
        ts = datetime.now().strftime("%H:%M:%S")
        state["alert_log"].append(f"{ts} — Alert #{state['alerts']}")
        if len(state["alert_log"]) > 10:
            state["alert_log"].pop(0)

    if state["score"] < 40:
        alert_triggered = False

    if session_start:
        state["session_time"] = int(time.time() - session_start)

    color = (0,255,0) if state["status"]=="ALERT" else (0,165,255) if state["status"]=="TIRED" else (0,0,255)
    cv2.putText(frame, f"STATUS: {state['status']}", (15,40), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Score: {state['score']}%", (15,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if state["score"] >= 70:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,180), -1)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.putText(frame, "! WAKE UP !", (w//2-130, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0,0,255), 3)

    if not face_found:
        cv2.putText(frame, "No Face Detected", (w//2-130, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100,100,255), 2)

    # Voice active indicator
    if state.get("voice_active"):
        cv2.circle(frame, (w-30, 30), 12, (0, 255, 0), -1)
        cv2.putText(frame, "MIC", (w-55, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    return frame

def camera_loop():
    global cap, session_start, latest_frame
    global consec_eye_closed, consec_yawn, alert_triggered, ear_history

    consec_eye_closed = 0
    consec_yawn       = 0
    alert_triggered   = False
    ear_history       = []

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    session_start = time.time()

    while state["running"]:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame = process_frame(frame)
        with frame_lock:
            latest_frame = frame.copy()

    cap.release()
    with frame_lock:
        latest_frame = None

def generate_frames():
    while True:
        if not state["running"]:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Click START to begin", (150, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100,100,100), 2)
            cv2.putText(blank, "Say 'Start Session' to begin with voice",
                        (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70,70,70), 1)
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

# ================================================
# ROUTES
# ================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global camera_thread, state
    if not state["running"]:
        voice_active = state.get("voice_active", False)
        state = reset_state()
        state["running"]      = True
        state["voice_active"] = voice_active
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
        print("Detection started!")
    return jsonify({"message": "Started", "running": True})

@app.route('/stop')
def stop():
    state["running"] = False
    print("Detection stopped!")
    return jsonify({"message": "Stopped", "running": False})

@app.route('/start_voice')
def start_voice():
    global voice_handler
    if not state.get("voice_active"):
        voice_handler = VoiceCommandHandler(state)
        voice_handler.start()
        state["voice_active"] = True
        return jsonify({"message": "Voice activated", "voice": True})
    return jsonify({"message": "Already active", "voice": True})

@app.route('/stop_voice')
def stop_voice():
    global voice_handler
    voice_handler = None
    state["voice_active"] = False
    return jsonify({"message": "stopped", "voice": False})


@app.route('/stats')
def stats():
    # Handle voice commands from dashboard
    cmd = state.get("voice_command")
    if cmd == "start" and not state["running"]:
        state["voice_command"] = None
    elif cmd == "stop" and state["running"]:
        state["running"] = False
        state["voice_command"] = None

    elapsed    = state["session_time"]
    mins, secs = divmod(elapsed, 60)
    return jsonify({
        "running":      state["running"],
        "status":       state["status"],
        "score":        state["score"],
        "eye_status":   state["eye_status"],
        "yawn_status":  state["yawn_status"],
        "alerts":       state["alerts"],
        "alert_log":    state["alert_log"],
        "session_time": f"{mins:02d}:{secs:02d}",
        "ear":          state["ear"],
        "mar":          state["mar"],
        "voice_active": state.get("voice_active", False)
    })

@app.route('/simulation')
def simulation():
    return render_template('simulation.html')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  DRIVER DROWSINESS DETECTION - WEB APP")
    print("="*50)
    print("  Open browser: http://localhost:5000")
    print("  Voice commands available!")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)