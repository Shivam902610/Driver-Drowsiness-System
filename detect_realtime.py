import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from datetime import datetime
from scipy.spatial import distance

# ================================================
# CONFIGURATION
# ================================================
MODEL_PATH = "drowsiness_model.keras"
IMG_SIZE         = (64, 64)
DROWSY_THRESHOLD = 0.6

# EAR threshold — below this = eye closing
EAR_THRESHOLD    = 0.25
# MAR threshold — above this = yawning
MAR_THRESHOLD    = 0.6
# Frames before alert triggers
ALERT_FRAMES     = 20

# MediaPipe landmark indices
# Left eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
# Mouth
MOUTH     = [61, 291, 0, 17, 269, 405]

# ================================================
# LOAD MODEL
# ================================================
print("Loading drowsiness model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!\n")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh    = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ================================================
# HELPER FUNCTIONS
# ================================================
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Calculate EAR — lower = more closed"""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    ear = (A + B) / (2.0 * C)
    return ear, pts

def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    """Calculate MAR — higher = more open (yawning)"""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    A = distance.euclidean(pts[2], pts[5])
    B = distance.euclidean(pts[0], pts[1])
    mar = A / B
    return mar, pts

def preprocess_eye(eye_img):
    eye_img = cv2.resize(eye_img, IMG_SIZE)
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    eye_img = eye_img.astype("float32") / 255.0
    return np.expand_dims(eye_img, axis=0)

def predict_eye(eye_img):
    pred = model.predict(preprocess_eye(eye_img), verbose=0)[0][0]
    return "CLOSED" if pred >= DROWSY_THRESHOLD else "OPEN"

def play_alert():
    try:
        import winsound
        winsound.Beep(1000, 600)
    except:
        print("ALERT! DROWSINESS DETECTED!")

def get_score_color(score):
    if score < 40:
        return (0, 255, 0)      # Green
    elif score < 70:
        return (0, 165, 255)    # Orange
    else:
        return (0, 0, 255)      # Red

# ================================================
# MAIN DETECTION LOOP
# ================================================
def run_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!"); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # State
    consec_eye_closed = 0
    consec_yawn       = 0
    alert_triggered   = False
    total_alerts      = 0
    session_start     = time.time()
    drowsiness_score  = 0
    ear_history       = []

    print("Webcam started! Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame    = cv2.flip(frame, 1)
        h, w     = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = face_mesh.process(rgb)

        face_found   = False
        eye_status   = "OPEN"
        yawn_status  = "NO YAWN"
        ear_avg      = 0.0
        mar_val      = 0.0

        if results.multi_face_landmarks:
            face_found = True
            lm = results.multi_face_landmarks[0].landmark

            # --- EAR Calculation ---
            left_ear,  left_pts  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            right_ear, right_pts = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear_avg = (left_ear + right_ear) / 2.0
            ear_history.append(ear_avg)
            if len(ear_history) > 30:
                ear_history.pop(0)

            # --- MAR Calculation ---
            mar_val, mouth_pts = mouth_aspect_ratio(lm, MOUTH, w, h)

            # --- Draw eye landmarks ---
            for pt in left_pts + right_pts:
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)
            for pt in mouth_pts:
                cv2.circle(frame, pt, 2, (255, 165, 0), -1)

            # --- Eye Status ---
            if ear_avg < EAR_THRESHOLD:
                eye_status = "CLOSED"
                consec_eye_closed += 1
            else:
                eye_status = "OPEN"
                consec_eye_closed = max(0, consec_eye_closed - 2)

            # --- Yawn Status ---
            if mar_val > MAR_THRESHOLD:
                yawn_status = "YAWNING"
                consec_yawn += 1
            else:
                yawn_status = "NO YAWN"
                consec_yawn = max(0, consec_yawn - 1)

            # --- Model Prediction on eye region ---
            # Crop left eye region for model
            left_eye_pts = np.array(left_pts)
            lx1 = max(0, left_eye_pts[:,0].min() - 10)
            ly1 = max(0, left_eye_pts[:,1].min() - 10)
            lx2 = min(w, left_eye_pts[:,0].max() + 10)
            ly2 = min(h, left_eye_pts[:,1].max() + 10)
            eye_roi = frame[ly1:ly2, lx1:lx2]
            if eye_roi.size > 0:
                model_prediction = predict_eye(eye_roi)
                # Combine EAR + model for better accuracy
                if model_prediction == "CLOSED" and eye_status == "CLOSED":
                    eye_status = "CLOSED"  # Both agree = definitely closed
                elif model_prediction == "OPEN" and ear_avg > EAR_THRESHOLD:
                    eye_status = "OPEN"    # Both agree = definitely open

            # --- PERCLOS Score (% eye closure over last 30 frames) ---
            if len(ear_history) > 0:
                perclos = sum(1 for e in ear_history if e < EAR_THRESHOLD) / len(ear_history) * 100
            else:
                perclos = 0

            # --- Combined Drowsiness Score ---
            eye_score  = min(consec_eye_closed / ALERT_FRAMES, 1.0) * 50
            yawn_score = min(consec_yawn / 15, 1.0) * 25
            perc_score = min(perclos / 50, 1.0) * 25
            drowsiness_score = int(eye_score + yawn_score + perc_score)

        else:
            # No face — slowly decrease score
            drowsiness_score = max(0, drowsiness_score - 1)
            consec_eye_closed = max(0, consec_eye_closed - 1)

        # --- ALERT ---
        if drowsiness_score >= 70 and not alert_triggered:
            play_alert()
            alert_triggered = True
            total_alerts   += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ALERT #{total_alerts} | Score: {drowsiness_score}")

        if drowsiness_score < 40:
            alert_triggered = False

        # ================================================
        # DRAW HUD
        # ================================================
        score_color = get_score_color(drowsiness_score)

        # Top bar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,70), (15,15,15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Status
        if drowsiness_score >= 70:
            status_text = "DROWSY"
        elif drowsiness_score >= 40:
            status_text = "TIRED"
        else:
            status_text = "ALERT"

        cv2.putText(frame, f"STATUS: {status_text}",
                    (15, 48), cv2.FONT_HERSHEY_DUPLEX, 1.3, score_color, 2)

        # Session info
        elapsed     = int(time.time() - session_start)
        mins, secs  = divmod(elapsed, 60)
        cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}",
                    (w-200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
        cv2.putText(frame, f"Alerts: {total_alerts}",
                    (w-200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)

        # Drowsiness score bar
        bx, by, bw, bh = 15, h-55, 300, 22
        cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (40,40,40), -1)
        fill = int((drowsiness_score/100) * bw)
        cv2.rectangle(frame, (bx,by), (bx+fill, by+bh), score_color, -1)
        cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (150,150,150), 1)
        cv2.putText(frame, f"Drowsiness Score: {drowsiness_score}%",
                    (bx, by-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        # Factor indicators
        eye_col  = (0,255,0) if eye_status  == "OPEN"    else (0,0,255)
        yawn_col = (0,255,0) if yawn_status == "NO YAWN" else (0,165,255)
        cv2.putText(frame, f"Eyes : {eye_status}  (EAR:{ear_avg:.2f})",
                    (w-280, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_col, 1)
        cv2.putText(frame, f"Yawn : {yawn_status}  (MAR:{mar_val:.2f})",
                    (w-280, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_col, 1)

        # Warning flash
        if drowsiness_score >= 70:
            flash = frame.copy()
            cv2.rectangle(flash, (0,0), (w,h), (0,0,180), -1)
            cv2.addWeighted(flash, 0.2, frame, 0.8, 0, frame)
            cv2.putText(frame, "! WAKE UP !",
                        (w//2-160, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,255), 3)

        if not face_found:
            cv2.putText(frame, "No Face Detected",
                        (w//2-130, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100,100,255), 2)

        cv2.imshow("Driver Drowsiness Detection | Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed    = int(time.time() - session_start)
    mins, secs = divmod(elapsed, 60)
    print(f"\n{'='*40}")
    print(f"SESSION SUMMARY")
    print(f"{'='*40}")
    print(f"Total Drive Time : {mins:02d}:{secs:02d}")
    print(f"Total Alerts     : {total_alerts}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    run_detection()