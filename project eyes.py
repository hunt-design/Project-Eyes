import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import urllib.request
import wave
import struct
import pygame  # Used to play the audio without freezing the camera

# ---------------- CONFIGURATION ----------------
MODEL_FILE = "face_landmarker.task"
ALARM_FILE = "siren.wav"
EAR_THRESHOLD = 0.22      
CLOSED_FRAMES = 15        
# -----------------------------------------------

# 1. GENERATE A REAL SIREN SOUND (No downloading required!)
if not os.path.exists(ALARM_FILE):
    print("Generating a loud two-tone siren...")
    sample_rate = 44100
    duration = 1.0 # 1-second loop
    with wave.open(ALARM_FILE, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        # Create a two-tone European police style siren
        for i in range(int(sample_rate * duration)):
            freq = 1200.0 if (i // (sample_rate // 2)) % 2 == 0 else 800.0
            value = int(32767.0 * math.sin(2.0 * math.pi * freq * i / sample_rate))
            wav_file.writeframesraw(struct.pack('<h', value))

# 2. Auto-download the MediaPipe AI model if missing
if not os.path.exists(MODEL_FILE):
    print(f"Downloading AI face model from Google... please wait.")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_FILE)

# 3. Initialize Pygame Mixer for the siren
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(ALARM_FILE)

# 4. Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)

# MediaPipe landmark indices for the eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def calculate_ear(landmarks, eye_indices):
    eye_pts = [landmarks[i] for i in eye_indices]
    v1 = euclidean_distance(eye_pts[1], eye_pts[5])
    v2 = euclidean_distance(eye_pts[2], eye_pts[4])
    h = euclidean_distance(eye_pts[0], eye_pts[3])
    return (v1 + v2) / (2.0 * h)

# Start Webcam
cap = cv2.VideoCapture(0)
frames_closed = 0
alarm_on = False

print("Starting Drowsiness Detector... Close your eyes to test the siren! Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            left_ear = calculate_ear(face_landmarks, LEFT_EYE)
            right_ear = calculate_ear(face_landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Display the EAR value on screen
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- TRIGGER ALARM LOGIC ---
            if avg_ear < EAR_THRESHOLD:
                frames_closed += 1
                if frames_closed >= CLOSED_FRAMES:
                    cv2.putText(frame, "WAKE UP!", (20, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    
                    # Play the generated siren continuously
                    if not alarm_on:
                        alarm_sound.play(-1) 
                        alarm_on = True
            else:
                frames_closed = 0
                # Stop the siren immediately when eyes open
                if alarm_on:
                    alarm_sound.stop()
                    alarm_on = False

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()