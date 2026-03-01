import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math

# 1. Load Models
pose_model = YOLO('yolov8n-pose.pt') 
# Using the LVIS model which has 1200+ classes including walking sticks
obj_model = YOLO('yolov8n-lvis.pt') 

# 2. Config & State
cap = cv2.VideoCapture(0)
history_velocity = {} 
history_angle = {}
FRAME_WINDOW = 10     
HUNCH_ANGLE_THRESHOLD = 20 # Degrees
GAIT_THRESHOLD = 15        # Speed threshold

# LVIS specific labels we want to trigger on
LVIS_VULNERABLE_LABELS = [
    'walking_stick', 'cane', 'crutch', 'wheelchair', 
    'walker', 'stroller', 'blind_cane'
]

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    trigger_extension = False
    
    # Draw Safety Zone
    cv2.rectangle(frame, (int(0.2*w), 0), (int(0.8*w), h), (255, 255, 255), 1)

    # --- STEP 1: LVIS Object-Based Triggers ---
    obj_results = obj_model.predict(frame, conf=0.25, verbose=False)[0] 
    for box in obj_results.boxes:
        label = obj_model.names[int(box.cls[0])]
        
        # Checking for specific LVIS mobility aid labels
        if label in LVIS_VULNERABLE_LABELS or label in ['chair', 'umbrella']: 
            trigger_extension = True
            b = box.xyxy[0].cpu().numpy()
            cv2.putText(frame, f"LVIS AID: {label}", (int(b[0]), int(b[1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)

    # --- STEP 2: Biometric Triggers (Hunch & Gait) ---
    pose_results = pose_model.predict(frame, conf=0.5, verbose=False)[0]
    if pose_results.keypoints is not None:
        kpts_list = pose_results.keypoints.xy.cpu().numpy()
        for i, kpts in enumerate(kpts_list):
            if len(kpts[0]) < 17: continue # Ensure skeleton is detected
            
            # Neck & Hip midpoints
            neck = (kpts[5] + kpts[6]) / 2  
            hips = (kpts[11] + kpts[12]) / 2 
            
            # Angle Calculation
            dx, dy = neck[0] - hips[0], hips[1] - neck[1]
            raw_angle = abs(math.degrees(math.atan2(dx, dy)))
            
            # Smoothing
            if i not in history_angle: history_angle[i] = deque(maxlen=5)
            history_angle[i].append(raw_angle)
            avg_angle = sum(history_angle[i]) / len(history_angle[i])

            # Velocity
            ankles = (kpts[15] + kpts[16]) / 2
            if i not in history_velocity: history_velocity[i] = deque(maxlen=FRAME_WINDOW)
            history_velocity[i].append(ankles[0])
            velocity = abs(history_velocity[i][-1] - history_velocity[i][0]) if len(history_velocity[i]) == FRAME_WINDOW else 0

            # Activation Zone Check
            if 0.2 * w < ankles[0] < 0.8 * w:
                if avg_angle > HUNCH_ANGLE_THRESHOLD or velocity < GAIT_THRESHOLD:
                    trigger_extension = True
                
                # Visual Indicator for Pose
                p_color = (0, 0, 255) if avg_angle > HUNCH_ANGLE_THRESHOLD else (0, 255, 0)
                cv2.circle(frame, (int(neck[0]), int(neck[1])), 5, p_color, -1)

    # --- STEP 3: System Output ---
    final_status = "EXTEND GREEN MAN" if trigger_extension else "NORMAL SIGNAL"
    s_color = (0, 0, 255) if trigger_extension else (0, 255, 0)
    cv2.rectangle(frame, (0,0), (w, 60), (0,0,0), -1) 
    cv2.putText(frame, final_status, (w//4, 40), 0, 1, s_color, 3)

    cv2.imshow("LTA Smart Guardian (LVIS Mode)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()